"""Synchronous WebSocket client for the survev CPC bridge (protocol v0).

One connection, one request -> one response, in order. The client is thread-unsafe by
design (use one client per env worker). Uses ``websockets.sync.client`` so it works on
Windows without an event loop and without asyncio plumbing in the training code.
"""

from __future__ import annotations

import inspect
import socket
from typing import Any, Mapping, Sequence

from websockets.exceptions import ConnectionClosed, InvalidURI, WebSocketException
from websockets.sync.client import connect

from .protocol import (
    DEFAULT_BRIDGE_URL,
    DEFAULT_SCENARIO,
    CpcAction,
    ObsMessage,
    ProtocolError,
    decode_message,
    encode_message,
    make_close,
    make_reset,
    make_reset_options,
    make_step,
    make_step_batch,
    parse_batch_response,
    parse_response,
)


class BridgeError(RuntimeError):
    """Connection-level failure talking to the bridge (timeouts, closed socket, refused)."""


class BridgeClient:
    """Blocking client: ``reset`` / ``step`` / ``step_batch`` / ``close``.

    Parameters
    ----------
    url:              ``ws://host:port`` of the bridge (``pnpm cpc:bridge`` or the mock).
    connect_timeout:  seconds to wait for the WebSocket handshake.
    timeout:          seconds to wait for each response (``None`` = wait forever).
    validate:         run the observation key allowlist (M6) on responses: every response
                      for the first ``validate_warmup`` responses, then every
                      ``validate_every``-th one (validation costs ~0.2 ms per agent obs).
    """

    def __init__(
        self,
        url: str = DEFAULT_BRIDGE_URL,
        connect_timeout: float = 10.0,
        timeout: float | None = 60.0,
        validate: bool = True,
        validate_warmup: int = 50,
        validate_every: int = 20,
    ) -> None:
        self.url = url
        self.timeout = timeout
        self.validate = validate
        self.validate_warmup = int(validate_warmup)
        self.validate_every = max(1, int(validate_every))
        self._responses = 0
        self._ws = None
        self._env_ids: set[int] = set()
        kwargs: dict[str, Any] = {
            "open_timeout": connect_timeout,
            "close_timeout": 5.0,
            "max_size": None,
            "compression": None,
        }
        params = inspect.signature(connect).parameters
        if "proxy" in params:  # websockets >= 14 would otherwise honour *_proxy env vars
            kwargs["proxy"] = None
        if "legacy" in params:  # websockets >= 16 warns unless connect() is a context manager
            kwargs["legacy"] = True
        try:
            self._ws = connect(url, **kwargs)
        except (OSError, InvalidURI, WebSocketException, TimeoutError) as exc:
            raise BridgeError(
                f"could not connect to bridge at {url}: {exc}. Is `pnpm cpc:bridge` running "
                f"(or pass --mock)?"
            ) from exc

    # -- context manager -----------------------------------------------------------------
    def __enter__(self) -> "BridgeClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.disconnect()

    @property
    def connected(self) -> bool:
        return self._ws is not None

    @property
    def env_ids(self) -> frozenset[int]:
        return frozenset(self._env_ids)

    # -- raw request/response -------------------------------------------------------------
    def request(self, message: Mapping[str, Any]) -> dict[str, Any]:
        """Send one JSON message and return the decoded JSON response."""
        if self._ws is None:
            raise BridgeError("client is disconnected")
        try:
            self._ws.send(encode_message(message))
            raw = self._ws.recv(timeout=self.timeout)
        except TimeoutError as exc:
            raise BridgeError(
                f"bridge did not answer a {message.get('type')!r} message within {self.timeout}s"
            ) from exc
        except ConnectionClosed as exc:
            self._ws = None
            raise BridgeError(f"bridge connection closed: {exc}") from exc
        except (OSError, socket.timeout, WebSocketException) as exc:
            raise BridgeError(f"bridge transport error: {exc}") from exc
        return decode_message(raw)

    def _should_validate(self) -> bool:
        self._responses += 1
        if not self.validate:
            return False
        if self._responses <= self.validate_warmup:
            return True
        return self._responses % self.validate_every == 0

    # -- protocol methods -------------------------------------------------------------------
    def reset(
        self,
        env_id: int,
        scenario: str = DEFAULT_SCENARIO,
        seed: str | int = "cpc-duo2v2-seed-0",
        options: Mapping[str, Any] | None = None,
        *,
        controlled: Sequence[str] | None = None,
        scripted: str | None = None,
        time_limit: float | None = None,
        map_size: int | None = None,
    ) -> ObsMessage:
        """Create/replace ``env_id`` and return the ``t = 0`` observation.

        Either pass a full ``options`` dict (spec shape) or the keyword shortcuts, which are
        merged on top of ``make_reset_options`` defaults.
        """
        opts = make_reset_options()
        if options:
            opts.update(options)
        if controlled is not None:
            opts["controlled"] = list(controlled)
        if scripted is not None:
            opts["scripted"] = scripted
        if time_limit is not None:
            opts["timeLimit"] = float(time_limit)
        if map_size is not None:
            opts["mapSize"] = int(map_size)
        response = self.request(make_reset(env_id, scenario, seed, opts))
        msg = parse_response(response, validate=self._should_validate())
        self._env_ids.add(int(env_id))
        return msg

    def step(
        self, env_id: int, actions: Mapping[str, CpcAction | Mapping[str, Any]], ticks: int = 10
    ) -> ObsMessage:
        response = self.request(make_step(env_id, actions, ticks))
        return parse_response(response, validate=self._should_validate())

    def step_batch(
        self,
        envs: Mapping[int, tuple[Mapping[str, CpcAction | Mapping[str, Any]], int]],
    ) -> dict[int, ObsMessage]:
        """Step several envs in one round trip: ``{env_id: (actions, ticks)}``."""
        if not envs:
            return {}
        response = self.request(make_step_batch(envs))
        return parse_batch_response(response, validate=self._should_validate())

    def close(self, env_id: int) -> None:
        response = self.request(make_close(env_id))
        if response.get("type") == "error":
            raise ProtocolError(f"close failed for env {env_id}: {response.get('message')}")
        if response.get("type") != "closed":
            raise ProtocolError(f"unexpected response to close: {response.get('type')!r}")
        self._env_ids.discard(int(env_id))

    def close_all(self) -> None:
        for env_id in sorted(self._env_ids):
            try:
                self.close(env_id)
            except (BridgeError, ProtocolError):
                pass
        self._env_ids.clear()

    def disconnect(self) -> None:
        """Close the WebSocket; the server closes every env of this connection."""
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                ws.close()
            except Exception:  # pragma: no cover - best effort
                pass
        self._env_ids.clear()
