"""Vectorized training environment on top of the bridge (no gymnasium / torchrl).

``SurvevVecEnv`` drives ``n_envs`` bridge envs over one connection with the batched step
message. Every controlled agent of every env is one *row* (row order: env-major, then the
``controlled`` order), so the policy is shared across controlled agents; when more than one
agent is controlled an agent-id one-hot is appended to the observation.

* ``reset() -> obs[n_rows, obs_dim]``
* ``step(actions[n_rows, n_action_dims]) -> (obs, rewards[n_rows], dones[n_rows], infos)``
  with auto-reset: when an env finishes, its rows get ``done=True`` and the returned obs
  rows already belong to the next episode. ``infos[row]`` then carries ``episode_metrics``
  (the agent's metric dict from ``info.metrics``), ``episode_return``, ``episode_length``,
  ``winner_team``, ``reason`` and ``team_win``. Every step, ``infos[row]["active"]`` says
  whether the agent is still alive (dead agents keep producing rows until the env ends).

``SurvevSingleEnv`` is the one-env / one-agent convenience wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .actions import ActionSpace
from .bridge_client import BridgeClient
from .featurizer import AgentMemory, Featurizer, FeaturizerConfig
from .protocol import DEFAULT_BRIDGE_URL, DEFAULT_SCENARIO, ObsMessage
from .rewards import RewardConfig, compute_reward_breakdown, has_gun

SeedFn = Callable[[int, int], str | int]


@dataclass
class EnvConfig:
    bridge_url: str = DEFAULT_BRIDGE_URL
    n_envs: int = 1
    controlled: tuple[str, ...] = ("team-a-0", "team-a-1")
    ticks: int = 10
    scenario: str = DEFAULT_SCENARIO
    scripted: str = "chaser"
    time_limit: float = 60.0
    map_size: int = 128
    loadout: str = "fists"  # "armed" = spawn with an ak47 (curriculum; mock + bridge extension)
    layout: str = "fixed"  # "random" = seeded spawn rotation/distance (bridge option, see survev-bridge-v0.md)
    goal: tuple[float, float] | None = None  # fixed waypoint for the goal block / waypoint rewards (world x, y)
    objective: dict[str, Any] | None = None  # e.g. {"mode": "race", "radius": 4, "minDist": 30, "maxDist": 70}
    end_on_elimination: bool = True  # False: run to the time limit after a wipe (race), end when controlled agents are dead
    base_seed: int = 0
    env_id_offset: int = 0
    connect_timeout: float = 10.0
    step_timeout: float = 120.0
    validate: bool = True

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["controlled"] = list(self.controlled)
        d["goal"] = list(self.goal) if self.goal is not None else None
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EnvConfig":
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        if kwargs.get("goal") is not None:
            kwargs["goal"] = (float(kwargs["goal"][0]), float(kwargs["goal"][1]))
        if "controlled" in kwargs:
            kwargs["controlled"] = tuple(kwargs["controlled"])
        return cls(**kwargs)


def default_seed_fn(base_seed: int, n_envs: int) -> SeedFn:
    """Unique string seeds per (env, episode): ``cpc-duo2v2-seed-<base + env + n_envs*episode>``."""

    def fn(env_index: int, episode_index: int) -> str:
        return f"cpc-duo2v2-seed-{base_seed + env_index + n_envs * episode_index}"

    return fn


class SurvevVecEnv:
    def __init__(
        self,
        bridge_url: str = DEFAULT_BRIDGE_URL,
        n_envs: int = 1,
        controlled: Sequence[str] = ("team-a-0", "team-a-1"),
        ticks: int = 10,
        scenario: str = DEFAULT_SCENARIO,
        seed_fn: SeedFn | None = None,
        featurizer: Featurizer | None = None,
        action_space: ActionSpace | None = None,
        reward_config: RewardConfig | None = None,
        scripted: str = "chaser",
        time_limit: float = 60.0,
        map_size: int = 128,
        base_seed: int = 0,
        env_id_offset: int = 0,
        connect_timeout: float = 10.0,
        step_timeout: float = 120.0,
        validate: bool = True,
        loadout: str = "fists",
        layout: str = "fixed",
        goal: tuple[float, float] | None = None,
        objective: Mapping[str, Any] | None = None,
        end_on_elimination: bool = True,
    ) -> None:
        if n_envs <= 0:
            raise ValueError("n_envs must be >= 1")
        if not controlled:
            raise ValueError("at least one controlled agent is required")
        self.n_envs = int(n_envs)
        self.controlled = tuple(controlled)
        self.n_controlled = len(self.controlled)
        self.ticks = int(ticks)
        self.scenario = scenario
        self.scripted = scripted
        self.time_limit = float(time_limit)
        self.map_size = int(map_size)
        self.loadout = str(loadout)
        self.layout = str(layout)
        self.goal = (float(goal[0]), float(goal[1])) if goal is not None else None
        self.objective = dict(objective) if objective else None
        self.end_on_elimination = bool(end_on_elimination)
        self.env_id_offset = int(env_id_offset)
        self.seed_fn = seed_fn or default_seed_fn(base_seed, self.n_envs)
        self.featurizer = featurizer or Featurizer(FeaturizerConfig(time_limit=self.time_limit))
        self.action_space = action_space or ActionSpace()
        self.reward_config = reward_config or RewardConfig()
        self.client = BridgeClient(
            bridge_url, connect_timeout=connect_timeout, timeout=step_timeout, validate=validate
        )
        self.n_rows = self.n_envs * self.n_controlled
        self.agent_onehot_dim = self.n_controlled if self.n_controlled > 1 else 0
        self.obs_dim = self.featurizer.size + self.agent_onehot_dim
        self.rows: list[tuple[int, str]] = [
            (i, aid) for i in range(self.n_envs) for aid in self.controlled
        ]
        self.last_messages: list[ObsMessage | None] = [None] * self.n_envs
        # message returned by the most recent step (before any auto-reset), for loggers
        self.step_messages: list[ObsMessage | None] = [None] * self.n_envs
        self.episode_index = [0] * self.n_envs
        self._memories: list[dict[str, AgentMemory]] = [{} for _ in range(self.n_envs)]
        self._ep_return = np.zeros(self.n_rows, dtype=np.float64)
        self._ep_len = np.zeros(self.n_rows, dtype=np.int64)
        self._ep_goal_hold = np.zeros(self.n_rows, dtype=np.int64)  # steps standing within goal_radius
        self._ep_goal_min = np.full(self.n_rows, np.inf, dtype=np.float64)  # closest approach to the goal
        self._ep_gun_t = np.full(self.n_rows, -1.0, dtype=np.float64)  # game time the agent first held a gun
        self._last_seed: list[str | int | None] = [None] * self.n_envs

    # -- helpers ---------------------------------------------------------------------------
    @property
    def vector_keys(self) -> list[str]:
        keys = self.featurizer.vector_keys
        if self.agent_onehot_dim:
            keys = keys + [f"agent_is_{aid}" for aid in self.controlled]
        return keys

    def env_id(self, env_index: int) -> int:
        return self.env_id_offset + env_index

    def row_index(self, env_index: int, agent_id: str) -> int:
        return env_index * self.n_controlled + self.controlled.index(agent_id)

    def _reset_env(self, i: int) -> ObsMessage:
        seed = self.seed_fn(i, self.episode_index[i])
        extra: dict[str, Any] = {}  # spec options only by default
        if self.loadout != "fists":
            extra["loadout"] = self.loadout
        if self.layout != "fixed":
            extra["layout"] = self.layout
        if self.objective:
            extra["objective"] = dict(self.objective)
        if not self.end_on_elimination:
            extra["endOnElimination"] = False
        msg = self.client.reset(
            self.env_id(i),
            scenario=self.scenario,
            seed=seed,
            options=extra or None,
            controlled=self.controlled,
            scripted=self.scripted,
            time_limit=self.time_limit,
            map_size=self.map_size,
        )
        self._last_seed[i] = seed
        self.episode_index[i] += 1
        self.last_messages[i] = msg
        self._memories[i] = {
            aid: self.featurizer.new_memory(
                [a for a in msg.agent_ids if msg.team_of(a) != msg.team_of(aid)]
            )
            for aid in self.controlled
        }
        for j in range(self.n_controlled):
            row = i * self.n_controlled + j
            self._ep_return[row] = 0.0
            self._ep_len[row] = 0
            self._ep_goal_hold[row] = 0
            self._ep_goal_min[row] = np.inf
            self._ep_gun_t[row] = -1.0
        return msg

    def _featurize_env(self, i: int, msg: ObsMessage, out: np.ndarray) -> None:
        for j, aid in enumerate(self.controlled):
            vec = self.featurizer.featurize(msg.obs[aid], msg.t, self._memories[i][aid], goal=self.goal)
            row = i * self.n_controlled + j
            out[row, : self.featurizer.size] = vec
            if self.agent_onehot_dim:
                out[row, self.featurizer.size + j] = 1.0

    # -- API ---------------------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        obs = np.zeros((self.n_rows, self.obs_dim), dtype=np.float32)
        for i in range(self.n_envs):
            msg = self._reset_env(i)
            self._featurize_env(i, msg, obs)
        return obs

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        actions = np.asarray(actions)
        if actions.ndim == 1:
            actions = actions.reshape(self.n_rows, -1)
        if actions.shape != (self.n_rows, self.action_space.n_dims):
            raise ValueError(
                f"actions must have shape {(self.n_rows, self.action_space.n_dims)}, got {actions.shape}"
            )
        batch: dict[int, tuple[dict[str, Any], int]] = {}
        for i in range(self.n_envs):
            prev = self.last_messages[i]
            if prev is None:
                raise RuntimeError("call reset() before step()")
            per_agent = {}
            for j, aid in enumerate(self.controlled):
                row = i * self.n_controlled + j
                per_agent[aid] = self.action_space.to_cpc_action(actions[row], prev.obs[aid])
            batch[self.env_id(i)] = (per_agent, self.ticks)
        results = self.client.step_batch(batch)

        obs = np.zeros((self.n_rows, self.obs_dim), dtype=np.float32)
        rewards = np.zeros(self.n_rows, dtype=np.float32)
        dones = np.zeros(self.n_rows, dtype=bool)
        infos: list[dict[str, Any]] = [{} for _ in range(self.n_rows)]
        for i in range(self.n_envs):
            prev = self.last_messages[i]
            new = results[self.env_id(i)]
            self.step_messages[i] = new
            breakdown = compute_reward_breakdown(
                prev.obs, new.obs, new.events, new.info, self.reward_config, self.controlled, t=new.t, goal=self.goal
            )
            for j, aid in enumerate(self.controlled):
                row = i * self.n_controlled + j
                r = float(breakdown.totals[aid])
                rewards[row] = r
                self._ep_return[row] += r
                self._ep_len[row] += 1
                me = new.obs[aid]["self"]
                infos[row] = {
                    "env_index": i,
                    "env_id": self.env_id(i),
                    "agent_id": aid,
                    "t": new.t,
                    "active": not bool(me.get("dead")),
                    "reward_components": breakdown.components[aid],
                    "n_events": len(new.events),
                }
                if self._ep_gun_t[row] < 0 and has_gun(me):
                    self._ep_gun_t[row] = float(new.t)
                if self.goal is not None:
                    pos = me.get("pos") or {}
                    d = float(np.hypot(float(pos.get("x", 0.0)) - self.goal[0], float(pos.get("y", 0.0)) - self.goal[1]))
                    infos[row]["goal_dist"] = d
                    if not me.get("dead") and not me.get("downed"):
                        self._ep_goal_min[row] = min(self._ep_goal_min[row], d)
                        if d <= self.reward_config.goal_radius:
                            self._ep_goal_hold[row] += 1
            if new.done:
                metrics = new.info.metrics or {}
                for j, aid in enumerate(self.controlled):
                    row = i * self.n_controlled + j
                    dones[row] = True
                    m = metrics.get(aid)
                    ep_metrics = m.as_float_dict() if m is not None else {}
                    ep_metrics["gun_pickup_time"] = float(self._ep_gun_t[row])  # -1 = never held a gun
                    ep_metrics["armed"] = 1.0 if self._ep_gun_t[row] >= 0 else 0.0
                    if self.goal is not None:  # waypoint stats ride along with the bridge metrics
                        ep_metrics["goal_hold_frac"] = float(self._ep_goal_hold[row]) / max(1, int(self._ep_len[row]))
                        ep_metrics["goal_min_dist"] = float(self._ep_goal_min[row]) if np.isfinite(self._ep_goal_min[row]) else -1.0
                    infos[row].update(
                        {
                            "episode_metrics": ep_metrics,
                            "episode_return": float(self._ep_return[row]),
                            "episode_length": int(self._ep_len[row]),
                            "episode_time": new.t,
                            "winner_team": new.info.winner_team,
                            "reason": new.info.reason,
                            "team_win": bool(m.team_win) if m is not None else False,
                            "seed": self._last_seed[i],
                        }
                    )
                # featurize the terminal message once so memories see the final frame,
                # then auto-reset and hand out the first obs of the next episode
                self._featurize_env(i, new, obs)
                terminal = obs[i * self.n_controlled : (i + 1) * self.n_controlled].copy()
                for j in range(self.n_controlled):
                    infos[i * self.n_controlled + j]["terminal_observation"] = terminal[j]
                msg = self._reset_env(i)
                self._featurize_env(i, msg, obs)
            else:
                self.last_messages[i] = new
                self._featurize_env(i, new, obs)
        return obs, rewards, dones, infos

    def close(self) -> None:
        try:
            self.client.close_all()
        finally:
            self.client.disconnect()

    def __enter__(self) -> "SurvevVecEnv":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class SurvevSingleEnv:
    """One env, one controlled agent: ``reset() -> obs[obs_dim]``, ``step(a) -> (obs, r, done, info)``."""

    def __init__(self, bridge_url: str = DEFAULT_BRIDGE_URL, controlled: str = "team-a-0", **kwargs: Any) -> None:
        self.vec = SurvevVecEnv(bridge_url, n_envs=1, controlled=(controlled,), **kwargs)
        self.obs_dim = self.vec.obs_dim
        self.action_space = self.vec.action_space
        self.featurizer = self.vec.featurizer

    @property
    def last_message(self) -> ObsMessage | None:
        return self.vec.last_messages[0]

    def reset(self) -> np.ndarray:
        return self.vec.reset()[0]

    def step(self, action: Sequence[int]) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        obs, rewards, dones, infos = self.vec.step(np.asarray(action).reshape(1, -1))
        return obs[0], float(rewards[0]), bool(dones[0]), infos[0]

    def close(self) -> None:
        self.vec.close()

    def __enter__(self) -> "SurvevSingleEnv":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def make_vec_env(
    env_cfg: EnvConfig,
    featurizer_cfg: FeaturizerConfig | None = None,
    action_space: ActionSpace | None = None,
    reward_cfg: RewardConfig | None = None,
    seed_fn: SeedFn | None = None,
) -> SurvevVecEnv:
    fcfg = featurizer_cfg or FeaturizerConfig(time_limit=env_cfg.time_limit)
    return SurvevVecEnv(
        bridge_url=env_cfg.bridge_url,
        n_envs=env_cfg.n_envs,
        controlled=env_cfg.controlled,
        ticks=env_cfg.ticks,
        scenario=env_cfg.scenario,
        seed_fn=seed_fn,
        featurizer=Featurizer(fcfg),
        action_space=action_space,
        reward_config=reward_cfg,
        scripted=env_cfg.scripted,
        time_limit=env_cfg.time_limit,
        layout=env_cfg.layout,
        goal=env_cfg.goal,
        objective=env_cfg.objective,
        end_on_elimination=env_cfg.end_on_elimination,
        map_size=env_cfg.map_size,
        base_seed=env_cfg.base_seed,
        env_id_offset=env_cfg.env_id_offset,
        connect_timeout=env_cfg.connect_timeout,
        step_timeout=env_cfg.step_timeout,
        validate=env_cfg.validate,
        loadout=env_cfg.loadout,
    )
