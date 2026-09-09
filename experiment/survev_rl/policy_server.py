"""Serve a PPO checkpoint to a live game over WebSocket (the inverse of the bridge).

The bridge lets Python *drive* offline games; this server lets a game process *ask* Python for
actions, so a trained agent can play inside a normal, client-observable game (a real survev
client renders it, a human can watch or play alongside). One JSON request -> one response:

    {"type": "reset", "agent_ids": [...], "teams": {aid: team}, "controlled": [aid, ...]}
        -> {"type": "ready", "controlled": [...], "checkpoint": "...", "deterministic": bool}
    {"type": "act", "t": 1.2, "obs": {aid: <AgentObservation>, ...}}
        -> {"type": "actions", "actions": {aid: <CpcAction wire form>}, "policy": {aid: {...}}}

Observations are the spec ``AgentObservation`` (exactly what the bridge returns), featurized
with the checkpoint's own ``FeaturizerConfig`` and per-agent ``AgentMemory``; actions go back
in the bridge's ``CpcAction`` wire form so the game applies them through ``applyCpcAction``.
The agent-id one-hot follows the checkpoint's ``controlled`` order. Run:

    python -m experiment.survev_rl.policy_server --checkpoint runs/ppo_v0/checkpoint_final.pt \
        --port 8766 [--deterministic]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from websockets.sync.server import serve

from .actions import ActionSpace
from .featurizer import AgentMemory, Featurizer, FeaturizerConfig
from .ppo import load_checkpoint


class CheckpointPolicy:
    """Stateful (per-agent memory) wrapper around a checkpoint for one live episode at a time."""

    def __init__(self, checkpoint: str | Path, device: str = "cpu", deterministic: bool = False) -> None:
        self.checkpoint = str(checkpoint)
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.agent, ckpt = load_checkpoint(checkpoint, self.device)
        meta = dict(ckpt.get("meta") or {})
        env_meta = dict(meta.get("env") or {})
        self.controlled: tuple[str, ...] = tuple(env_meta.get("controlled") or ("team-a-0", "team-a-1"))
        goal = env_meta.get("goal")
        self.goal: tuple[float, float] | None = (float(goal[0]), float(goal[1])) if goal else None
        self.featurizer = Featurizer(
            FeaturizerConfig.from_dict(meta.get("featurizer") or {"time_limit": env_meta.get("time_limit", 60.0)})
        )
        as_meta = meta.get("action_space") or {}
        self.action_space = ActionSpace(
            mode=as_meta.get("mode", "primitive"),
            assist=as_meta.get("assist", True),
            auto_pickup=as_meta.get("auto_pickup", False),
            aim_assist=as_meta.get("aim_assist", False),
        )
        self.onehot_dim = len(self.controlled) if len(self.controlled) > 1 else 0
        self.obs_dim = self.featurizer.size + self.onehot_dim
        if self.obs_dim != self.agent.obs_dim:
            raise ValueError(f"checkpoint obs_dim {self.agent.obs_dim} != featurizer {self.obs_dim}")
        self.global_step = int(ckpt.get("global_step", 0))
        self.memories: dict[str, AgentMemory] = {}
        self.n_act = 0

    def reset(self, agent_ids: list[str], teams: dict[str, str]) -> None:
        self.memories = {
            aid: self.featurizer.new_memory([a for a in agent_ids if teams.get(a) != teams.get(aid)])
            for aid in self.controlled
        }
        self.n_act = 0

    def act(self, t: float, obs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.memories:
            self.reset(list(obs.keys()), {aid: o["self"]["team"] for aid, o in obs.items()})
        ids = [aid for aid in self.controlled if aid in obs]
        if not ids:
            return {}, {}
        x = np.zeros((len(ids), self.obs_dim), dtype=np.float32)
        for row, aid in enumerate(ids):
            x[row, : self.featurizer.size] = self.featurizer.featurize(obs[aid], t, self.memories[aid], goal=self.goal)
            if self.onehot_dim:
                x[row, self.featurizer.size + self.controlled.index(aid)] = 1.0
        with torch.no_grad():
            action, _, _, value = self.agent.get_action_and_value(
                torch.as_tensor(x, device=self.device), deterministic=self.deterministic
            )
        action_np = action.cpu().numpy()
        actions: dict[str, Any] = {}
        debug: dict[str, Any] = {}
        for row, aid in enumerate(ids):
            me = obs[aid]["self"]
            if me.get("dead") or me.get("downed"):
                actions[aid] = {}
                debug[aid] = {"inactive": True}
                continue
            actions[aid] = self.action_space.to_cpc_action(action_np[row], obs[aid]).to_json()
            desc = self.action_space.describe(action_np[row])
            desc["value"] = float(value[row])
            debug[aid] = desc
        self.n_act += 1
        return actions, debug


def make_handler(policy: CheckpointPolicy, log_f: Any = None, deterministic: bool = False) -> Any:
    """One connection = one live game; the handler serves ``reset`` / ``act`` until it disconnects."""

    def handler(ws: Any) -> None:
        print("[policy] client connected", flush=True)
        for raw in ws:
            try:
                msg = json.loads(raw)
                kind = msg.get("type")
                if kind == "reset":
                    policy.reset(list(msg.get("agent_ids") or []), dict(msg.get("teams") or {}))
                    ws.send(json.dumps({
                        "type": "ready", "controlled": list(policy.controlled),
                        "checkpoint": policy.checkpoint, "deterministic": deterministic,
                    }))
                elif kind == "act":
                    t0 = time.perf_counter()
                    actions, debug = policy.act(float(msg.get("t", 0.0)), dict(msg.get("obs") or {}))
                    ws.send(json.dumps({"type": "actions", "actions": actions, "policy": debug}))
                    if log_f is not None:
                        log_f.write(json.dumps({
                            "t": msg.get("t"), "ms": round((time.perf_counter() - t0) * 1000, 2),
                            "actions": actions, "policy": debug,
                            "self": {aid: o["self"] for aid, o in (msg.get("obs") or {}).items()},
                        }) + "\n")
                        log_f.flush()
                else:
                    ws.send(json.dumps({"type": "error", "message": f"unknown type {kind!r}"}))
            except Exception as exc:  # keep serving; report the failure to the game
                print(f"[policy] error: {exc!r}", file=sys.stderr, flush=True)
                ws.send(json.dumps({"type": "error", "message": repr(exc)}))
        print("[policy] client disconnected", flush=True)

    return handler


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Serve a PPO checkpoint to a live game over WebSocket.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--device", default="cpu")
    p.add_argument("--deterministic", action="store_true", help="argmax actions instead of sampling")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log", default=None, help="append one JSON line per act() call here")
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    policy = CheckpointPolicy(args.checkpoint, args.device, args.deterministic)
    log_f = open(args.log, "a", encoding="utf-8") if args.log else None
    print(
        f"[policy] {args.checkpoint} step={policy.global_step} controlled={list(policy.controlled)} "
        f"obs_dim={policy.obs_dim} deterministic={args.deterministic}",
        flush=True,
    )
    with serve(make_handler(policy, log_f, args.deterministic), args.host, args.port, max_size=None) as server:
        print(f"[policy] listening on ws://{args.host}:{args.port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
