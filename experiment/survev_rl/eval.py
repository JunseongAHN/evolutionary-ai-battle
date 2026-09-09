"""Evaluate a PPO checkpoint (or a random baseline) against the scripted chaser.

Prints win rate / mean survival_time / hp_mean / hp_end / damage dealt & taken over N
episodes and writes one JSONL record per episode (obs summaries, raw + CpcAction actions,
rewards, events, per-agent metrics). Rewards are logged for reference only: evaluation is
the metric vector (harness convention).

Examples::

    python -m experiment.survev_rl.eval --checkpoint runs/ppo_v0/checkpoint_final.pt \
        --bridge ws://127.0.0.1:8765 --episodes 50 --out runs/ppo_v0/eval.jsonl
    python -m experiment.survev_rl.eval --mock --policy random --episodes 5 --out runs/eval_random.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .actions import ActionSpace
from .env import EnvConfig, SurvevVecEnv, make_vec_env
from .featurizer import FeaturizerConfig
from .protocol import DEFAULT_BRIDGE_URL, AgentObservation, ObsMessage
from .rewards import RewardConfig

PolicyFn = Callable[[np.ndarray], np.ndarray]
SUMMARY_KEYS: tuple[str, ...] = (
    "survival_time",
    "hp_mean",
    "hp_end",
    "damage_dealt",
    "damage_taken",
    "kills",
    "shots",
    "hits_given",
    "downed_time",
    "partner_survival_time",
)


def obs_summary(obs: AgentObservation) -> dict[str, Any]:
    me = obs["self"]
    return {
        "pos": [round(float(me["pos"]["x"]), 2), round(float(me["pos"]["y"]), 2)],
        "hp": round(float(me["hp"]), 1),
        "downed": bool(me.get("downed")),
        "dead": bool(me.get("dead")),
        "weapon": me.get("weapon"),
        "clip": int(me.get("clip", 0)),
        "reserve": int(me.get("reserve", 0)),
        "n_enemies_visible": len(obs.get("players") or []),
        "n_loot_visible": len(obs.get("loot") or []),
        "nearest_enemy_dist": min((float(p["dist"]) for p in obs.get("players") or []), default=None),
    }


def random_policy(action_space: ActionSpace, seed: int) -> PolicyFn:
    rng = np.random.default_rng(seed)

    def fn(obs: np.ndarray) -> np.ndarray:
        return np.stack([action_space.sample(rng) for _ in range(obs.shape[0])])

    return fn


def checkpoint_policy(checkpoint: str | Path, device: str, deterministic: bool) -> tuple[PolicyFn, dict[str, Any]]:
    import torch

    from .ppo import load_checkpoint

    agent, ckpt = load_checkpoint(checkpoint, device)

    def fn(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=device)
            action, _, _, _ = agent.get_action_and_value(x, deterministic=deterministic)
        return action.cpu().numpy()

    return fn, ckpt


def run_episodes(
    env: SurvevVecEnv,
    policy: PolicyFn,
    n_episodes: int,
    full_obs: bool = False,
    on_episode: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Roll out until ``n_episodes`` episodes finished across the vec env; returns records."""
    obs = env.reset()
    partial: list[dict[str, Any]] = [
        {"steps": [], "returns": {aid: 0.0 for aid in env.controlled}} for _ in range(env.n_envs)
    ]
    finished: list[dict[str, Any]] = []
    episode_counter = 0
    while len(finished) < n_episodes:
        prev_msgs: list[ObsMessage] = [m for m in env.last_messages]  # type: ignore[misc]
        actions = policy(obs)
        cpc_actions = {
            i: {aid: env.action_space.to_cpc_action(actions[env.row_index(i, aid)], prev_msgs[i].obs[aid]).to_json()
                for aid in env.controlled}
            for i in range(env.n_envs)
        }
        obs, rewards, dones, infos = env.step(actions)
        for i in range(env.n_envs):
            new = env.step_messages[i]
            assert new is not None
            step_rec: dict[str, Any] = {
                "t": prev_msgs[i].t,
                "t_next": new.t,
                "obs": {
                    aid: (prev_msgs[i].obs[aid] if full_obs else obs_summary(prev_msgs[i].obs[aid]))
                    for aid in env.controlled
                },
                "actions": {
                    aid: {"raw": [int(v) for v in actions[env.row_index(i, aid)]], "cpc": cpc_actions[i][aid]}
                    for aid in env.controlled
                },
                "rewards": {aid: float(rewards[env.row_index(i, aid)]) for aid in env.controlled},
                "events": new.events,
            }
            partial[i]["steps"].append(step_rec)
            for aid in env.controlled:
                partial[i]["returns"][aid] += float(rewards[env.row_index(i, aid)])
            if dones[env.row_index(i, env.controlled[0])]:
                info = infos[env.row_index(i, env.controlled[0])]
                metrics = {aid: m.to_json() for aid, m in (new.info.metrics or {}).items()}
                record = {
                    "episode": episode_counter,
                    "env_id": env.env_id(i),
                    "seed": info.get("seed"),
                    "controlled": list(env.controlled),
                    "agent_ids": list(new.agent_ids),
                    "teams": dict(new.teams),
                    "ticks": env.ticks,
                    "length": len(partial[i]["steps"]),
                    "duration_s": new.t,
                    "winner_team": new.info.winner_team,
                    "reason": new.info.reason,
                    "team_win": bool(info.get("team_win", False)),
                    "episode_return": dict(partial[i]["returns"]),
                    "metrics": metrics,
                    "steps": partial[i]["steps"],
                }
                episode_counter += 1
                finished.append(record)
                if on_episode is not None:
                    on_episode(record)
                partial[i] = {"steps": [], "returns": {aid: 0.0 for aid in env.controlled}}
                if len(finished) >= n_episodes:
                    break
    return finished


def summarize(records: Sequence[Mapping[str, Any]], controlled: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"episodes": len(records)}
    if not records:
        return out
    out["win_rate"] = statistics.fmean(1.0 if r["team_win"] else 0.0 for r in records)
    out["draw_rate"] = statistics.fmean(1.0 if r["winner_team"] is None else 0.0 for r in records)
    out["mean_duration_s"] = statistics.fmean(float(r["duration_s"]) for r in records)
    out["mean_return"] = statistics.fmean(
        statistics.fmean(float(v) for v in r["episode_return"].values()) for r in records
    )
    per_agent: dict[str, dict[str, float]] = {}
    for aid in controlled:
        per_agent[aid] = {
            k: statistics.fmean(float(r["metrics"][aid][k]) for r in records if aid in r["metrics"])
            for k in SUMMARY_KEYS
        }
    out["per_agent"] = per_agent
    out["mean"] = {k: statistics.fmean(per_agent[aid][k] for aid in controlled) for k in SUMMARY_KEYS}
    return out


def print_summary(summary: Mapping[str, Any]) -> None:
    print(f"episodes: {summary['episodes']}")
    if not summary.get("episodes"):
        return
    print(
        f"win_rate: {summary['win_rate']:.3f}  draw_rate: {summary['draw_rate']:.3f}  "
        f"mean_duration: {summary['mean_duration_s']:.1f}s  mean_return: {summary['mean_return']:.3f}"
    )
    m = summary["mean"]
    print(
        f"survival_time: {m['survival_time']:.2f}s  hp_mean: {m['hp_mean']:.1f}  hp_end: {m['hp_end']:.1f}  "
        f"damage_dealt: {m['damage_dealt']:.1f}  damage_taken: {m['damage_taken']:.1f}  "
        f"kills: {m['kills']:.2f}  shots: {m['shots']:.1f}  hits: {m['hits_given']:.1f}"
    )
    for aid, vals in summary["per_agent"].items():
        print(
            f"  {aid}: survival {vals['survival_time']:.2f}s hp_mean {vals['hp_mean']:.1f} "
            f"dealt {vals['damage_dealt']:.1f} taken {vals['damage_taken']:.1f} "
            f"partner_survival {vals['partner_survival_time']:.2f}s"
        )


def to_harness_episode(record: Mapping[str, Any]) -> dict[str, Any]:
    """Partial mapping of an eval record to the harness ``EpisodeTrajectory`` shape.

    TODO(S6): the full ``EpisodeTrajectory`` (per-step ``BattleSnapshot`` / ``TacticalObservation``
    / ``BattleAction`` records validated by ``experiment.core.schema_validation.validate_episode``)
    is produced by the bridge's own JSONL export (PR-S6), which sees every agent's full state.
    Until then this stub only fills the episode-level fields and ``final_metrics`` (combat /
    survival / cooperation groups from the bridge metrics); ``steps`` stays empty.
    """
    final_metrics: dict[str, Any] = {}
    for aid, m in (record.get("metrics") or {}).items():
        final_metrics[aid] = {
            "agent_id": aid,
            "team_id": record.get("teams", {}).get(aid, ""),
            "combat": {
                "damageDealt": float(m.get("damage_dealt", 0.0)),
                "damageTaken": float(m.get("damage_taken", 0.0)),
                "kills": int(m.get("kills", 0)),
                "shots": int(m.get("shots", 0)),
                "hits": int(m.get("hits_given", 0)),
            },
            "survival": {
                "survivalTime": float(m.get("survival_time", 0.0)),
                "aliveAtEnd": bool(m.get("alive_at_end", False)),
                "downedTime": float(m.get("downed_time", 0.0)),
                "hpMean": float(m.get("hp_mean", 0.0)),
                "hpEnd": float(m.get("hp_end", 0.0)),
                "teamWin": bool(m.get("team_win", False)),
            },
            "cooperation": {
                "applicable": True,
                "partnerSurvivalTime": float(m.get("partner_survival_time", 0.0)),
                "partnerHpEnd": float(m.get("partner_hp_end", 0.0)),
            },
        }
    return {
        "schema_version": "cpc-common-v0",
        "episode_id": f"survev-{record.get('seed')}-{record.get('episode')}",
        "config": {
            "schema_version": "cpc-common-v0",
            "mode": "duo",
            "team_count": 2,
            "players_per_team": 2,
            "max_steps": int(record.get("length", 0)),
        },
        "steps": [],  # TODO(S6): filled from the bridge JSONL export
        "final_metrics": final_metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a PPO checkpoint vs the scripted chaser.")
    p.add_argument("--checkpoint", default=None, help="checkpoint .pt (omit with --policy random)")
    p.add_argument("--policy", default="checkpoint", choices=["checkpoint", "random"])
    p.add_argument("--deterministic", action="store_true", help="argmax actions instead of sampling")
    p.add_argument("--bridge", default=DEFAULT_BRIDGE_URL)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--controlled", default=None, help="override controlled agents (comma-separated)")
    p.add_argument("--scripted", default=None, choices=["chaser", "idle", "racer"], help="override the checkpoint's opponent")
    p.add_argument("--ticks", type=int, default=None)
    p.add_argument("--time-limit", type=float, default=None)
    p.add_argument("--loadout", default=None, choices=["fists", "armed"], help="override the checkpoint's loadout")
    p.add_argument("--layout", default=None, choices=["fixed", "random"], help="override the checkpoint's spawn layout")
    p.add_argument("--seed", type=int, default=1000, help="base episode seed (kept apart from training)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--full-obs", action="store_true", help="store full AgentObservations per step")
    p.add_argument("--out", default=None, help="JSONL output path (one record per episode)")
    return p


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    mock = None
    bridge_url = args.bridge
    if args.mock:
        from .mock_bridge import MockBridgeThread

        mock = MockBridgeThread().start()
        bridge_url = mock.url

    meta: dict[str, Any] = {}
    policy: PolicyFn
    if args.policy == "checkpoint":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required unless --policy random")
        policy, ckpt = checkpoint_policy(args.checkpoint, args.device, args.deterministic)
        meta = dict(ckpt.get("meta") or {})
    env_meta = dict(meta.get("env") or {})
    controlled = tuple(args.controlled.split(",")) if args.controlled else tuple(env_meta.get("controlled") or ("team-a-0", "team-a-1"))
    env_cfg = EnvConfig(
        bridge_url=bridge_url,
        n_envs=args.n_envs,
        controlled=controlled,
        ticks=args.ticks or int(env_meta.get("ticks", 10)),
        scripted=args.scripted or str(env_meta.get("scripted", "chaser")),
        time_limit=args.time_limit or float(env_meta.get("time_limit", 60.0)),
        map_size=int(env_meta.get("map_size", 128)),
        loadout=args.loadout or str(env_meta.get("loadout", "fists")),
        layout=args.layout or str(env_meta.get("layout", "fixed")),
        goal=tuple(env_meta["goal"]) if env_meta.get("goal") else None,
        objective=env_meta.get("objective"),
        end_on_elimination=bool(env_meta.get("end_on_elimination", True)),
        base_seed=args.seed,
    )
    feat_cfg = FeaturizerConfig.from_dict(meta.get("featurizer") or {"time_limit": env_cfg.time_limit})
    as_meta = meta.get("action_space") or {}
    action_space = ActionSpace(
        mode=as_meta.get("mode", "primitive"), assist=as_meta.get("assist", True), auto_pickup=as_meta.get("auto_pickup", False)
    )
    reward_cfg = RewardConfig.from_dict(meta.get("reward") or {})
    env = make_vec_env(env_cfg, feat_cfg, action_space, reward_cfg)
    if args.policy == "random":
        policy = random_policy(action_space, args.seed)
    if meta.get("obs_dim") and int(meta["obs_dim"]) != env.obs_dim:
        raise SystemExit(f"checkpoint obs_dim {meta['obs_dim']} != env obs_dim {env.obs_dim}; check --controlled")

    out_path = Path(args.out) if args.out else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")

    def on_episode(rec: dict[str, Any]) -> None:
        if out_path is not None:
            with out_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=float) + "\n")
        m = rec["metrics"]
        print(
            f"episode {rec['episode']}: {rec['reason']} winner={rec['winner_team']} t={rec['duration_s']:.1f}s "
            + " ".join(f"{aid}[surv {m[aid]['survival_time']:.1f} dealt {m[aid]['damage_dealt']:.0f}]" for aid in controlled),
            flush=True,
        )

    try:
        records = run_episodes(env, policy, args.episodes, full_obs=args.full_obs, on_episode=on_episode)
    finally:
        env.close()
        if mock is not None:
            mock.stop()
    summary = summarize(records, controlled)
    print_summary(summary)
    if out_path is not None:
        out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {out_path} and {out_path.with_suffix('.summary.json')}")
    return summary


if __name__ == "__main__":
    main()
