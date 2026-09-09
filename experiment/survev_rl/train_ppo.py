"""Train PPO against the survev bridge (or the in-process mock).

Examples::

    # real bridge (start it first: `pnpm cpc:bridge` in the survev repo)
    python -m experiment.survev_rl.train_ppo --bridge ws://127.0.0.1:8765 --n-envs 16 \
        --ticks 10 --total-steps 2000000 --device cuda --out runs/ppo_v0

    # smoke test on the mock bridge
    python -m experiment.survev_rl.train_ppo --mock --n-envs 2 --total-steps 2000 --device cpu \
        --out runs/smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .actions import ActionSpace
from .env import EnvConfig, make_vec_env
from .featurizer import FeaturizerConfig
from .protocol import DEFAULT_BRIDGE_URL
from .rewards import RewardConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PPO training for survev duo2v2_field.")
    # bridge / env
    p.add_argument("--bridge", default=DEFAULT_BRIDGE_URL, help="bridge WebSocket URL")
    p.add_argument("--mock", action="store_true", help="start the mock bridge in a thread and use it")
    p.add_argument("--mock-port", type=int, default=0, help="port for --mock (0 = free port)")
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--ticks", type=int, default=10, help="game ticks per decision (10 = 0.1 s)")
    p.add_argument("--controlled", default="team-a-0,team-a-1", help="comma-separated agent ids")
    p.add_argument("--scripted", default="chaser", choices=["chaser", "idle", "racer"],
                   help="opponent: chaser hunts, racer goes for the race point and fights only inside 25 u")
    p.add_argument("--time-limit", type=float, default=60.0)
    p.add_argument("--map-size", type=int, default=128)
    p.add_argument("--loadout", default="fists", choices=["fists", "armed"],
                   help="curriculum: 'armed' spawns everyone with an ak47 (mock; bridge extension request)")
    p.add_argument("--layout", default="fixed", choices=["fixed", "random"],
                   help="spawn geometry: 'random' rotates the spawn axis and draws the distance per seed (bridge)")
    p.add_argument("--goal", default=None,
                   help="waypoint 'x,y' or 'center' (132,132); adds the goal block to the observation and enables "
                        "the waypoint reward terms in --reward-json (goal_progress / goal_hold / enemy_at_goal)")
    p.add_argument("--objective", default="none", choices=["none", "race"],
                   help="'race': shared moving capture point (bridge option objective.mode); adds the goal block fed "
                        "from obs.objective, runs episodes to the time limit after a wipe, rewards via 'capture' terms")
    p.add_argument("--objective-radius", type=float, default=4.0)
    p.add_argument("--objective-dist", default="30,70", help="min,max distance between consecutive points")
    p.add_argument("--end-on-elimination", action="store_true",
                   help="with --objective race: still end the episode when one team is left (default: keep racing)")
    p.add_argument("--opp-aim-noise", type=float, default=0.0,
                   help="opponent strength: std-dev in degrees of the scripted bots' aim error (bridge scriptedOptions)")
    p.add_argument("--opp-reaction", type=float, default=0.0,
                   help="opponent strength: seconds an enemy must be inside 30 u before the bots open fire")
    p.add_argument("--opp-engage-dist", type=float, default=None,
                   help="opponent strength: racer break-off distance in u (default 25)")
    p.add_argument("--seed", type=int, default=0, help="torch/numpy seed and base episode seed")
    p.add_argument("--no-validate", action="store_true", help="skip the observation key allowlist")
    # featurizer / actions / rewards
    p.add_argument("--rotate-obs", action="store_true", help="rotate relative vectors into the facing frame")
    p.add_argument("--no-memory", action="store_true", help="disable last-seen enemy memory features")
    p.add_argument("--no-assist", action="store_true", help="disable the equip/reload assist layer")
    p.add_argument("--auto-pickup", action="store_true", help="assist extra: Interact whenever loot is in reach")
    p.add_argument("--aim-assist", action="store_true",
                   help="while fire is held, aim at the nearest visible enemy (the exact aim the scripted bots have)")
    p.add_argument("--team-mix", type=float, default=0.0, help="blend factor with team-mean reward [0,1]")
    p.add_argument("--reward-json", default=None, help="JSON file/inline overriding RewardConfig fields")
    # ppo
    p.add_argument("--total-steps", type=int, default=2_000_000, help="agent transitions (rows x steps)")
    p.add_argument("--rollout-steps", type=int, default=128)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--no-anneal-lr", action="store_true")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=None)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--device", default="cpu", help="cuda | cpu | auto")
    p.add_argument("--checkpoint-every", type=int, default=10, help="updates between checkpoints")
    p.add_argument("--tensorboard", action="store_true")
    p.add_argument("--resume", default=None, help="checkpoint to resume from")
    p.add_argument("--out", default="runs/ppo_v0")
    p.add_argument("--quiet", action="store_true")
    return p


def parse_goal(spec: str | None) -> tuple[float, float] | None:
    """'center' -> the field center, 'x,y' -> that point, None -> no waypoint."""
    if spec is None or spec == "":
        return None
    if spec == "center":
        return (132.0, 132.0)
    x, y = (float(v) for v in spec.split(","))
    return (x, y)


def parse_objective(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.objective == "none":
        return None
    lo, hi = (float(v) for v in args.objective_dist.split(","))
    return {"mode": args.objective, "radius": float(args.objective_radius), "minDist": lo, "maxDist": hi}


def parse_scripted_options(args: argparse.Namespace) -> dict[str, Any] | None:
    opts: dict[str, Any] = {}
    if args.opp_aim_noise > 0:
        opts["aimNoiseDeg"] = float(args.opp_aim_noise)
    if args.opp_reaction > 0:
        opts["reactionDelay"] = float(args.opp_reaction)
    if args.opp_engage_dist is not None:
        opts["engageDist"] = float(args.opp_engage_dist)
    return opts or None


def parse_reward_override(spec: str | None, team_mix: float) -> RewardConfig:
    data: dict[str, Any] = {}
    if spec:
        path = Path(spec)
        text = path.read_text(encoding="utf-8") if path.exists() else spec
        data = json.loads(text)
    data.setdefault("team_mix", team_mix)
    return RewardConfig.from_dict({**RewardConfig().to_dict(), **data})


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    try:
        from .ppo import PPOConfig, train
    except ImportError as exc:  # pragma: no cover
        print(f"torch is required for training ({exc}): pip install torch", file=sys.stderr)
        raise SystemExit(1) from exc

    mock = None
    bridge_url = args.bridge
    if args.mock:
        from .mock_bridge import MockBridgeThread

        mock = MockBridgeThread(port=args.mock_port).start()
        bridge_url = mock.url
        print(f"[train] mock bridge at {bridge_url}")

    controlled = tuple(a.strip() for a in args.controlled.split(",") if a.strip())
    env_cfg = EnvConfig(
        bridge_url=bridge_url,
        n_envs=args.n_envs,
        controlled=controlled,
        ticks=args.ticks,
        scripted=args.scripted,
        time_limit=args.time_limit,
        map_size=args.map_size,
        loadout=args.loadout,
        layout=args.layout,
        goal=parse_goal(args.goal),
        objective=parse_objective(args),
        end_on_elimination=args.objective == "none" or args.end_on_elimination,
        scripted_options=parse_scripted_options(args),
        base_seed=args.seed,
        validate=not args.no_validate,
    )
    feat_cfg = FeaturizerConfig(
        rotate_to_facing=args.rotate_obs, memory=not args.no_memory, time_limit=args.time_limit,
        goal=env_cfg.goal is not None or env_cfg.objective is not None,
    )
    action_space = ActionSpace(
        mode="primitive", assist=not args.no_assist, auto_pickup=args.auto_pickup, aim_assist=args.aim_assist
    )
    reward_cfg = parse_reward_override(args.reward_json, args.team_mix)
    ppo_cfg = PPOConfig(
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        num_minibatches=args.minibatches,
        update_epochs=args.epochs,
        learning_rate=args.lr,
        anneal_lr=not args.no_anneal_lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        hidden_dim=args.hidden,
        seed=args.seed,
        device=args.device,
        checkpoint_every=args.checkpoint_every,
        tensorboard=args.tensorboard,
    )
    env = make_vec_env(env_cfg, feat_cfg, action_space, reward_cfg)
    meta = {
        "env": env_cfg.to_dict(),
        "featurizer": feat_cfg.to_dict(),
        "action_space": {"mode": action_space.mode, "assist": action_space.assist,
                         "auto_pickup": action_space.auto_pickup, "aim_assist": action_space.aim_assist},
        "reward": reward_cfg.to_dict(),
        "vector_keys": env.vector_keys,
        "obs_dim": env.obs_dim,
        "nvec": list(action_space.nvec),
    }
    print(f"[train] obs_dim={env.obs_dim} nvec={list(action_space.nvec)} rows={env.n_rows} out={args.out}")
    try:
        result = train(ppo_cfg, env, args.out, meta=meta, resume=args.resume, progress=not args.quiet)
    finally:
        env.close()
        if mock is not None:
            mock.stop()
    print(f"[train] done: {json.dumps(result, default=float)}")
    return result


if __name__ == "__main__":
    main()
