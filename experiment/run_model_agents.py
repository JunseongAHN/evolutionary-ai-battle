from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from experiment.policy_agent import PPOPolicyAgent
    from experiment.training.cpc_env import CPCEnv
except ModuleNotFoundError:
    EXPERIMENT_ROOT = Path(__file__).resolve().parent
    REPO_ROOT = EXPERIMENT_ROOT.parent
    for path in (EXPERIMENT_ROOT, REPO_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from experiment.policy_agent import PPOPolicyAgent
    from experiment.training.cpc_env import CPCEnv


def run_two_agent_eval(
    checkpoint_a: str,
    checkpoint_b: str | None = None,
    episodes: int = 1,
    device: str = "cpu",
    deterministic: bool = True,
    export_path: str | None = None,
) -> dict[str, Any]:
    del episodes, deterministic, export_path
    agent_a = PPOPolicyAgent.from_checkpoint(checkpoint_a, device=device)
    agent_b = PPOPolicyAgent.from_checkpoint(checkpoint_b or checkpoint_a, device=device)
    del agent_a, agent_b

    if not _env_supports_two_agent_control():
        raise NotImplementedError(
            "Two-agent model eval is blocked because training.cpc_env.CPCEnv currently exposes "
            "one controllable self agent and step(action) accepts one raw action. Use a "
            "multi-agent env whose step accepts an action mapping before assigning two loaded "
            "PPOPolicyAgent instances."
        )

    raise NotImplementedError("Two-agent eval support should be wired here once a multi-agent CPC env is available.")


def _env_supports_two_agent_control() -> bool:
    env = CPCEnv(seed=0, max_steps=1)
    obs = env.reset()
    return isinstance(obs, dict) and "observations" in obs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run loaded PPO model agents in the Python CPC env.")
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-b")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--export")
    args = parser.parse_args()
    result = run_two_agent_eval(
        checkpoint_a=args.checkpoint_a,
        checkpoint_b=args.checkpoint_b,
        episodes=args.episodes,
        device=args.device,
        deterministic=args.deterministic,
        export_path=args.export,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
