from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

EXPERIMENT_ROOT = Path(__file__).resolve().parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

import torch
from tensordict import TensorDict

from training.cpc_actions import decode_action
from training.cpc_env import CPCEnv
from training.eval_ppo import eval_checkpoint
from training.ppo_policy import MultiDiscreteActorCritic
from training.torchrl_env import TorchRLCPCEnv
from training.train_ppo import PPOConfig, collect_rollout, load_config, train_ppo


REQUIRED_METRIC_COLUMNS = (
    "update",
    "step",
    "episodic_return_mean",
    "episode_length_mean",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
)

OPTIONAL_METRIC_COLUMNS = (
    "avg_ally_distance",
    "isolation_rate",
    "damage_dealt",
    "damage_taken",
)


def set_all_seeds(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def compute_state_dict_checksum(state_dict_or_model) -> str:
    state_dict = (
        state_dict_or_model.state_dict()
        if hasattr(state_dict_or_model, "state_dict")
        else state_dict_or_model
    )
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().to("cpu").contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def run_seed_probe(seed: int, *, rollout_steps: int = 8, max_episode_steps: int = 8) -> dict[str, Any]:
    cfg = PPOConfig(
        seed=seed,
        device="cpu",
        rollout_steps=rollout_steps,
        max_episode_steps=max_episode_steps,
        hidden_dim=16,
    )
    set_all_seeds(seed)
    env = TorchRLCPCEnv(seed=seed, max_steps=max_episode_steps, device="cpu")
    policy = MultiDiscreteActorCritic(hidden_dim=cfg.hidden_dim)
    first = policy.sample_action(env.reset()).action

    set_all_seeds(seed)
    env = TorchRLCPCEnv(seed=seed, max_steps=max_episode_steps, device="cpu")
    policy = MultiDiscreteActorCritic(hidden_dim=cfg.hidden_dim)
    rollout = collect_rollout(env, policy, cfg)
    return {
        "first_action": _action_to_ints(first),
        "rollout_actions": {
            key: rollout["actions"][key].detach().to("cpu").tolist()
            for key in ("move", "aim", "fire")
        },
        "metrics": rollout["last_metrics"],
    }


def check_same_seed_reproducibility(seed: int) -> tuple[str, dict[str, Any]]:
    first = run_seed_probe(seed)
    second = run_seed_probe(seed)
    if first["first_action"] != second["first_action"]:
        return "FAIL", {"first": first["first_action"], "second": second["first_action"]}
    if first["rollout_actions"] != second["rollout_actions"]:
        return "FAIL", {"reason": "rollout action sequence differed"}
    if set(first["metrics"]) != set(second["metrics"]):
        return "FAIL", {"reason": "metric keys differed"}
    for key, value in first["metrics"].items():
        if not math.isclose(float(value), float(second["metrics"].get(key, 0.0)), rel_tol=1e-5, abs_tol=1e-6):
            return "FAIL", {"reason": f"metric differed: {key}"}
    return "PASS", {"first_action": first["first_action"], "metrics": first["metrics"]}


def check_forced_move_fire_action(seed: int) -> tuple[str, dict[str, Any]]:
    raw_action = {"move": 5, "aim": 10, "fire": 1}
    env = CPCEnv(seed=seed, max_steps=4)
    obs, reward, done, info = env.step(raw_action)
    decoded = info.get("decoded_action")
    if not decoded:
        return "FAIL", {"reason": "decoded_action missing"}
    if info.get("raw_action") != raw_action:
        return "FAIL", {"reason": "raw_action missing or changed"}
    if int(decoded["fire"]) != 1:
        return "FAIL", {"reason": "fire not enabled"}
    if abs(float(decoded["moveX"])) + abs(float(decoded["moveY"])) <= 0.0:
        return "FAIL", {"reason": "movement is zero"}
    if not isinstance(reward, float) or not isinstance(done, bool) or not isinstance(obs, dict):
        return "FAIL", {"reason": "invalid step output"}

    torch_env = TorchRLCPCEnv(seed=seed, max_steps=4, device="cpu")
    td = torch_env.reset()
    td["move"] = torch.tensor(raw_action["move"], dtype=torch.int64)
    td["aim"] = torch.tensor(raw_action["aim"], dtype=torch.int64)
    td["fire"] = torch.tensor(raw_action["fire"], dtype=torch.int64)
    if hasattr(torch_env.action_spec, "is_in"):
        try:
            if not bool(torch_env.action_spec.is_in(TensorDict({
                "move": td["move"],
                "aim": td["aim"],
                "fire": td["fire"],
            }, batch_size=[]))):
                return "FAIL", {"reason": "action rejected by action_spec"}
        except Exception:
            pass
    torch_env.step(td)
    return "PASS", {
        "raw_action": raw_action,
        "decoded_action": normalize_decoded_action(decoded),
    }


def check_decode_bounds() -> tuple[str, dict[str, Any]]:
    samples = [
        {"move": move, "aim": aim, "fire": fire}
        for move in range(9)
        for aim in range(16)
        for fire in range(2)
    ]
    for raw in samples:
        decoded = decode_action(raw)
        for key in ("moveX", "moveY", "aimX", "aimY"):
            if not -1.000001 <= float(decoded[key]) <= 1.000001:
                return "FAIL", {"raw_action": raw, "decoded_action": decoded, "bad_key": key}
        if int(decoded["fire"]) not in (0, 1):
            return "FAIL", {"raw_action": raw, "reason": "invalid fire"}
        if raw["move"] != 0 and abs(float(decoded["moveX"])) + abs(float(decoded["moveY"])) <= 0.0:
            return "FAIL", {"raw_action": raw, "reason": "non-zero move decoded to zero"}
        if raw["fire"] == 1 and int(decoded["fire"]) != 1:
            return "FAIL", {"raw_action": raw, "reason": "fire lost during decode"}
    return "PASS", {"checked_actions": len(samples)}


def validate_metrics_csv(path: str | Path, *, min_rows: int = 2) -> tuple[str, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return "FAIL", {"reason": "metrics.csv missing"}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = set(reader.fieldnames or [])
    missing = [column for column in REQUIRED_METRIC_COLUMNS if column not in columns]
    if missing:
        return "FAIL", {"reason": "required columns missing", "missing": missing}
    if len(rows) < min_rows:
        return "FAIL", {"reason": "not enough metric rows", "rows": len(rows)}
    return "PASS", {
        "rows": len(rows),
        "optional_columns": [column for column in OPTIONAL_METRIC_COLUMNS if column in columns],
    }


def validate_checkpoint_load(checkpoint: str | Path, *, eval_episodes: int) -> tuple[str, dict[str, Any]]:
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        return "FAIL", {"reason": "checkpoint missing"}
    data = torch.load(checkpoint, map_location="cpu")
    state = data.get("policy_state_dict")
    if not state:
        return "FAIL", {"reason": "policy_state_dict missing"}
    cfg = data.get("config", {})
    model = MultiDiscreteActorCritic(hidden_dim=int(data.get("hidden_dim", cfg.get("hidden_dim", 64))))
    model.load_state_dict(state)
    checkpoint_checksum = compute_state_dict_checksum(state)
    loaded_checksum = compute_state_dict_checksum(model)
    if checkpoint_checksum != loaded_checksum:
        return "FAIL", {"reason": "state checksum mismatch"}
    report = eval_checkpoint(checkpoint, episodes=eval_episodes)
    if "mean_episode_return" not in report or "mean_episode_length" not in report:
        return "FAIL", {"reason": "eval report missing required fields"}
    return "PASS", {
        "checksum": loaded_checksum,
        "eval_report": report,
    }


def validate_eval_report(report: dict[str, Any], *, episodes: int) -> tuple[str, dict[str, Any]]:
    if report.get("episodes") != episodes:
        return "FAIL", {"reason": "episode count mismatch", "report": report}
    for key in ("mean_episode_return", "mean_episode_length"):
        if key not in report:
            return "FAIL", {"reason": f"{key} missing"}
    if "mean_metrics" not in report:
        return "FAIL", {"reason": "mean_metrics missing"}
    return "PASS", report


def run_acceptance(
    *,
    config: str | Path,
    seed: int,
    eval_episodes: int,
    device: str,
) -> dict[str, Any]:
    checks: dict[str, str] = {}
    details: dict[str, Any] = {}
    artifacts: dict[str, str] = {}

    status, detail = check_same_seed_reproducibility(seed)
    checks["same_seed_reproducibility"] = status
    details["same_seed_reproducibility"] = detail

    status, detail = check_forced_move_fire_action(seed)
    checks["move_fire_step"] = status
    checks["raw_and_decoded_action_visible"] = status
    details["move_fire_step"] = detail
    sample = {
        "raw_action": detail.get("raw_action", {"move": 5, "aim": 10, "fire": 1}),
        "decoded_action": detail.get("decoded_action", {}),
    }

    status, detail = check_decode_bounds()
    checks["decode_bounds"] = status
    details["decode_bounds"] = detail

    cfg = load_config(config, smoke=True)
    cfg.seed = seed
    cfg.device = device
    cfg.total_steps = max(cfg.total_steps, cfg.rollout_steps * 2)
    cfg.max_episode_steps = min(cfg.max_episode_steps, 10)
    result = train_ppo(cfg)
    artifacts = {
        "run_dir": result["run_dir"],
        "checkpoint": result["checkpoint"],
        "checkpoint_latest": result.get("checkpoint_latest", result["checkpoint"]),
        "checkpoint_selected": result.get("checkpoint_selected", result["checkpoint"]),
        "checkpoint_min_reward": result.get("checkpoint_min_reward", result["checkpoint"]),
        "checkpoint_max_reward": result.get("checkpoint_max_reward", result["checkpoint"]),
        "selected_reward_checkpoint": result.get("selected_reward_checkpoint", ""),
        "metrics_csv": result["metrics_csv"],
    }

    status, detail = validate_checkpoint_load(result["checkpoint"], eval_episodes=eval_episodes)
    checks["checkpoint_load_eval"] = status
    details["checkpoint_load_eval"] = detail

    status, detail = validate_metrics_csv(result["metrics_csv"], min_rows=2)
    checks["metrics_csv_rows"] = status
    details["metrics_csv_rows"] = detail

    eval_report = eval_checkpoint(result["checkpoint"], episodes=eval_episodes)
    status, detail = validate_eval_report(eval_report, episodes=eval_episodes)
    checks["eval_10_episodes"] = status
    details["eval_10_episodes"] = detail

    report = {
        "status": "PASS" if all(value == "PASS" for value in checks.values()) else "FAIL",
        "checks": checks,
        "artifacts": artifacts,
        "sample": sample,
        "details": details,
    }
    if device == "cuda":
        report["warnings"] = [
            "CUDA was requested. CPU reproducibility remains the merge gate; minor CUDA floating variance is not treated as policy quality failure."
        ]
    return report


def normalize_decoded_action(decoded: dict[str, Any]) -> dict[str, float | int]:
    return {
        "move_x": float(decoded["moveX"]),
        "move_y": float(decoded["moveY"]),
        "aim_x": float(decoded["aimX"]),
        "aim_y": float(decoded["aimY"]),
        "fire": int(decoded["fire"]),
    }


def _action_to_ints(action: dict[str, torch.Tensor]) -> dict[str, int]:
    return {key: int(value.detach().to("cpu").reshape(-1)[0].item()) for key, value in action.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PR3 PPO smoke acceptance checks.")
    parser.add_argument("--config", default="experiment/configs/ppo_smoke.yaml")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    args = parser.parse_args()
    report = run_acceptance(
        config=args.config,
        seed=args.seed,
        eval_episodes=args.eval_episodes,
        device=args.device,
    )
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
