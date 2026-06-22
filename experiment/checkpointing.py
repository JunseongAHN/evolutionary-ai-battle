from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import torch


LATEST_CHECKPOINT_NAME = "checkpoint_latest.pt"
MIN_REWARD_CHECKPOINT_NAME = "checkpoint_min_reward.pt"
MAX_REWARD_CHECKPOINT_NAME = "checkpoint_max_reward.pt"
SELECTED_CHECKPOINT_NAME = "checkpoint_selected.pt"
SELECTED_REWARD_METADATA_NAME = "selected_reward_checkpoint.json"


def save_checkpoint(
    path: str | Path,
    *,
    policy,
    optimizer=None,
    config: dict[str, Any] | None = None,
    update: int,
    global_step: int,
    metrics: dict[str, Any] | None = None,
    selection_metric: str,
    selection_mode: str,
    selection_value: float | None,
    action_metadata: dict[str, Any],
    observation_metadata: dict[str, Any],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "config": config or {},
        "update": int(update),
        "global_step": int(global_step),
        "metrics": metrics or {},
        "selection_metric": selection_metric,
        "selection_mode": selection_mode,
        "selection_value": None if selection_value is None else float(selection_value),
        "action_metadata": dict(action_metadata),
        "observation_metadata": dict(observation_metadata),
        "obs_dim": observation_metadata.get("obs_dim"),
        "hidden_dim": (config or {}).get("hidden_dim"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=map_location)
    if "policy_state_dict" not in payload:
        raise ValueError(f"Checkpoint is missing policy_state_dict: {checkpoint_path}")
    return payload


def save_selected_checkpoint_if_needed(
    *,
    run_dir: str | Path,
    latest_checkpoint: str | Path,
    selected_checkpoint: str | Path,
    metadata_path: str | Path,
    update: int,
    global_step: int,
    metrics: dict[str, Any],
    selection_metric: str,
    selection_mode: str,
    selection_value: float | None,
) -> tuple[bool, float | None]:
    if selection_mode not in {"min", "max"}:
        raise ValueError(f"selection_mode must be 'min' or 'max', got {selection_mode!r}")

    run_dir = Path(run_dir)
    selected_checkpoint = Path(selected_checkpoint)
    metadata_path = Path(metadata_path)
    previous = _read_selected_metadata(metadata_path)
    previous_value = previous.get("selection_value")

    is_selected = _is_better(selection_value, previous_value, selection_mode)
    current_selected_value = float(selection_value) if is_selected and selection_value is not None else previous_value
    if is_selected:
        selected_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(latest_checkpoint, selected_checkpoint)
        _write_selected_metadata(
            metadata_path,
            {
                "selection_metric": selection_metric,
                "selection_mode": selection_mode,
                "selection_value": current_selected_value,
                "selected_update": int(update),
                "selected_global_step": int(global_step),
                "checkpoint": str(selected_checkpoint),
                "checkpoint_latest": str(latest_checkpoint),
                "metrics": metrics,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
    elif not metadata_path.exists():
        _write_selected_metadata(
            metadata_path,
            {
                "selection_metric": selection_metric,
                "selection_mode": selection_mode,
                "selection_value": None,
                "selected_update": None,
                "selected_global_step": None,
                "checkpoint": str(selected_checkpoint),
                "checkpoint_latest": str(latest_checkpoint),
                "metrics": {},
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )

    return is_selected, None if current_selected_value is None else float(current_selected_value)


def selected_paths(run_dir: str | Path) -> dict[str, Path]:
    run_dir = Path(run_dir)
    return {
        "checkpoint_latest": run_dir / LATEST_CHECKPOINT_NAME,
        "checkpoint_min_reward": run_dir / MIN_REWARD_CHECKPOINT_NAME,
        "checkpoint_max_reward": run_dir / MAX_REWARD_CHECKPOINT_NAME,
        "checkpoint_selected": run_dir / SELECTED_CHECKPOINT_NAME,
        "selected_reward_checkpoint": run_dir / SELECTED_REWARD_METADATA_NAME,
    }


def _is_better(candidate: float | None, previous: Any, mode: str) -> bool:
    if candidate is None:
        return False
    candidate_value = float(candidate)
    if not math.isfinite(candidate_value):
        return False
    if previous is None:
        return True
    previous_value = float(previous)
    if mode == "min":
        return candidate_value < previous_value
    return candidate_value > previous_value


def _read_selected_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_selected_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
