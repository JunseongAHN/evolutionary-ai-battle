from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parent
for path in (EXPERIMENT_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from checkpointing import load_checkpoint
from core.env_core import PythonBattleCoreEnv
from core.schema import AgentId, BattleAction, MultiAgentAction, SCHEMA_VERSION
from debug.render_state import draw_debug_view
from debug.state_inspector import summarize_events, summarize_snapshot
from training.cpc_actions import decode_action
from training.ppo_policy import MultiDiscreteActorCritic


DEFAULT_GOOGLE_DRIVE_CHECKPOINT = (
    "/content/drive/MyDrive/repos/evolutionary-ai-battle/"
    "experiment/runs/ppo_smoke_20260622_105638/checkpoint_latest.pt"
)
BEST_CHECKPOINT_NAMES = (
    "checkpoint_selected.pt",
    "checkpoint_max_reward.pt",
    "checkpoint_latest.pt",
    "checkpoint.pt",
)


def mount_google_drive(mount_point: str = "/content/drive", *, force_remount: bool = False) -> bool:
    """Mount Google Drive when running inside Colab.

    Returns True if the Colab drive helper was available and called, otherwise
    False. Local runs simply return False and continue with normal paths.
    """
    try:
        from google.colab import drive  # type: ignore
    except Exception:
        return False

    drive.mount(mount_point, force_remount=force_remount)
    return True


def _resolve_drive_aware_path(path: str | Path | None, *, mount_drive: bool) -> Path:
    if path is None:
        raise ValueError("path must not be None")

    raw_path = str(path)
    candidate = Path(raw_path).expanduser()
    if candidate.exists():
        return candidate

    if not _looks_like_google_drive_path(raw_path):
        return candidate

    if mount_drive:
        mounted = mount_google_drive()
        candidate = Path(raw_path).expanduser()
        if candidate.exists() or mounted:
            return candidate

    local_candidate = _google_drive_colab_path_to_local_path(raw_path)
    if local_candidate is not None:
        return local_candidate

    return candidate


def resolve_checkpoint_path(
    checkpoint: str | Path | None = None,
    *,
    run_dir: str | Path | None = None,
    mount_drive: bool = True,
) -> Path:
    """Resolve an explicit checkpoint or the best-known checkpoint in a run dir."""
    if checkpoint is None and run_dir is None:
        checkpoint = DEFAULT_GOOGLE_DRIVE_CHECKPOINT
    if checkpoint is not None:
        path = _resolve_drive_aware_path(checkpoint, mount_drive=mount_drive)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    directory = _resolve_drive_aware_path(run_dir, mount_drive=mount_drive)
    if not directory.exists():
        raise FileNotFoundError(f"Run directory not found: {directory}")

    for name in BEST_CHECKPOINT_NAMES:
        candidate = directory / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No checkpoint found in {directory}; tried {', '.join(BEST_CHECKPOINT_NAMES)}"
    )


def load_policy_from_checkpoint(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "auto",
) -> tuple[MultiDiscreteActorCritic, dict[str, Any], torch.device]:
    resolved_device = _resolve_device(str(device)) if isinstance(device, str) else device
    checkpoint_data = load_checkpoint(checkpoint, map_location=resolved_device)
    cfg = checkpoint_data.get("config", {})
    hidden_dim = int(checkpoint_data.get("hidden_dim") or cfg.get("hidden_dim", 64))
    policy = MultiDiscreteActorCritic(hidden_dim=hidden_dim)
    policy.load_state_dict(checkpoint_data["policy_state_dict"])
    policy.to(resolved_device)
    policy.eval()
    return policy, checkpoint_data, resolved_device


@torch.no_grad()
def run_model_gameplay(
    pt_file: str | Path | None = None,
    *,
    checkpoint: str | Path | None = None,
    run_dir: str | Path | None = None,
    seed: int = 0,
    max_steps: int | None = None,
    device: str | torch.device = "auto",
    deterministic: bool = True,
    gui: bool = False,
    width: int = 1000,
    height: int = 700,
    fps: int = 15,
    highlight_agent_id: AgentId = "team-a-0",
    print_every: int = 25,
) -> dict[str, Any]:
    if pt_file is not None and checkpoint is not None:
        raise ValueError("Use either pt_file or checkpoint, not both.")
    checkpoint = pt_file if pt_file is not None else checkpoint
    checkpoint_path = resolve_checkpoint_path(checkpoint, run_dir=run_dir)
    policy, checkpoint_data, resolved_device = load_policy_from_checkpoint(
        checkpoint_path,
        device=device,
    )
    env = PythonBattleCoreEnv()
    cfg = checkpoint_data.get("config", {})
    if max_steps is None:
        max_steps = int(cfg.get("max_episode_steps", env.max_steps))
    max_steps = max(1, min(int(max_steps), int(env.max_steps)))

    observations = env.reset(seed=seed)
    highlight_agent_id = _coerce_agent_id(env.agent_ids, highlight_agent_id)
    last_snapshot = env._snapshot([])
    last_actions = build_model_multi_agent_action(
        env=env,
        observations=observations,
        snapshot=last_snapshot,
        policy=policy,
        device=resolved_device,
        deterministic=deterministic,
        policy_id=str(checkpoint_path),
    )

    if gui:
        return _run_model_gameplay_gui(
            env=env,
            observations=observations,
            policy=policy,
            device=resolved_device,
            deterministic=deterministic,
            policy_id=str(checkpoint_path),
            max_steps=max_steps,
            width=width,
            height=height,
            fps=fps,
            highlight_agent_id=highlight_agent_id,
            checkpoint_path=checkpoint_path,
            checkpoint_data=checkpoint_data,
        )

    recent_events = []
    terminated = False
    truncated = False
    while env.step_index < max_steps and not (terminated or truncated):
        last_actions = build_model_multi_agent_action(
            env=env,
            observations=observations,
            snapshot=last_snapshot,
            policy=policy,
            device=resolved_device,
            deterministic=deterministic,
            policy_id=str(checkpoint_path),
        )
        step = env.step(last_actions)
        observations = step["observations"]
        last_snapshot = step["info"]["snapshot"]
        recent_events = step["info"]["events"]
        terminated = bool(step["terminated"])
        truncated = bool(step["truncated"] or env.step_index >= max_steps)
        if print_every > 0 and (recent_events or env.step_index % print_every == 0):
            print(summarize_snapshot(last_snapshot))
            if recent_events:
                print(summarize_events(recent_events))

    report = _build_report(
        env=env,
        checkpoint_path=checkpoint_path,
        checkpoint_data=checkpoint_data,
        terminated=terminated,
        truncated=truncated,
        max_steps=max_steps,
        last_actions=last_actions,
    )
    print(json.dumps(report, indent=2))
    return report


def build_model_multi_agent_action(
    *,
    env: PythonBattleCoreEnv,
    observations: Mapping[AgentId, Mapping[str, Any]],
    snapshot: Mapping[str, Any],
    policy: MultiDiscreteActorCritic,
    device: torch.device,
    deterministic: bool,
    policy_id: str,
) -> MultiAgentAction:
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": env.episode_id,
        "step": env.step_index,
        "actions": {
            agent_id: build_model_action(
                episode_id=env.episode_id,
                step=env.step_index,
                agent_id=agent_id,
                raw_action=_select_raw_action(
                    policy,
                    _policy_features_from_core_observation(
                        observations[agent_id],
                        snapshot=snapshot,
                        agent_id=agent_id,
                        device=device,
                    ),
                    deterministic=deterministic,
                ),
                policy_id=policy_id,
            )
            for agent_id in env.agent_ids
        },
    }


def build_model_action(
    *,
    episode_id: str,
    step: int,
    agent_id: AgentId,
    raw_action: Mapping[str, int],
    policy_id: str,
) -> BattleAction:
    decoded = decode_action(raw_action)
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "step": step,
        "agent_id": agent_id,
        "action": {
            "move_x": float(decoded["moveX"]),
            "move_y": float(decoded["moveY"]),
            "aim_x": float(decoded["aimX"]),
            "aim_y": float(decoded["aimY"]),
            "fire": float(decoded["fire"]),
        },
        "source": {
            "policy_type": "ppo_checkpoint",
            "policy_id": policy_id,
            "raw_action": dict(raw_action),
        },
    }


def _policy_features_from_core_observation(
    observation: Mapping[str, Any],
    *,
    snapshot: Mapping[str, Any],
    agent_id: AgentId,
    device: torch.device,
) -> torch.Tensor:
    self_obs = observation["self"]
    ally = _nearest_entity(observation.get("visible_allies", []), self_obs["position"])
    enemy = _nearest_entity(observation.get("visible_enemies", []), self_obs["position"])
    ally_pos = _absolute_entity_position(self_obs["position"], ally)
    enemy_pos = _absolute_entity_position(self_obs["position"], enemy)
    ally_hp = float(ally.get("hp", 0.0)) if ally else 0.0
    enemy_hp = float(enemy.get("hp", 0.0)) if enemy else 0.0
    distance_to_ally = _distance(self_obs["position"], ally_pos) if ally else 1000.0
    ally_under_pressure = _ally_under_pressure(snapshot, agent_id)

    values = [
        float(self_obs["hp"]) / 100.0,
        ally_hp / 100.0,
        enemy_hp / 100.0,
        float(self_obs["position"]["x"]) / 1000.0,
        float(self_obs["position"]["y"]) / 1000.0,
        float(ally_pos["x"]) / 1000.0,
        float(ally_pos["y"]) / 1000.0,
        float(enemy_pos["x"]) / 1000.0,
        float(enemy_pos["y"]) / 1000.0,
        distance_to_ally / 1000.0,
        1.0 if ally_under_pressure else 0.0,
        1.0 if float(self_obs["hp"]) <= 35.0 else 0.0,
        float(observation["step"]) / 100.0,
    ]
    return torch.tensor([values], dtype=torch.float32, device=device)


def _select_raw_action(
    policy: MultiDiscreteActorCritic,
    features: torch.Tensor,
    *,
    deterministic: bool,
) -> dict[str, int]:
    if deterministic:
        move_logits, aim_logits, fire_logits, _ = policy(features)
        return {
            "move": int(move_logits.argmax(dim=-1).reshape(-1)[0].item()),
            "aim": int(aim_logits.argmax(dim=-1).reshape(-1)[0].item()),
            "fire": int(fire_logits.argmax(dim=-1).reshape(-1)[0].item()),
        }

    output = policy.sample_action(features)
    return {
        "move": int(output.action["move"].reshape(-1)[0].item()),
        "aim": int(output.action["aim"].reshape(-1)[0].item()),
        "fire": int(output.action["fire"].reshape(-1)[0].item()),
    }


def _run_model_gameplay_gui(
    *,
    env: PythonBattleCoreEnv,
    observations: Mapping[AgentId, Mapping[str, Any]],
    policy: MultiDiscreteActorCritic,
    device: torch.device,
    deterministic: bool,
    policy_id: str,
    max_steps: int,
    width: int,
    height: int,
    fps: int,
    highlight_agent_id: AgentId,
    checkpoint_path: Path,
    checkpoint_data: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        import pygame
    except ImportError:
        print("pygame is required for --gui. Install it with: pip install pygame")
        return run_model_gameplay(
            checkpoint_path,
            seed=int(env.episode_id.rsplit("-", 1)[-1]),
            max_steps=max_steps,
            device=device,
            deterministic=deterministic,
            gui=False,
            highlight_agent_id=highlight_agent_id,
        )

    pygame.init()
    pygame.display.set_caption("CPC PPO Checkpoint Gameplay")
    screen = pygame.display.set_mode((width, height))
    font = pygame.font.SysFont("consolas", 16)
    small_font = pygame.font.SysFont("consolas", 12)
    clock = pygame.time.Clock()
    paused = False
    running = True
    last_snapshot = env._snapshot([])
    recent_events = []
    terminated = False
    truncated = False
    last_actions = build_model_multi_agent_action(
        env=env,
        observations=observations,
        snapshot=last_snapshot,
        policy=policy,
        device=device,
        deterministic=deterministic,
        policy_id=policy_id,
    )

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_TAB:
                    highlight_agent_id = _next_agent_id(env.agent_ids, highlight_agent_id)

        if not paused and env.step_index < max_steps and not (terminated or truncated):
            last_actions = build_model_multi_agent_action(
                env=env,
                observations=observations,
                snapshot=last_snapshot,
                policy=policy,
                device=device,
                deterministic=deterministic,
                policy_id=policy_id,
            )
            step = env.step(last_actions)
            observations = step["observations"]
            last_snapshot = step["info"]["snapshot"]
            recent_events = step["info"]["events"]
            terminated = bool(step["terminated"])
            truncated = bool(step["truncated"] or env.step_index >= max_steps)

        draw_debug_view(
            pygame,
            screen,
            font,
            small_font,
            last_snapshot,
            highlight_agent_id,
            last_actions["actions"][highlight_agent_id],
            recent_events,
            terminated,
            truncated,
            fire_range=env.fire_range,
        )
        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()
    report = _build_report(
        env=env,
        checkpoint_path=checkpoint_path,
        checkpoint_data=checkpoint_data,
        terminated=terminated,
        truncated=truncated,
        max_steps=max_steps,
        last_actions=last_actions,
    )
    print(json.dumps(report, indent=2))
    return report


def _build_report(
    *,
    env: PythonBattleCoreEnv,
    checkpoint_path: Path,
    checkpoint_data: Mapping[str, Any],
    terminated: bool,
    truncated: bool,
    max_steps: int,
    last_actions: MultiAgentAction,
) -> dict[str, Any]:
    alive_by_team: dict[str, int] = {}
    agents = {}
    for agent_id, agent in env.agents.items():
        agents[agent_id] = {
            "team_id": agent["team_id"],
            "hp": float(agent["hp"]),
            "alive": bool(agent["alive"]),
            "position": dict(agent["position"]),
            "last_raw_action": last_actions["actions"][agent_id]["source"].get("raw_action"),
        }
        if agent["alive"]:
            alive_by_team[agent["team_id"]] = alive_by_team.get(agent["team_id"], 0) + 1

    winner = None
    if len(alive_by_team) == 1:
        winner = next(iter(alive_by_team))

    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_update": checkpoint_data.get("update"),
        "checkpoint_global_step": checkpoint_data.get("global_step"),
        "selection_metric": checkpoint_data.get("selection_metric"),
        "selection_mode": checkpoint_data.get("selection_mode"),
        "selection_value": checkpoint_data.get("selection_value"),
        "episode_id": env.episode_id,
        "steps": env.step_index,
        "max_steps": max_steps,
        "terminated": terminated,
        "truncated": truncated,
        "winner": winner,
        "alive_by_team": alive_by_team,
        "agents": agents,
    }


def _nearest_entity(entities: list[Mapping[str, Any]], self_position: Mapping[str, float]) -> Mapping[str, Any] | None:
    alive = [entity for entity in entities if bool(entity.get("alive", True))]
    candidates = alive or list(entities)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda entity: _distance(
            self_position,
            _absolute_entity_position(self_position, entity),
        ),
    )


def _absolute_entity_position(
    self_position: Mapping[str, float],
    entity: Mapping[str, Any] | None,
) -> dict[str, float]:
    if not entity:
        return {"x": float(self_position["x"]), "y": float(self_position["y"])}
    if "position" in entity:
        return {
            "x": float(entity["position"]["x"]),
            "y": float(entity["position"]["y"]),
        }
    relative = entity.get("relative_position", {"x": 0.0, "y": 0.0})
    return {
        "x": float(self_position["x"]) + float(relative.get("x", 0.0)),
        "y": float(self_position["y"]) + float(relative.get("y", 0.0)),
    }


def _ally_under_pressure(snapshot: Mapping[str, Any], agent_id: AgentId) -> bool:
    agents = snapshot["agents"]
    actor = agents[agent_id]
    allies = [
        agent
        for other_id, agent in agents.items()
        if other_id != agent_id and agent["team_id"] == actor["team_id"] and agent["alive"]
    ]
    enemies = [
        agent
        for agent in agents.values()
        if agent["team_id"] != actor["team_id"] and agent["alive"]
    ]
    return any(_distance(ally["position"], enemy["position"]) <= 260.0 for ally in allies for enemy in enemies)


def _distance(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    return ((float(a["x"]) - float(b["x"])) ** 2 + (float(a["y"]) - float(b["y"])) ** 2) ** 0.5


def _google_drive_colab_path_to_local_path(path: str | Path) -> Path | None:
    suffix = _google_drive_my_drive_suffix(path)
    if suffix is None:
        return None

    roots = _local_google_drive_roots()
    for root in roots:
        candidate = root.joinpath(*suffix)
        if candidate.exists():
            return candidate
    if roots:
        return roots[0].joinpath(*suffix)
    return None


def _google_drive_my_drive_suffix(path: str | Path) -> list[str] | None:
    normalized = str(path).replace("\\", "/")
    prefixes = (
        "/content/drive/MyDrive/",
        "/content/drive/My Drive/",
        "content/drive/MyDrive/",
        "content/drive/My Drive/",
    )
    for prefix in prefixes:
        if normalized.startswith(prefix):
            suffix = normalized[len(prefix):]
            return [part for part in suffix.split("/") if part]
    return None


def _local_google_drive_roots() -> list[Path]:
    roots: list[Path] = []
    for env_name in ("GOOGLE_DRIVE_ROOT", "GDRIVE_ROOT", "MYDRIVE_ROOT"):
        value = os.environ.get(env_name)
        if value:
            roots.append(Path(value).expanduser())

    home = Path.home()
    roots.extend(
        [
            home / "Google Drive" / "My Drive",
            home / "Google Drive" / "MyDrive",
            home / "My Drive",
            home / "MyDrive",
        ]
    )
    if os.name == "nt":
        roots.extend(Path(f"{letter}:\\My Drive") for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ")
        roots.extend(Path(f"{letter}:\\MyDrive") for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ")

    unique_roots = []
    seen = set()
    for root in roots:
        key = str(root).lower()
        if key not in seen:
            unique_roots.append(root)
            seen.add(key)
    return unique_roots


def _looks_like_google_drive_path(path: str | Path) -> bool:
    normalized = str(path).replace("\\", "/")
    return (
        normalized.startswith("/content/drive/")
        or normalized.startswith("content/drive/")
        or "/MyDrive/" in normalized
        or "/My Drive/" in normalized
    )


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


def _coerce_agent_id(agent_ids: list[str], requested: str) -> str:
    return requested if requested in agent_ids else agent_ids[0]


def _next_agent_id(agent_ids: list[str], current: str) -> str:
    index = agent_ids.index(current) if current in agent_ids else -1
    return agent_ids[(index + 1) % len(agent_ids)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a PPO checkpoint and play it in the debug battle env.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sampled", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--highlight-agent-id", default="team-a-0")
    parser.add_argument("--print-every", type=int, default=25)
    args = parser.parse_args()
    run_model_gameplay(
        checkpoint=args.checkpoint,
        run_dir=args.run_dir,
        seed=args.seed,
        max_steps=args.max_steps,
        device=args.device,
        deterministic=not args.sampled,
        gui=args.gui,
        fps=args.fps,
        highlight_agent_id=args.highlight_agent_id,
        print_every=args.print_every,
    )


if __name__ == "__main__":
    main()
