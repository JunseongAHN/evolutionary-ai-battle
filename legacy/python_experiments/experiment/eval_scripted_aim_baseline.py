from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from experiment.analyze_local_combat_eval import analyze_result
    from experiment.baselines.scripted_combat_policies import ScriptedAimAtEnemyPolicy, VALID_MODES
    from experiment.training.cpc_actions import decode_action
    from experiment.training.cpc_env import CPCEnv
except ModuleNotFoundError:
    EXPERIMENT_ROOT = Path(__file__).resolve().parent
    REPO_ROOT = EXPERIMENT_ROOT.parent
    for path in (EXPERIMENT_ROOT, REPO_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from analyze_local_combat_eval import analyze_result
    from baselines.scripted_combat_policies import ScriptedAimAtEnemyPolicy, VALID_MODES
    from training.cpc_actions import decode_action
    from training.cpc_env import CPCEnv


DIRECTIONS = ("right", "left", "up", "down", "upper_right", "lower_right", "upper_left", "lower_left")


def run_scripted_baseline(
    *,
    episodes: int,
    max_steps: int,
    seed: int,
    mode: str,
    fixed_enemy_direction: str | None = None,
    render_pygame: bool = False,
) -> dict[str, Any]:
    policy = ScriptedAimAtEnemyPolicy(mode=mode)
    viewer = _create_viewer(render_pygame)
    result_episodes = []
    for episode_index in range(max(1, int(episodes))):
        env = CPCEnv(
            seed=int(seed) + episode_index,
            max_steps=int(max_steps),
            stage="local_combat",
            shrink_safe_zone=False,
            use_zone_reward=False,
            randomize_enemy_spawn_direction=fixed_enemy_direction is None,
            enemy_spawn_direction=fixed_enemy_direction,
        )
        observation = env.reset(seed=int(seed) + episode_index)
        total_reward = 0.0
        done = False
        steps = []
        while not done and env.step_count < int(max_steps):
            step_index = env.step_count
            scripted = policy.act_with_diagnostics(observation, env)
            action = scripted.action
            decoded = decode_action(action)
            state_before = _state_snapshot(env)
            observation, reward, done, info = env.step(action)
            total_reward += float(reward)
            step = _compact_step(
                step_index=step_index,
                action=action,
                decoded=decoded,
                reward=reward,
                info=info,
                env=env,
                state_before=state_before,
                policy_diagnostics=scripted.diagnostics,
            )
            steps.append(step)
            if viewer is not None and not viewer.render_step(step["env"]["state"], step):
                break
        result_episodes.append(
            {
                "episode_index": episode_index,
                "initial_observation": {},
                "steps": steps,
                "episode_return": {"agent": total_reward},
                "episode_length": len(steps),
                "final_metrics": env.metrics.summary(),
                "stopped_early": False,
            }
        )
    if viewer is not None:
        viewer.close()
    return {
        "schema_version": "cpc-common-v0",
        "source": "eval_scripted_aim_baseline",
        "config": {
            "episodes": int(episodes),
            "max_steps": int(max_steps),
            "seed": int(seed),
            "mode": mode,
            "fixed_enemy_direction": fixed_enemy_direction,
        },
        "episodes": result_episodes,
    }


def sweep_directions(*, episodes: int, max_steps: int, seed: int, mode: str) -> list[dict[str, Any]]:
    rows = []
    for direction in DIRECTIONS:
        result = run_scripted_baseline(
            episodes=episodes,
            max_steps=max_steps,
            seed=seed,
            mode=mode,
            fixed_enemy_direction=direction,
        )
        analysis = analyze_result(result)
        aggregate = analysis.get("aggregate", {})
        rows.append(
            {
                "direction": direction,
                "shot_fired_count": aggregate.get("shot_fired_count", 0.0),
                "bullet_hit_count": aggregate.get("self_bullet_hit_count", 0.0),
                "hit_ratio": aggregate.get("hit_ratio", 0.0),
                "damage_dealt": aggregate.get("damage_dealt", 0.0),
                "enemy_dead": aggregate.get("enemy_dead", 0.0),
                "warnings": aggregate.get("warnings", {}),
            }
        )
    return rows


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    analysis = analyze_result(result)
    aggregate = analysis.get("aggregate", {})
    summary = {
        "episode_count": analysis.get("episode_count", 0),
        "mean_return": aggregate.get("total_reward", 0.0),
        "mean_damage_dealt_ratio": aggregate.get("damage_dealt_ratio", 0.0),
        "mean_damage_taken_ratio": aggregate.get("damage_taken_ratio", 0.0),
        "mean_damage_trade_ratio": aggregate.get("damage_trade_ratio", 0.0),
        "mean_fire_requested_count": aggregate.get("fire_requested_count", 0.0),
        "mean_shot_fired_count": aggregate.get("shot_fired_count", 0.0),
        "mean_self_bullet_hit_count": aggregate.get("self_bullet_hit_count", 0.0),
        "mean_self_bullet_expired_count": aggregate.get("self_bullet_expired_count", 0.0),
        "mean_self_bullet_missed_count": aggregate.get("self_bullet_missed_count", 0.0),
        "mean_self_bullet_alive_at_episode_end": aggregate.get("self_bullet_alive_at_episode_end", 0.0),
        "mean_bullet_hit_per_shot": aggregate.get("bullet_hit_per_shot", 0.0),
        "mean_hit_ratio": aggregate.get("hit_ratio", 0.0),
        "mean_shot_exact_aim_rate": aggregate.get("shot_exact_aim_rate", 0.0),
        "mean_shot_within_1_bin_rate": aggregate.get("shot_within_1_bin_rate", 0.0),
        "mean_avg_distance_to_enemy": aggregate.get("avg_distance_to_enemy", 0.0),
        "enemy_dead_rate": aggregate.get("enemy_dead", 0.0),
        "self_dead_rate": aggregate.get("self_dead", 0.0),
        "warnings": aggregate.get("warnings", {}),
        "diagnosis": diagnose(result, analysis),
    }
    return summary


def diagnose(result: dict[str, Any], analysis: dict[str, Any]) -> list[str]:
    issues = []
    aggregate = analysis.get("aggregate", {})
    if float(aggregate.get("self_bullet_hit_count", 0.0)) > 0.0:
        return ["scripted baseline hit at least one enemy; projectile mechanics can produce hits"]
    shot_steps = [step for ep in result.get("episodes", []) for step in ep.get("steps", []) if step.get("fire", {}).get("shot_fired")]
    if not shot_steps:
        issues.append("no actual shots fired; check can_fire/fire logging")
        return issues
    if any(step.get("scripted", {}).get("selected_aim_bin") != step.get("scripted", {}).get("ideal_aim_bin") for step in shot_steps):
        issues.append("aim bin conversion mismatch: selected_aim_bin != ideal_aim_bin")
    if any(int(step.get("aim", {}).get("aim_bin_error", 99)) != 0 for step in shot_steps):
        issues.append("env aim debug disagrees with scripted aim: aim_bin_error is nonzero")
    if not any(any(event.get("type") == "bullet_spawned" for event in step.get("events", [])) for step in shot_steps):
        issues.append("shot_fired happened without bullet_spawned event; possible logging bug")
    if float(aggregate.get("self_bullet_missed_count", 0.0)) > 0.0:
        issues.append("bullets spawned but expired; possible range, enemy movement, or aim-at-fire-time issue")
    if float(aggregate.get("self_bullet_alive_at_episode_end", 0.0)) > 0.0:
        issues.append("bullets still alive at episode end; increase max_steps or inspect projectile speed/range")
    if float(aggregate.get("damage_dealt", 0.0)) == 0.0:
        issues.append("no damage dealt; inspect bullet path, segment collision, hit radius, and target movement")
    return issues or ["no obvious diagnosis; inspect saved per-shot diagnostics"]


def print_summary(summary: dict[str, Any]) -> None:
    print("Scripted aim-at-enemy baseline summary")
    for key, value in summary.items():
        if key == "diagnosis":
            continue
        print(f"{key}: {value}")
    print("diagnosis:")
    for issue in summary["diagnosis"]:
        print(f"- {issue}")


def print_sweep(rows: list[dict[str, Any]]) -> None:
    print("| direction | shot_fired_count | bullet_hit_count | hit_ratio | damage_dealt | enemy_dead | warnings |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for row in rows:
        print(
            f"| {row['direction']} | {float(row['shot_fired_count']):.2f} | "
            f"{float(row['bullet_hit_count']):.2f} | {float(row['hit_ratio']):.3f} | "
            f"{float(row['damage_dealt']):.2f} | {float(row['enemy_dead']):.2f} | {row['warnings']} |"
        )


def _compact_step(
    *,
    step_index: int,
    action: dict[str, int],
    decoded: dict[str, Any],
    reward: float,
    info: dict[str, Any],
    env: CPCEnv,
    state_before: dict[str, Any],
    policy_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    spawn_event = next((event for event in info.get("bullet_events", []) if event.get("type") == "bullet_spawned"), {})
    step = {
        "t": step_index,
        "action": {
            "raw": dict(action),
            "decoded": {
                "move": {"x": decoded["moveX"], "y": decoded["moveY"]},
                "aim": {"x": decoded["aimX"], "y": decoded["aimY"]},
                "fire": decoded["fire"],
            },
        },
        "scripted": {
            **policy_diagnostics,
            "bullet_spawn_pos": spawn_event.get("pos"),
            "bullet_dir": {"x": decoded["aimX"], "y": decoded["aimY"]} if spawn_event else None,
        },
        "aim": {
            "aim_bin": info.get("aim_debug", {}).get("aim_bin"),
            "ideal_aim_bin": info.get("aim_debug", {}).get("ideal_aim_bin"),
            "aim_bin_error": info.get("aim_debug", {}).get("aim_bin_error"),
            "alignment": info.get("aim_debug", {}).get("aim_alignment", 0.0),
            "angle_error_deg": info.get("aim_debug", {}).get("angle_error_deg", 0.0),
        },
        "fire": {
            "requested": info.get("fire", {}).get("fire_requested", False),
            "shot_fired": info.get("fire", {}).get("shot_fired", False),
            "blocked_reason": info.get("fire", {}).get("fire_blocked_reason"),
            "cooldown_before": info.get("fire", {}).get("cooldown_remaining_steps_before"),
            "cooldown_after": info.get("fire", {}).get("cooldown_remaining_steps_after"),
        },
        "range": info.get("range_debug", {}),
        "events": info.get("bullet_events", []),
        "bullets": info.get("bullets", []),
        "reward": float(reward),
        "reward_components": info.get("reward_components", {}),
        "metrics_delta": {
            "damage_dealt_delta": info.get("damage_delta", {}).get("enemy_hp", 0.0),
            "damage_taken_delta": info.get("damage_delta", {}).get("self_hp", 0.0),
        },
        "state_after": {
            "self": {"hp": env.state.get("self_hp"), "pos": env.state.get("self_pos")},
            "enemy": {"hp": env.state.get("enemy_hp"), "pos": env.state.get("enemy_pos")},
            "dist": {"enemy": info.get("range_debug", {}).get("distance_to_enemy")},
        },
        "env": {
            "state_before_step": state_before,
            "state": _viewer_state(env, info),
            "info": info,
        },
    }
    return step


def _state_snapshot(env: CPCEnv) -> dict[str, Any]:
    return {
        "self_pos": dict(env.state["self_pos"]),
        "ally_pos": dict(env.state["ally_pos"]),
        "enemy_pos": dict(env.state["enemy_pos"]),
        "self_hp": float(env.state["self_hp"]),
        "ally_hp": float(env.state["ally_hp"]),
        "enemy_hp": float(env.state["enemy_hp"]),
        "projectiles": [dict(bullet) for bullet in env.projectiles],
    }


def _viewer_state(env: CPCEnv, info: dict[str, Any]) -> dict[str, Any]:
    return {
        **_state_snapshot(env),
        "map": {"width": env.width, "height": env.height},
        "safe_zone": info.get("safe_zone", {}),
        "aim_debug": info.get("aim_debug", {}),
        "zone_debug": info.get("zone_debug", {}),
        "range_debug": info.get("range_debug", {}),
        "bullets": info.get("bullets", []),
    }


def _create_viewer(render_pygame: bool) -> Any | None:
    if not render_pygame:
        return None
    try:
        from experiment.gui.pygame_viewer import PygameCPCViewer
    except ImportError as exc:
        raise ImportError("pygame is required for --render-pygame. Install with: pip install pygame") from exc
    return PygameCPCViewer(fps=10)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a scripted aim-at-enemy Stage 1 projectile baseline.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=VALID_MODES, default="stand_still")
    parser.add_argument("--fixed-enemy-direction", choices=DIRECTIONS)
    parser.add_argument("--save-result")
    parser.add_argument("--save-result-mode", choices=("compact",), default="compact")
    parser.add_argument("--render-pygame", action="store_true")
    parser.add_argument("--sweep-directions", action="store_true")
    args = parser.parse_args()

    if args.sweep_directions:
        rows = sweep_directions(episodes=args.episodes, max_steps=args.max_steps, seed=args.seed, mode=args.mode)
        print_sweep(rows)
        return

    result = run_scripted_baseline(
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        mode=args.mode,
        fixed_enemy_direction=args.fixed_enemy_direction,
        render_pygame=args.render_pygame,
    )
    summary = summarize(result)
    print_summary(summary)
    if args.save_result:
        path = Path(args.save_result)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"saved_result: {path}")


if __name__ == "__main__":
    main()
