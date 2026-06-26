from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Iterable

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parent
for path in (EXPERIMENT_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.env_core import PythonBattleCoreEnv
from core.schema import AgentId, BattleAction, MultiAgentAction, SCHEMA_VERSION, Vec2
from debug.state_inspector import summarize_agent, summarize_events, summarize_observation
from gui.geometry import screen_to_world
from gui.pygame_viewer import PygameCPCViewer


KEY_ALIASES = {
    "w": "w",
    "a": "a",
    "s": "s",
    "d": "d",
    "up": "w",
    "left": "a",
    "down": "s",
    "right": "d",
}

BULLET_SPEED = 45.0


def keyboard_to_move(keys: set[str]) -> tuple[float, float]:
    normalized = {KEY_ALIASES.get(key.lower(), key.lower()) for key in keys}
    move_x = (1.0 if "d" in normalized else 0.0) + (-1.0 if "a" in normalized else 0.0)
    move_y = (1.0 if "s" in normalized else 0.0) + (-1.0 if "w" in normalized else 0.0)
    length = math.hypot(move_x, move_y)
    if length > 1.0:
        move_x /= length
        move_y /= length
    return move_x, move_y


def mouse_to_aim(agent_position: Vec2, mouse_world_position: Vec2) -> tuple[float, float]:
    dx = float(mouse_world_position["x"]) - float(agent_position["x"])
    dy = float(mouse_world_position["y"]) - float(agent_position["y"])
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return 0.0, 0.0
    return dx / length, dy / length


def build_user_action(
    episode_id: str,
    step: int,
    agent_id: str,
    move_x: float,
    move_y: float,
    aim_x: float,
    aim_y: float,
    fire: float,
) -> BattleAction:
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "step": step,
        "agent_id": agent_id,
        "action": {
            "move_x": max(-1.0, min(1.0, float(move_x))),
            "move_y": max(-1.0, min(1.0, float(move_y))),
            "aim_x": max(-1.0, min(1.0, float(aim_x))),
            "aim_y": max(-1.0, min(1.0, float(aim_y))),
            "fire": 1.0 if fire > 0.5 else 0.0,
        },
        "source": {"policy_type": "user_controlled", "policy_id": "manual-debug"},
    }


def build_noop_action(episode_id: str, step: int, agent_id: str) -> BattleAction:
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "step": step,
        "agent_id": agent_id,
        "action": {"move_x": 0.0, "move_y": 0.0, "aim_x": 0.0, "aim_y": 0.0, "fire": 0.0},
        "source": {"policy_type": "random", "policy_id": "manual-debug-noop"},
    }


def spawn_debug_bullet(agent: dict, max_range: float, aim: Vec2 | None = None) -> dict | None:
    aim = aim or agent.get("aim", {"x": 0.0, "y": 0.0})
    aim_x = float(aim.get("x", 0.0))
    aim_y = float(aim.get("y", 0.0))
    length = math.hypot(aim_x, aim_y)
    if length <= 1e-6:
        return None

    position = agent["position"]
    return {
        "origin": {"x": float(position["x"]), "y": float(position["y"])},
        "position": {"x": float(position["x"]), "y": float(position["y"])},
        "velocity": {"x": aim_x / length, "y": aim_y / length},
        "distance": 0.0,
        "max_range": float(max_range),
    }


def update_debug_bullets(bullets: list[dict], speed: float = BULLET_SPEED) -> list[dict]:
    active_bullets = []
    for bullet in bullets:
        next_distance = float(bullet["distance"]) + speed
        if next_distance > float(bullet["max_range"]):
            continue

        origin = bullet["origin"]
        velocity = bullet["velocity"]
        bullet = {
            **bullet,
            "distance": next_distance,
            "position": {
                "x": float(origin["x"]) + (float(velocity["x"]) * next_distance),
                "y": float(origin["y"]) + (float(velocity["y"]) * next_distance),
            },
        }
        active_bullets.append(bullet)
    return active_bullets


def build_manual_multi_agent_action(
    *,
    episode_id: str,
    step: int,
    agent_ids: Iterable[AgentId],
    controlled_agent_id: AgentId,
    user_action: BattleAction,
) -> MultiAgentAction:
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "step": step,
        "actions": {
            agent_id: user_action if agent_id == controlled_agent_id else build_noop_action(episode_id, step, agent_id)
            for agent_id in agent_ids
        },
    }


def run_manual_debug(
    seed: int = 0,
    controlled_agent_id: str = "team-a-0",
    width: int = 1000,
    height: int = 700,
    fps: int = 30,
) -> None:
    try:
        viewer = PygameCPCViewer(width=width, height=height, fps=fps, title="CPC Core Env Manual Debug")
    except ImportError:
        print("pygame is required for manual env debugging. Install it with: pip install pygame")
        return
    pygame = viewer.pygame

    env = PythonBattleCoreEnv()
    observations = env.reset(seed=seed)
    controlled_agent_id = _coerce_agent_id(env.agent_ids, controlled_agent_id)
    print(f"Reset seed={seed} controlled_agent={controlled_agent_id}")

    paused = False
    single_step = False
    running = True
    last_step = None
    last_action = build_noop_action(env.episode_id, env.step_index, controlled_agent_id)
    last_snapshot = env._snapshot([])
    recent_events = []
    debug_bullets: list[dict] = []
    pending_fire_aim: Vec2 | None = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_world = _screen_to_core_world(viewer, event.pos, last_snapshot)
                agent_position = env.agents[controlled_agent_id]["position"]
                aim_x, aim_y = mouse_to_aim(agent_position, mouse_world)
                pending_fire_aim = {"x": aim_x, "y": aim_y}
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    observations = env.reset(seed=seed)
                    last_step = None
                    recent_events = []
                    debug_bullets = []
                    pending_fire_aim = None
                    last_action = build_noop_action(env.episode_id, env.step_index, controlled_agent_id)
                    last_snapshot = env._snapshot([])
                    print(f"Reset seed={seed} controlled_agent={controlled_agent_id}")
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_n:
                    single_step = True
                    paused = True
                elif event.key == pygame.K_TAB:
                    controlled_agent_id = _next_agent_id(env.agent_ids, controlled_agent_id)
                    print(f"Controlled agent: {controlled_agent_id}")

        keys = pygame.key.get_pressed()
        pressed = _pressed_key_names(pygame, keys)
        move_x, move_y = keyboard_to_move(pressed)
        mouse_world = _screen_to_core_world(viewer, pygame.mouse.get_pos(), last_snapshot)
        agent_position = env.agents[controlled_agent_id]["position"]
        aim_x, aim_y = mouse_to_aim(agent_position, mouse_world)
        action_aim_x = float(pending_fire_aim["x"]) if pending_fire_aim is not None else aim_x
        action_aim_y = float(pending_fire_aim["y"]) if pending_fire_aim is not None else aim_y
        fire = 1.0 if pending_fire_aim is not None else 0.0
        last_action = build_user_action(
            env.episode_id,
            env.step_index,
            controlled_agent_id,
            move_x,
            move_y,
            action_aim_x,
            action_aim_y,
            fire,
        )

        should_step = not paused or single_step
        if should_step and not (last_step and (last_step["terminated"] or last_step["truncated"])):
            debug_bullets = update_debug_bullets(debug_bullets)
            multi_action = build_manual_multi_agent_action(
                episode_id=env.episode_id,
                step=env.step_index,
                agent_ids=env.agent_ids,
                controlled_agent_id=controlled_agent_id,
                user_action=last_action,
            )
            last_step = env.step(multi_action)
            observations = last_step["observations"]
            recent_events = last_step["info"]["events"]
            last_snapshot = last_step["info"]["snapshot"]
            if pending_fire_aim is not None:
                bullet = spawn_debug_bullet(env.agents[controlled_agent_id], env.fire_range, pending_fire_aim)
                if bullet is not None:
                    debug_bullets.append(bullet)
            single_step = False
            pending_fire_aim = None
            if recent_events or env.step_index % max(1, fps) == 0:
                _print_debug_step(last_snapshot, controlled_agent_id, observations[controlled_agent_id], recent_events)

        terminated = bool(last_step and last_step["terminated"])
        truncated = bool(last_step and last_step["truncated"])
        env_state = _core_snapshot_to_viewer_state(last_snapshot, controlled_agent_id, debug_bullets)
        step_record = _manual_step_record(last_action, recent_events, terminated, truncated)
        running = running and viewer.render_step(env_state, step_record, handle_events=False)

    viewer.close()


def _screen_to_core_world(
    viewer: PygameCPCViewer,
    point: tuple[int, int],
    snapshot: dict,
) -> dict[str, float]:
    map_info = {
        "width": snapshot.get("map", {}).get("width", 1000.0),
        "height": snapshot.get("map", {}).get("height", 1000.0),
    }
    return screen_to_world(point, map_info, (viewer.width - viewer.panel_width, viewer.height), viewer.padding)


def _core_snapshot_to_viewer_state(snapshot: dict, controlled_agent_id: str, debug_bullets: list[dict] | None = None) -> dict:
    agents = {}
    controlled = snapshot["agents"][controlled_agent_id]
    for agent_id in snapshot["agent_ids"]:
        agent = snapshot["agents"][agent_id]
        role = "self"
        if agent_id != controlled_agent_id:
            role = "ally" if agent["team_id"] == controlled["team_id"] else "enemy"
        agents[agent_id] = {
            "position": dict(agent["position"]),
            "hp": float(agent["hp"]),
            "alive": bool(agent["alive"]),
            "aim": dict(agent.get("aim", {"x": 0.0, "y": 0.0})),
            "team_id": agent["team_id"],
            "role": role,
        }

    safe_zone = snapshot.get("safe_zone", {})
    return {
        "step": snapshot["step"],
        "step_count": snapshot["step"],
        "map": {
            "width": float(snapshot["map"]["width"]),
            "height": float(snapshot["map"]["height"]),
            "center": dict(safe_zone.get("center", {"x": 500.0, "y": 500.0})),
            "safe_radius": safe_zone.get("radius"),
        },
        "safe_zone": safe_zone,
        "combat": {"fire_range": 260.0},
        "agents": agents,
        "bullets": [_debug_bullet_to_viewer_bullet(bullet) for bullet in debug_bullets or []],
        "metrics": {},
        "events": snapshot.get("events", []),
    }


def _debug_bullet_to_viewer_bullet(bullet: dict) -> dict:
    position = dict(bullet["position"])
    origin = dict(bullet["origin"])
    velocity = bullet["velocity"]
    previous_distance = max(0.0, float(bullet["distance"]) - BULLET_SPEED)
    previous_pos = {
        "x": float(origin["x"]) + (float(velocity["x"]) * previous_distance),
        "y": float(origin["y"]) + (float(velocity["y"]) * previous_distance),
    }
    return {
        "bullet_id": bullet.get("bullet_id", "manual-debug-bullet"),
        "owner_id": "manual",
        "spawn_pos": origin,
        "previous_pos": previous_pos,
        "pos": position,
        "radius": 8.0,
        "alive": True,
    }


def _manual_step_record(
    action: BattleAction,
    recent_events: list,
    terminated: bool,
    truncated: bool,
) -> dict:
    body = action["action"]
    return {
        "agents": {
            "agent": {
                "agent_id": action["agent_id"],
                "decoded_action": {
                    "move_x": float(body["move_x"]),
                    "move_y": float(body["move_y"]),
                    "aim_x": float(body["aim_x"]),
                    "aim_y": float(body["aim_y"]),
                    "fire": float(body["fire"]),
                },
                "raw_action": {},
                "policy_debug": {},
            }
        },
        "env": {
            "rewards": {"agent": 0.0},
            "terminated": terminated,
            "truncated": truncated,
            "done": terminated or truncated,
            "info": {
                "reward_components": {},
                "metrics": {},
                "events": recent_events,
            },
        },
    }


def _pressed_key_names(pygame, keys) -> set[str]:
    names = set()
    if keys[pygame.K_w]:
        names.add("w")
    if keys[pygame.K_a]:
        names.add("a")
    if keys[pygame.K_s]:
        names.add("s")
    if keys[pygame.K_d]:
        names.add("d")
    return names


def _coerce_agent_id(agent_ids: list[str], requested: str) -> str:
    if requested in agent_ids:
        return requested
    print(f"Unknown agent_id={requested}; using {agent_ids[0]}")
    return agent_ids[0]


def _next_agent_id(agent_ids: list[str], current: str) -> str:
    index = agent_ids.index(current) if current in agent_ids else -1
    return agent_ids[(index + 1) % len(agent_ids)]


def _print_debug_step(snapshot, controlled_agent_id: str, observation, events: list) -> None:
    print(summarize_agent(snapshot, controlled_agent_id))
    print(summarize_observation(observation))
    if events:
        print(summarize_events(events))


if __name__ == "__main__":
    run_manual_debug()
