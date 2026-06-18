from __future__ import annotations

import math
from typing import Iterable

from core.env_core import PythonBattleCoreEnv
from core.schema import AgentId, BattleAction, MultiAgentAction, SCHEMA_VERSION, Vec2
from debug.render_state import draw_debug_view, screen_to_world
from debug.state_inspector import summarize_agent, summarize_events, summarize_observation


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
        import pygame
    except ImportError:
        print("pygame is required for manual env debugging. Install it with: pip install pygame")
        return

    env = PythonBattleCoreEnv()
    observations = env.reset(seed=seed)
    controlled_agent_id = _coerce_agent_id(env.agent_ids, controlled_agent_id)
    print(f"Reset seed={seed} controlled_agent={controlled_agent_id}")

    pygame.init()
    pygame.display.set_caption("CPC Core Env Manual Debug")
    screen = pygame.display.set_mode((width, height))
    font = pygame.font.SysFont("consolas", 16)
    small_font = pygame.font.SysFont("consolas", 12)
    clock = pygame.time.Clock()
    paused = False
    single_step = False
    running = True
    last_step = None
    last_action = build_noop_action(env.episode_id, env.step_index, controlled_agent_id)
    last_snapshot = env._snapshot([])
    recent_events = []
    debug_bullets: list[dict] = []
    pending_fire_aim: Vec2 | None = None
    viewport = (18, 18, width - 356, height - 36)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_world = screen_to_world(event.pos, last_snapshot, viewport)
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
        mouse_world = screen_to_world(pygame.mouse.get_pos(), last_snapshot, viewport)
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
            if pending_fire_aim is not None:
                bullet = spawn_debug_bullet(env.agents[controlled_agent_id], env.fire_range, pending_fire_aim)
                if bullet is not None:
                    debug_bullets.append(bullet)
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
            single_step = False
            pending_fire_aim = None
            if recent_events or env.step_index % max(1, fps) == 0:
                _print_debug_step(last_snapshot, controlled_agent_id, observations[controlled_agent_id], recent_events)

        terminated = bool(last_step and last_step["terminated"])
        truncated = bool(last_step and last_step["truncated"])
        viewport = draw_debug_view(
            pygame,
            screen,
            font,
            small_font,
            last_snapshot,
            controlled_agent_id,
            last_action,
            recent_events,
            terminated,
            truncated,
            fire_range=env.fire_range,
            debug_bullets=debug_bullets,
        )
        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


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
