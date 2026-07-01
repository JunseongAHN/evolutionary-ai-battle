from __future__ import annotations

from typing import Any, Mapping

from . import colors
from .geometry import map_rect, world_radius_to_screen, world_to_screen


class PygameCPCViewer:
    def __init__(
        self,
        width: int = 900,
        height: int = 900,
        fps: int = 10,
        title: str = "CPC Model Gameplay",
    ):
        try:
            import pygame
        except ImportError as exc:
            raise ImportError("pygame is required for --render-pygame. Install with: pip install pygame") from exc

        self.pygame = pygame
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.panel_width = 280
        self.padding = 36
        pygame.init()
        pygame.display.set_caption(title)
        self.surface = pygame.display.set_mode((self.width, self.height))
        self.font = pygame.font.SysFont("consolas", 16)
        self.small_font = pygame.font.SysFont("consolas", 13)
        self.clock = pygame.time.Clock()

    def render_step(self, env_state: dict, step_record: dict | None = None, *, handle_events: bool = True) -> bool:
        pygame = self.pygame
        if handle_events:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False

        self.surface.fill(colors.BACKGROUND)
        map_area = (self.width - self.panel_width, self.height)
        map_info = _map_info(env_state)
        self._draw_arena(env_state, map_info, map_area)
        pygame.draw.rect(self.surface, (72, 78, 92), map_rect(map_info, map_area, self.padding), width=2)
        self._draw_obstacles(env_state, map_info, map_area)
        self._draw_goal(env_state, map_info, map_area)
        self._draw_projectiles(env_state, map_info, map_area)
        self._draw_agents(env_state, map_info, map_area, step_record)
        self._draw_panel(env_state, step_record)
        pygame.display.flip()
        self.clock.tick(self.fps)
        return True

    def close(self) -> None:
        self.pygame.quit()

    def _draw_arena(self, env_state: Mapping[str, Any], map_info: Mapping[str, Any], map_area: tuple[int, int]) -> None:
        if env_state.get("stage") == "local_combat" or map_info.get("use_zone_reward") is False:
            return
        safe_radius = map_info.get("safe_radius")
        if safe_radius is None:
            return

        pygame = self.pygame
        overlay = pygame.Surface(map_area, pygame.SRCALPHA)
        zone_debug = env_state.get("zone_debug", {})
        overlay.fill((230, 45, 55, 82) if zone_debug.get("outside_safe_zone") else colors.DANGER_ZONE)
        center = map_info.get("center", {"x": 500.0, "y": 500.0})
        safe_center = world_to_screen(center, map_info, map_area, self.padding)
        safe_radius_px = world_radius_to_screen(float(safe_radius), map_info, map_area, self.padding)
        pygame.draw.circle(overlay, (0, 0, 0, 0), safe_center, safe_radius_px)
        self.surface.blit(overlay, (0, 0))
        pygame.draw.circle(
            self.surface,
            colors.WARNING if zone_debug.get("outside_safe_zone") else colors.SAFE_ZONE,
            safe_center,
            safe_radius_px,
            width=2 if zone_debug.get("outside_safe_zone") else 1,
        )

    def _draw_obstacles(
        self,
        env_state: Mapping[str, Any],
        map_info: Mapping[str, Any],
        map_area: tuple[int, int],
    ) -> None:
        obstacles = env_state.get("obstacles") or map_info.get("obstacles") or []
        for obstacle in obstacles:
            if obstacle.get("type", "circle") != "circle":
                continue
            position = {"x": obstacle.get("x", 0.0), "y": obstacle.get("y", 0.0)}
            center = world_to_screen(position, map_info, map_area, self.padding)
            radius = max(2, world_radius_to_screen(float(obstacle.get("radius", 0.0)), map_info, map_area, self.padding))
            self.pygame.draw.circle(self.surface, colors.OBSTACLE, center, radius)
            self.pygame.draw.circle(self.surface, colors.OBSTACLE_EDGE, center, radius, width=2)

    def _draw_agents(
        self,
        env_state: Mapping[str, Any],
        map_info: Mapping[str, Any],
        map_area: tuple[int, int],
        step_record: Mapping[str, Any] | None,
    ) -> None:
        agents = _agents(env_state)
        action = _decoded_action(step_record)
        for name, agent in agents.items():
            position = agent.get("position")
            if not position:
                continue

            center = world_to_screen(position, map_info, map_area, self.padding)
            alive = bool(agent.get("alive", float(agent.get("hp", 0.0)) > 0.0))
            color = _agent_color(name, agent) if alive else colors.DEAD
            radius = 13 if agent.get("role") == "self" or name == "self" else 10
            self.pygame.draw.circle(self.surface, color, center, radius)
            self._draw_hp_bar(center, float(agent.get("hp", 0.0)), y_offset=-25)
            label = self.small_font.render(name, True, colors.TEXT if alive else colors.MUTED)
            self.surface.blit(label, (center[0] - label.get_width() // 2, center[1] + 14))

            if agent.get("role") == "self" or name == "self":
                aim_debug = _aim_debug(env_state, step_record)
                target_dir = aim_debug.get("target_dir", {})
                if target_dir:
                    target_end = (
                        int(center[0] + float(target_dir.get("x", 0.0)) * 58),
                        int(center[1] + float(target_dir.get("y", 0.0)) * 58),
                    )
                    self.pygame.draw.line(self.surface, colors.TARGET, center, target_end, width=2)
                aim = _aim_vector(agent, action)
                end = (int(center[0] + aim["x"] * 42), int(center[1] + aim["y"] * 42))
                self.pygame.draw.line(self.surface, colors.FIRE, center, end, width=2)
                move = _move_vector(action)
                if move["x"] or move["y"]:
                    move_end = (int(center[0] + move["x"] * 34), int(center[1] + move["y"] * 34))
                    self.pygame.draw.line(self.surface, colors.TEXT, center, move_end, width=2)
                if float(action.get("fire", 0.0)) >= 1.0:
                    self.pygame.draw.circle(self.surface, colors.FIRE, center, radius + 7, width=2)
                status = _aim_status(step_record, aim_debug)
                if status:
                    label = self.small_font.render(status, True, colors.FIRE if status == "EXACT AIM" else colors.WARNING)
                    self.surface.blit(label, (center[0] - label.get_width() // 2, center[1] - 45))

    def _draw_goal(
        self,
        env_state: Mapping[str, Any],
        map_info: Mapping[str, Any],
        map_area: tuple[int, int],
    ) -> None:
        goal = env_state.get("goal", {})
        position = goal.get("position") if isinstance(goal, Mapping) else None
        if not goal or not goal.get("enabled") or position is None:
            return
        if isinstance(position, (list, tuple)):
            position = {"x": position[0], "y": position[1]}
        center = world_to_screen(position, map_info, map_area, self.padding)
        radius = max(
            4,
            world_radius_to_screen(float(goal.get("radius", 0.0)), map_info, map_area, self.padding),
        )
        overlay = self.pygame.Surface((radius * 2 + 4, radius * 2 + 4), self.pygame.SRCALPHA)
        local_center = (radius + 2, radius + 2)
        self.pygame.draw.circle(overlay, colors.GOAL_FILL, local_center, radius)
        self.surface.blit(overlay, (center[0] - radius - 2, center[1] - radius - 2))
        self.pygame.draw.circle(self.surface, colors.GOAL, center, radius, width=2)
        self.pygame.draw.line(self.surface, colors.GOAL, (center[0] - 7, center[1]), (center[0] + 7, center[1]), width=2)
        self.pygame.draw.line(self.surface, colors.GOAL, (center[0], center[1] - 7), (center[0], center[1] + 7), width=2)

    def _draw_hp_bar(self, center: tuple[int, int], hp: float, y_offset: int) -> None:
        hp_width = 44
        ratio = max(0.0, min(1.0, hp / 100.0))
        rect = self.pygame.Rect(center[0] - hp_width // 2, center[1] + y_offset, hp_width, 5)
        self.pygame.draw.rect(self.surface, colors.HP_RED, rect)
        self.pygame.draw.rect(self.surface, colors.HP_GREEN, (rect.x, rect.y, int(rect.w * ratio), rect.h))

    def _draw_projectiles(
        self,
        env_state: Mapping[str, Any],
        map_info: Mapping[str, Any],
        map_area: tuple[int, int],
    ) -> None:
        for projectile in env_state.get("bullets", env_state.get("projectiles", [])):
            position = projectile.get("pos") or projectile.get("position")
            if not position:
                continue
            previous = projectile.get("previous_pos") or projectile.get("spawn_pos") or position
            previous_center = world_to_screen(previous, map_info, map_area, self.padding)
            center = world_to_screen(position, map_info, map_area, self.padding)
            radius = max(4, world_radius_to_screen(float(projectile.get("radius", 8.0)), map_info, map_area, self.padding))
            self.pygame.draw.line(self.surface, colors.FIRE, previous_center, center, width=2)
            self.pygame.draw.circle(self.surface, colors.BULLET, center, radius)
            self.pygame.draw.circle(self.surface, colors.FIRE, center, radius + 2, width=1)
            direction = projectile.get("direction", {})
            if direction:
                direction_end = (
                    int(center[0] + float(direction.get("x", 0.0)) * 22),
                    int(center[1] + float(direction.get("y", 0.0)) * 22),
                )
                self.pygame.draw.line(self.surface, colors.BULLET, center, direction_end, width=2)

    def _draw_panel(self, env_state: Mapping[str, Any], step_record: Mapping[str, Any] | None) -> None:
        left = self.width - self.panel_width
        self.pygame.draw.rect(self.surface, colors.PANEL, (left, 0, self.panel_width, self.height))
        lines = _panel_lines(env_state, step_record)
        y = 16
        for index, line in enumerate(lines):
            rendered = self.font.render(line, True, colors.TEXT if index < 8 else colors.MUTED)
            self.surface.blit(rendered, (left + 14, y))
            y += 23
        if env_state.get("manual_step", {}).get("save_button"):
            rect = self.pygame.Rect(left + 14, self.height - 54, self.panel_width - 28, 36)
            mouse_pos = self.pygame.mouse.get_pos()
            fill = (60, 70, 88) if rect.collidepoint(mouse_pos) else (48, 55, 70)
            self.pygame.draw.rect(self.surface, fill, rect, border_radius=6)
            self.pygame.draw.rect(self.surface, colors.MUTED, rect, width=1, border_radius=6)
            label = self.font.render("Save Snapshot (G)", True, colors.TEXT)
            self.surface.blit(label, (rect.centerx - label.get_width() // 2, rect.centery - label.get_height() // 2))


def _map_info(env_state: Mapping[str, Any]) -> dict[str, Any]:
    map_info = dict(env_state.get("map", {}))
    state = env_state.get("state", {})
    safe_zone = env_state.get("safe_zone", {})
    if "width" not in map_info:
        map_info["width"] = env_state.get("width", 1000.0)
    if "height" not in map_info:
        map_info["height"] = env_state.get("height", 1000.0)
    map_info.setdefault("center", safe_zone.get("center", {"x": 500.0, "y": 500.0}))
    map_info.setdefault("safe_radius", safe_zone.get("radius", state.get("safe_radius")))
    map_info.setdefault("use_zone_reward", env_state.get("map", {}).get("use_zone_reward"))
    map_info.setdefault("fire_range", env_state.get("combat", {}).get("fire_range", 260.0))
    return map_info


def _agents(env_state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if isinstance(env_state.get("agents"), dict):
        return dict(env_state["agents"])
    state = env_state.get("state", {})
    return {
        "self": {"position": state.get("self_pos"), "hp": state.get("self_hp", 0.0), "alive": state.get("self_hp", 0.0) > 0.0},
        "ally": {"position": state.get("ally_pos"), "hp": state.get("ally_hp", 0.0), "alive": state.get("ally_hp", 0.0) > 0.0},
        "enemy": {"position": state.get("enemy_pos"), "hp": state.get("enemy_hp", 0.0), "alive": state.get("enemy_hp", 0.0) > 0.0},
    }


def _decoded_action(step_record: Mapping[str, Any] | None) -> dict[str, float]:
    agents = (step_record or {}).get("agents", {})
    agent = agents.get("agent") or agents.get("agent_a") or {}
    return dict(agent.get("decoded_action", {}))


def _aim_debug(env_state: Mapping[str, Any], step_record: Mapping[str, Any] | None) -> dict[str, Any]:
    env_record = (step_record or {}).get("env", {})
    info = env_record.get("info", {})
    return dict(info.get("aim_debug", env_state.get("aim_debug", {})))


def _aim_vector(agent: Mapping[str, Any], action: Mapping[str, Any]) -> dict[str, float]:
    if "aim_x" in action:
        return {"x": float(action.get("aim_x", 0.0)), "y": float(action.get("aim_y", 0.0))}
    return dict(agent.get("aim", {"x": 1.0, "y": 0.0}))


def _move_vector(action: Mapping[str, Any]) -> dict[str, float]:
    return {"x": float(action.get("move_x", 0.0)), "y": float(action.get("move_y", 0.0))}


def _bullet_events(env_state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    events = env_state.get("bullet_events")
    if isinstance(events, list):
        return events
    return []


def _agent_color(name: str, agent: Mapping[str, Any]) -> tuple[int, int, int]:
    role = agent.get("role")
    if role == "self":
        return colors.SELF
    if role == "ally":
        return colors.ALLY
    if role == "enemy":
        return colors.ENEMY
    team_id = str(agent.get("team_id", ""))
    if team_id.endswith("a") or team_id == "team-a":
        return colors.SELF
    if team_id.endswith("b") or team_id == "team-b":
        return colors.ENEMY
    if name == "self":
        return colors.SELF
    if name == "ally":
        return colors.ALLY
    return colors.ENEMY


def _panel_lines(env_state: Mapping[str, Any], step_record: Mapping[str, Any] | None) -> list[str]:
    env_record = (step_record or {}).get("env", {})
    info = env_record.get("info", {})
    rewards = env_record.get("rewards", {})
    fire_info = info.get("fire", {})
    components = info.get("reward_components", env_state.get("reward_components", {}))
    metrics = info.get("metrics", env_state.get("metrics", {}))
    aim_debug = info.get("aim_debug", env_state.get("aim_debug", {}))
    zone_debug = info.get("zone_debug", env_state.get("zone_debug", {}))
    range_debug = info.get("range_debug", env_state.get("range_debug", {}))
    action = _decoded_action(step_record)
    weapon = env_state.get("weapon", {})
    goal = env_state.get("goal", {})
    events = info.get("events", env_state.get("events", []))
    goal_position = goal.get("position") if isinstance(goal, Mapping) else None
    lines = [
        f"step: {env_state.get('step', env_state.get('step_count', '-'))}",
        f"reward: {float(rewards.get('agent', 0.0)):.3f}",
        f"done: {env_record.get('done', False)}",
        f"move: {float(action.get('move_x', 0.0)):.1f}, {float(action.get('move_y', 0.0)):.1f}",
        f"aim: {float(action.get('aim_x', 0.0)):.2f}, {float(action.get('aim_y', 0.0)):.2f}",
        f"fire: {float(action.get('fire', 0.0)):.1f}",
        f"requested: {bool(fire_info.get('fire_requested', float(action.get('fire', 0.0)) >= 1.0))}",
        f"shot fired: {bool(fire_info.get('shot_fired', False))}",
        f"cooldown: {int(fire_info.get('cooldown_remaining_steps_after', weapon.get('cooldown_remaining_steps', 0)))}",
        f"aim bin: {aim_debug.get('aim_bin', '-')}",
        f"ideal bin: {aim_debug.get('ideal_aim_bin', '-')}",
        f"bin error: {aim_debug.get('aim_bin_error', '-')}",
        f"align: {float(aim_debug.get('aim_alignment', 0.0)):.2f}",
        f"angle err: {float(aim_debug.get('angle_error_deg', 0.0)):.1f}",
        f"range: {float(range_debug.get('distance_to_enemy', 0.0)):.1f}",
        f"range ok: {bool(range_debug.get('in_good_range', False))}",
        f"trade: {float(metrics.get('damage_trade_ratio', 0.0)):.3f}",
        f"goal: {_format_debug_position(goal_position)}",
        f"goal dist: {_format_optional_number(goal.get('distance') if isinstance(goal, Mapping) else None)}",
        f"goal count: {int(goal.get('reached_count', 0)) if isinstance(goal, Mapping) else 0}",
        f"events: {','.join(str(event.get('type')) for event in events[-3:] if isinstance(event, Mapping)) or '-'}",
    ]
    manual_step = env_state.get("manual_step", {})
    if manual_step:
        lines.append(f"mode: {manual_step.get('mode', 'manual')}")
        lines.append(f"action: {manual_step.get('current_action', '-')}")
        for control_line in manual_step.get("controls", []):
            lines.append(str(control_line))
    tactical_debug = env_state.get("tactical_debug", {})
    if tactical_debug:
        lines.extend(
            [
                "",
                "tactical debug",
                f"tactical_mode: {tactical_debug.get('tactical_mode', '-')}",
                f"target_cell: {tactical_debug.get('target_cell', '-')}",
                f"next_cell: {tactical_debug.get('next_cell', '-')}",
                f"move_bin: {tactical_debug.get('move_bin', '-')}",
                f"aim_dir: ({float(tactical_debug.get('aim_dir_x', 0.0)):.2f}, "
                f"{float(tactical_debug.get('aim_dir_y', 0.0)):.2f})",
                f"fire: {tactical_debug.get('fire', '-')}",
            ]
        )
    hierarchical_debug = env_state.get("hierarchical_debug", {})
    if hierarchical_debug:
        lines.extend(
            [
                "",
                "hierarchical debug",
                f"intent: {hierarchical_debug.get('intent', '-')}",
                f"global: {hierarchical_debug.get('global_plan_reason', '-')}",
                f"tactical: {hierarchical_debug.get('tactical_mode', '-')}",
                f"profile: {hierarchical_debug.get('combat_profile', '-')}",
                f"anchor: {hierarchical_debug.get('anchor', '-')}",
                f"target: {hierarchical_debug.get('target_cell', '-')}",
                f"next: {hierarchical_debug.get('next_cell', '-')}",
                f"fire reason: {hierarchical_debug.get('fire_reason', '-')}",
                f"mode age/lock: {hierarchical_debug.get('mode_age', 0)}/{hierarchical_debug.get('mode_locked', False)}",
                f"anchor age/reuse: {hierarchical_debug.get('anchor_age', 0)}/{hierarchical_debug.get('anchor_reused', False)}",
                f"plan fallback: {hierarchical_debug.get('fallback_previous_plan', False)}",
                f"range: {hierarchical_debug.get('combat_range_state', '-')} {hierarchical_debug.get('dist_ratio', '-')}",
                f"strafe: {hierarchical_debug.get('strafe_direction', '-')}",
                f"perpendicular: {hierarchical_debug.get('perpendicular_strafe', False)}",
                f"bullet dodge: {hierarchical_debug.get('bullet_dodge_active', False)} {hierarchical_debug.get('dodge_reason', '-')}",
                f"dodge move: {hierarchical_debug.get('selected_dodge_move', '-')}",
                f"dodge blocked: {hierarchical_debug.get('dodge_blocked_reasons', {})}",
                f"range lock: {hierarchical_debug.get('range_hysteresis_locked', False)}",
                f"combat exit: {hierarchical_debug.get('combat_exit_blocked_reason', '-')}",
            ]
        )
    lines.extend(["", "reward components"])
    for key, value in list(components.items())[:8]:
        lines.append(f"{key}: {float(value):.3f}")
    lines.extend(["", "metrics"])
    for key in ("damage_dealt_ratio", "damage_taken_ratio", "hit_ratio", "bullet_hit_per_shot"):
        if key in metrics:
            lines.append(f"{key}: {float(metrics[key]):.3f}")
    return lines


def _format_debug_position(position: Any) -> str:
    if isinstance(position, Mapping):
        return f"({float(position.get('x', 0.0)):.1f},{float(position.get('y', 0.0)):.1f})"
    if isinstance(position, (list, tuple)) and len(position) >= 2:
        return f"({float(position[0]):.1f},{float(position[1]):.1f})"
    return "-"


def _format_optional_number(value: Any) -> str:
    return "-" if value is None else f"{float(value):.1f}"


def _aim_status(
    step_record: Mapping[str, Any] | None,
    aim_debug: Mapping[str, Any],
) -> str:
    env_record = (step_record or {}).get("env", {})
    info = env_record.get("info", {})
    fire_info = info.get("fire", {})
    shot_fired = bool(fire_info.get("shot_fired", info.get("shot_fired", False)))
    aim_bin_error = int(aim_debug.get("aim_bin_error", 0) or 0)
    if shot_fired and aim_bin_error >= 2:
        return "OFF TARGET"
    if aim_bin_error == 0:
        return "EXACT AIM"
    return ""
