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
        self._draw_danger_overlay(env_state, map_info, map_area)
        pygame.draw.rect(self.surface, (72, 78, 92), map_rect(map_info, map_area, self.padding), width=2)
        self._draw_projectiles(env_state, map_info, map_area)
        self._draw_agents(env_state, map_info, map_area, step_record)
        self._draw_panel(env_state, step_record)
        pygame.display.flip()
        self.clock.tick(self.fps)
        return True

    def close(self) -> None:
        self.pygame.quit()

    def _draw_danger_overlay(self, env_state: Mapping[str, Any], map_info: Mapping[str, Any], map_area: tuple[int, int]) -> None:
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

        bullet_events = _bullet_events(env_state)
        for event in bullet_events:
            if event.get("type") not in ("bullet_hit", "bullet_expired"):
                continue
            position = event.get("pos")
            if not position:
                continue
            center = world_to_screen(position, map_info, map_area, self.padding)
            color = colors.FIRE if event.get("type") == "bullet_hit" else colors.MUTED
            self.pygame.draw.circle(self.surface, color, center, 10, width=2)

    def _draw_panel(self, env_state: Mapping[str, Any], step_record: Mapping[str, Any] | None) -> None:
        left = self.width - self.panel_width
        self.pygame.draw.rect(self.surface, colors.PANEL, (left, 0, self.panel_width, self.height))
        lines = _panel_lines(env_state, step_record)
        y = 16
        for index, line in enumerate(lines):
            rendered = self.font.render(line, True, colors.TEXT if index < 8 else colors.MUTED)
            self.surface.blit(rendered, (left + 14, y))
            y += 23


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
    action = _decoded_action(step_record)
    weapon = env_state.get("weapon", {})
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
        f"align: {float(aim_debug.get('aim_alignment', 0.0)):.2f}",
        f"angle err: {float(aim_debug.get('angle_error_deg', 0.0)):.1f}",
        f"outside: {bool(zone_debug.get('outside_safe_zone', False))}",
        "",
        "reward components",
    ]
    for key, value in list(components.items())[:8]:
        lines.append(f"{key}: {float(value):.3f}")
    lines.extend(["", "metrics"])
    for key in ("avg_ally_distance", "isolation_rate", "damage_dealt", "damage_taken"):
        if key in metrics:
            lines.append(f"{key}: {float(metrics[key]):.3f}")
    return lines
