from __future__ import annotations

from core.schema import BattleAction, BattleSnapshot


TEAM_COLORS = {
    "team-a": (65, 140, 255),
    "team-b": (235, 85, 85),
}
BACKGROUND = (24, 27, 34)
PANEL = (34, 38, 48)
TEXT = (235, 238, 245)
MUTED = (155, 163, 178)
DEAD = (95, 99, 110)
HIGHLIGHT = (255, 216, 92)
HP_GREEN = (64, 205, 116)
HP_RED = (215, 70, 70)
ZONE_RED = (230, 45, 55, 72)


def world_to_screen(
    position: dict,
    snapshot: BattleSnapshot,
    viewport: tuple[int, int, int, int],
) -> tuple[int, int]:
    left, top, width, height = viewport
    map_spec = snapshot["map"]
    scale_x = width / max(float(map_spec["width"]), 1.0)
    scale_y = height / max(float(map_spec["height"]), 1.0)
    return (
        int(left + float(position["x"]) * scale_x),
        int(top + float(position["y"]) * scale_y),
    )


def screen_to_world(
    point: tuple[int, int],
    snapshot: BattleSnapshot,
    viewport: tuple[int, int, int, int],
) -> dict[str, float]:
    left, top, width, height = viewport
    map_spec = snapshot["map"]
    x = (point[0] - left) / max(width, 1) * float(map_spec["width"])
    y = (point[1] - top) / max(height, 1) * float(map_spec["height"])
    return {
        "x": max(0.0, min(float(map_spec["width"]), x)),
        "y": max(0.0, min(float(map_spec["height"]), y)),
    }


def draw_debug_view(
    pygame,
    surface,
    font,
    small_font,
    snapshot: BattleSnapshot,
    controlled_agent_id: str,
    action: BattleAction,
    recent_events: list,
    terminated: bool,
    truncated: bool,
    fire_range: float = 260.0,
    debug_bullets: list | None = None,
) -> tuple[int, int, int, int]:
    surface.fill(BACKGROUND)
    screen_width, screen_height = surface.get_size()
    panel_width = 320
    margin = 18
    viewport = (margin, margin, screen_width - panel_width - (margin * 2), screen_height - (margin * 2))

    _draw_safe_zone_overlay(pygame, surface, snapshot, viewport)
    pygame.draw.rect(surface, (72, 78, 92), viewport, width=2)
    actor = snapshot["agents"][controlled_agent_id]
    actor_center = world_to_screen(actor["position"], snapshot, viewport)
    range_px = world_radius_to_screen(float(fire_range), snapshot, viewport)
    pygame.draw.circle(surface, HIGHLIGHT, actor_center, range_px, width=1)

    for bullet in debug_bullets or []:
        bullet_center = world_to_screen(bullet["position"], snapshot, viewport)
        pygame.draw.circle(surface, HIGHLIGHT, bullet_center, 4)

    for agent_id in snapshot["agent_ids"]:
        agent = snapshot["agents"][agent_id]
        center = world_to_screen(agent["position"], snapshot, viewport)
        team_color = TEAM_COLORS.get(agent["team_id"], (180, 180, 180))
        color = team_color if agent["alive"] else DEAD
        radius = 12 if agent_id == controlled_agent_id else 9
        pygame.draw.circle(surface, color, center, radius)
        if agent_id == controlled_agent_id:
            pygame.draw.circle(surface, HIGHLIGHT, center, radius + 4, width=2)

        aim = agent.get("aim", {"x": 0.0, "y": 0.0})
        end = (int(center[0] + float(aim["x"]) * 36), int(center[1] + float(aim["y"]) * 36))
        pygame.draw.line(surface, HIGHLIGHT if agent_id == controlled_agent_id else MUTED, center, end, width=2)

        hp_width = 42
        hp_ratio = max(0.0, min(1.0, float(agent["hp"]) / 100.0))
        hp_rect = pygame.Rect(center[0] - hp_width // 2, center[1] - 24, hp_width, 5)
        pygame.draw.rect(surface, HP_RED, hp_rect)
        pygame.draw.rect(surface, HP_GREEN, (hp_rect.x, hp_rect.y, int(hp_rect.w * hp_ratio), hp_rect.h))
        label = small_font.render(agent_id, True, TEXT if agent["alive"] else MUTED)
        surface.blit(label, (center[0] - label.get_width() // 2, center[1] + 14))

    panel_rect = pygame.Rect(screen_width - panel_width, 0, panel_width, screen_height)
    pygame.draw.rect(surface, PANEL, panel_rect)
    lines = _panel_lines(snapshot, controlled_agent_id, action, recent_events, terminated, truncated)
    y = 16
    for index, line in enumerate(lines):
        color = TEXT if index < 8 else MUTED
        rendered = font.render(line, True, color)
        surface.blit(rendered, (screen_width - panel_width + 14, y))
        y += 24
    return viewport


def _draw_safe_zone_overlay(
    pygame,
    surface,
    snapshot: BattleSnapshot,
    viewport: tuple[int, int, int, int],
) -> None:
    safe_zone = snapshot.get("safe_zone")
    if not safe_zone:
        return

    left, top, width, height = viewport
    overlay = pygame.Surface((width, height), pygame.SRCALPHA)
    overlay.fill(ZONE_RED)

    center = world_to_screen(safe_zone["center"], snapshot, viewport)
    local_center = (center[0] - left, center[1] - top)
    radius = world_radius_to_screen(float(safe_zone["radius"]), snapshot, viewport)
    pygame.draw.circle(overlay, (0, 0, 0, 0), local_center, radius)
    surface.blit(overlay, (left, top))


def world_radius_to_screen(
    radius: float,
    snapshot: BattleSnapshot,
    viewport: tuple[int, int, int, int],
) -> int:
    map_spec = snapshot["map"]
    scale_x = viewport[2] / max(float(map_spec["width"]), 1.0)
    scale_y = viewport[3] / max(float(map_spec["height"]), 1.0)
    return int(radius * min(scale_x, scale_y))


def _panel_lines(
    snapshot: BattleSnapshot,
    controlled_agent_id: str,
    action: BattleAction,
    recent_events: list,
    terminated: bool,
    truncated: bool,
) -> list[str]:
    agent = snapshot["agents"][controlled_agent_id]
    body = action["action"]
    lines = [
        f"step: {snapshot['step']}",
        f"agent: {controlled_agent_id}",
        f"team: {agent['team_id']}",
        f"hp: {float(agent['hp']):.1f}",
        f"alive: {agent['alive']}",
        f"pos: {float(agent['position']['x']):.1f}, {float(agent['position']['y']):.1f}",
        f"aim: {float(agent.get('aim', {}).get('x', 0.0)):.2f}, {float(agent.get('aim', {}).get('y', 0.0)):.2f}",
        f"action: mx={body['move_x']:.1f} my={body['move_y']:.1f}",
        f"aim action: ax={body['aim_x']:.2f} ay={body['aim_y']:.2f}",
        f"fire: {body['fire']:.1f}",
        f"terminated: {terminated}",
        f"truncated: {truncated}",
        "",
        "recent events",
    ]
    for event in recent_events[-8:]:
        actor = event.get("actor_id", "-")
        target = event.get("target_id", "-")
        lines.append(f"{event['event_type']} a={actor} t={target}")
    return lines
