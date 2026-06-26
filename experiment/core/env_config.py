from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Vec2Config:
    x: float
    y: float


@dataclass(frozen=True)
class PlayerConfig:
    spawn: Vec2Config
    radius: float
    hp: float
    move_speed: float
    aim_turn_speed: float
    weapon_range: float
    fire_cooldown_steps: int


@dataclass(frozen=True)
class AllyConfig:
    spawn: Vec2Config
    radius: float
    hp: float


@dataclass(frozen=True)
class EnemyConfig:
    id: str
    spawn: Vec2Config
    radius: float
    hp: float
    move_speed: float
    behavior: str = "stationary"


@dataclass(frozen=True)
class ObstacleConfig:
    id: str
    type: str
    x: float
    y: float
    radius: float


@dataclass(frozen=True)
class ZoneConfig:
    enabled: bool = False
    center: Vec2Config | None = None
    safe_radius_start: float = 420.0
    safe_radius_end: float = 420.0


@dataclass(frozen=True)
class EnvConfig:
    seed: int
    max_steps: int
    dt: float
    map_width: float
    map_height: float
    player: PlayerConfig
    enemies: list[EnemyConfig]
    obstacles: list[ObstacleConfig] = field(default_factory=list)
    zone: ZoneConfig = field(default_factory=ZoneConfig)
    ally: AllyConfig | None = None


def default_env_config() -> EnvConfig:
    return EnvConfig(
        seed=0,
        max_steps=50,
        dt=1.0,
        map_width=1000.0,
        map_height=1000.0,
        player=PlayerConfig(
            spawn=Vec2Config(430.0, 500.0),
            radius=12.0,
            hp=100.0,
            move_speed=35.0,
            aim_turn_speed=1.0,
            weapon_range=280.0,
            fire_cooldown_steps=5,
        ),
        ally=AllyConfig(
            spawn=Vec2Config(370.0, 545.0),
            radius=12.0,
            hp=100.0,
        ),
        enemies=[
            EnemyConfig(
                id="enemy",
                spawn=Vec2Config(690.0, 500.0),
                radius=12.0,
                hp=100.0,
                move_speed=18.0,
                behavior="chase",
            )
        ],
        obstacles=[],
        zone=ZoneConfig(enabled=False, center=Vec2Config(500.0, 500.0)),
    )


def resolve_env_config_path(path: str | Path) -> Path:
    raw_path = Path(path)
    candidates = [raw_path]
    repo_root = Path(__file__).resolve().parents[2]
    experiment_root = repo_root / "experiment"
    if not raw_path.is_absolute():
        candidates.extend(
            [
                repo_root / raw_path,
                experiment_root / raw_path,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return raw_path.resolve(strict=False)


def load_env_config(path: str | Path) -> EnvConfig:
    config_path = resolve_env_config_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"env config not found: {config_path}")
    data = _load_yaml_mapping(config_path)
    return env_config_from_dict(data)


def env_config_from_dict(data: dict[str, Any]) -> EnvConfig:
    defaults = default_env_config()
    env_data = _mapping(data.get("env", {}), "env")
    map_data = _mapping(data.get("map", {}), "map")
    player_data = _mapping(data.get("player", {}), "player")
    zone_data = _mapping(data.get("zone", {}), "zone")

    player = _player_config(player_data, defaults.player)
    ally = _ally_config(data.get("ally"), defaults.ally)
    enemies = _enemy_configs(data.get("enemies"), defaults.enemies)
    obstacles = _obstacle_configs(data.get("obstacles"))
    zone = _zone_config(zone_data, defaults.zone)

    return EnvConfig(
        seed=_int(env_data, "seed", defaults.seed),
        max_steps=_int(env_data, "max_steps", defaults.max_steps),
        dt=_float(env_data, "dt", defaults.dt),
        map_width=_float(map_data, "width", defaults.map_width),
        map_height=_float(map_data, "height", defaults.map_height),
        player=player,
        enemies=enemies,
        obstacles=obstacles,
        zone=zone,
        ally=ally,
    )


def _player_config(data: dict[str, Any], defaults: PlayerConfig) -> PlayerConfig:
    return PlayerConfig(
        spawn=_vec2_config(data.get("spawn"), defaults.spawn, "player.spawn"),
        radius=_float(data, "radius", defaults.radius),
        hp=_float(data, "hp", defaults.hp),
        move_speed=_float(data, "move_speed", defaults.move_speed),
        aim_turn_speed=_float(data, "aim_turn_speed", defaults.aim_turn_speed),
        weapon_range=_float(data, "weapon_range", defaults.weapon_range),
        fire_cooldown_steps=_int(data, "fire_cooldown_steps", defaults.fire_cooldown_steps),
    )


def _ally_config(data: Any, defaults: AllyConfig | None) -> AllyConfig | None:
    if data is None:
        return defaults
    ally_data = _mapping(data, "ally")
    if defaults is None:
        defaults = AllyConfig(spawn=Vec2Config(0.0, 0.0), radius=12.0, hp=100.0)
    return AllyConfig(
        spawn=_vec2_config(ally_data.get("spawn"), defaults.spawn, "ally.spawn"),
        radius=_float(ally_data, "radius", defaults.radius),
        hp=_float(ally_data, "hp", defaults.hp),
    )


def _enemy_configs(data: Any, defaults: list[EnemyConfig]) -> list[EnemyConfig]:
    if data is None:
        return list(defaults)
    if not isinstance(data, list):
        raise ValueError("enemies must be a list")
    if not data:
        return []
    fallback = defaults[0] if defaults else EnemyConfig(
        id="enemy",
        spawn=Vec2Config(0.0, 0.0),
        radius=12.0,
        hp=100.0,
        move_speed=18.0,
    )
    enemies: list[EnemyConfig] = []
    for index, item in enumerate(data):
        item_data = _mapping(item, f"enemies[{index}]")
        enemies.append(
            EnemyConfig(
                id=str(item_data.get("id", fallback.id)),
                spawn=_vec2_config(item_data.get("spawn"), fallback.spawn, f"enemies[{index}].spawn"),
                radius=_float(item_data, "radius", fallback.radius),
                hp=_float(item_data, "hp", fallback.hp),
                move_speed=_float(item_data, "move_speed", fallback.move_speed),
                behavior=str(item_data.get("behavior", fallback.behavior)),
            )
        )
    return enemies


def _obstacle_configs(data: Any) -> list[ObstacleConfig]:
    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError("obstacles must be a list")
    obstacles: list[ObstacleConfig] = []
    for index, item in enumerate(data):
        item_data = _mapping(item, f"obstacles[{index}]")
        obstacles.append(
            ObstacleConfig(
                id=str(item_data.get("id", f"obstacle-{index}")),
                type=str(item_data.get("type", "circle")),
                x=_float(item_data, "x", 0.0),
                y=_float(item_data, "y", 0.0),
                radius=_float(item_data, "radius", 0.0),
            )
        )
    return obstacles


def _zone_config(data: dict[str, Any], defaults: ZoneConfig) -> ZoneConfig:
    center_data = data.get("center")
    center = defaults.center if center_data is None else _vec2_config(center_data, defaults.center or Vec2Config(0.0, 0.0), "zone.center")
    return ZoneConfig(
        enabled=_bool(data, "enabled", defaults.enabled),
        center=center,
        safe_radius_start=_float(data, "safe_radius_start", defaults.safe_radius_start),
        safe_radius_end=_float(data, "safe_radius_end", defaults.safe_radius_end),
    )


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        data = _parse_simple_yaml(text)
    else:
        data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"env config root must be a mapping: {path}")
    return data


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    lines = _yaml_lines(text)
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    for index, (indent, stripped) in enumerate(lines):
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if stripped.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"YAML list item has no list parent: {stripped}")
            item_text = stripped[2:].strip()
            if ":" in item_text:
                item: dict[str, Any] = {}
                parent.append(item)
                _assign_yaml_mapping_value(item, item_text, indent, index, lines, stack)
                stack.append((indent, item))
            else:
                parent.append(_parse_scalar(item_text))
            continue

        if not isinstance(parent, dict):
            raise ValueError(f"YAML mapping entry has no mapping parent: {stripped}")
        _assign_yaml_mapping_value(parent, stripped, indent, index, lines, stack)
    return root


def _assign_yaml_mapping_value(
    parent: dict[str, Any],
    text: str,
    indent: int,
    index: int,
    lines: list[tuple[int, str]],
    stack: list[tuple[int, Any]],
) -> None:
    if ":" not in text:
        raise ValueError(f"expected YAML key/value entry: {text}")
    key, value = text.split(":", 1)
    key = key.strip()
    value = value.strip()
    if value:
        parent[key] = _parse_scalar(value)
        return
    container: Any = [] if _next_is_list(index, indent, lines) else {}
    parent[key] = container
    stack.append((indent, container))


def _yaml_lines(text: str) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        lines.append((len(line) - len(line.lstrip(" ")), stripped))
    return lines


def _next_is_list(index: int, indent: int, lines: list[tuple[int, str]]) -> bool:
    for next_indent, next_text in lines[index + 1:]:
        if next_indent <= indent:
            return False
        return next_text.startswith("- ")
    return False


def _parse_scalar(value: str) -> Any:
    cleaned = value.strip().strip('"').strip("'")
    lowered = cleaned.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return int(cleaned)
    except ValueError:
        pass
    try:
        return float(cleaned)
    except ValueError:
        return cleaned


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _vec2_config(value: Any, default: Vec2Config, name: str) -> Vec2Config:
    data = _mapping(value, name)
    return Vec2Config(
        x=_float(data, "x", default.x),
        y=_float(data, "y", default.y),
    )


def _float(data: dict[str, Any], key: str, default: float) -> float:
    value = data.get(key, default)
    if value is None:
        return float(default)
    return float(value)


def _int(data: dict[str, Any], key: str, default: int) -> int:
    value = data.get(key, default)
    if value is None:
        return int(default)
    return int(value)


def _bool(data: dict[str, Any], key: str, default: bool) -> bool:
    value = data.get(key, default)
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


__all__ = [
    "AllyConfig",
    "EnemyConfig",
    "EnvConfig",
    "ObstacleConfig",
    "PlayerConfig",
    "Vec2Config",
    "ZoneConfig",
    "default_env_config",
    "env_config_from_dict",
    "load_env_config",
    "resolve_env_config_path",
]
