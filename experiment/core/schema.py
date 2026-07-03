import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, NotRequired, TypedDict

SCHEMA_VERSION = "cpc-common-v0"


Vec2Tuple = tuple[float, float]


@dataclass(frozen=True)
class CPCObservation:
    """Framework-independent policy input assembled from an environment state."""

    actor_id: str
    state: Mapping[str, Any]
    context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CPCAction:
    move: Vec2Tuple
    aim: Vec2Tuple
    fire: bool


@dataclass(frozen=True)
class TacticalFrame:
    who: Literal["self", "team"]
    task: Literal["advance_goal", "fight_enemy", "regroup_teammate", "escape_enemy", "hold_anchor"]
    target: str
    where: Vec2Tuple
    how: Literal["navigate", "poke_out", "hold"]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"who": self.who, "task": self.task, "target": self.target,
                "where": list(self.where), "how": self.how, "reason": self.reason}

    @property
    def identity(self) -> tuple[str, str, str]:
        return self.who, self.task, self.target


@dataclass(frozen=True)
class CPCDecision:
    action: CPCAction
    selected_frame: TacticalFrame
    trace: str


@dataclass(frozen=True)
class PrimitiveAction:
    move_bin: int | None
    move: Vec2Tuple | None
    aim: Vec2Tuple
    fire_requested: bool
    fire_applied: bool

    def as_dict(self) -> dict[str, Any]:
        return {"move_bin": self.move_bin, "move": list(self.move) if self.move is not None else None,
                "aim": list(self.aim), "fire_requested": self.fire_requested,
                "fire_applied": self.fire_applied}


@dataclass(frozen=True)
class StepOutcome:
    damage_dealt: float
    damage_taken: float
    bot_hp: float
    player_hp: float | None
    enemy_hp: float | None
    goal_reached: bool
    teammate_distance: float | None

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class DecisionRecord:
    selected_frame: TacticalFrame
    primitive_action: PrimitiveAction
    outcome: StepOutcome

    def as_dict(self) -> dict[str, Any]:
        return {"selected_frame": self.selected_frame.as_dict(),
                "primitive_action": self.primitive_action.as_dict(),
                "outcome": self.outcome.as_dict()}


class CPCPolicy(ABC):
    @abstractmethod
    def decide(self, observation: CPCObservation) -> CPCDecision:
        raise NotImplementedError


def render_selected_frame(frame: TacticalFrame) -> str:
    coordinates = [int(value) if float(value).is_integer() else value for value in frame.where]
    where = json.dumps(coordinates, separators=(",", ":"))
    return (f"SELECTED {frame.who}/{frame.task}:{frame.target} WHERE {where} "
            f"HOW {frame.how} REASON {frame.reason}")

BattleMode = Literal["solo", "duo"]
AgentId = str
TeamId = str


class Vec2(TypedDict):
    x: float
    y: float


class MapSpec(TypedDict):
    width: float
    height: float
    coordinate_system: Literal["world-2d"]


class ObservationSpec(TypedDict):
    mode: Literal["local_tactical"]
    vector_keys: List[str]
    max_visible_enemies: int
    max_visible_allies: int
    max_visible_obstacles: int
    max_recent_events: int
    entity_feature_keys: Dict[str, List[str]]


class ActionSpec(TypedDict):
    action_type: Literal["continuous_2d"]
    action_keys: List[str]
    bounds: Dict[str, List[float]]


class BattleConfig(TypedDict):
    schema_version: Literal["cpc-common-v0"]
    mode: BattleMode
    team_count: int
    players_per_team: Literal[1, 2]
    max_steps: int
    map: MapSpec
    observation_spec: ObservationSpec
    action_spec: ActionSpec


class SelfObservation(TypedDict):
    hp: float
    alive: bool
    position: Vec2


class EntityObservation(TypedDict):
    entity_id: str
    team_id: TeamId
    relative_position: Vec2
    distance: float
    hp: float
    alive: bool
    has_line_of_sight: NotRequired[bool]
    is_threatening_self: NotRequired[bool]
    is_threatening_teammate: NotRequired[bool]


class ObstacleObservation(TypedDict):
    obstacle_id: str
    relative_position: Vec2
    width: float
    height: float
    distance: float
    blocks_line_of_sight: bool


class EventObservation(TypedDict):
    event_type: str
    age_steps: int
    relative_position: NotRequired[Vec2]
    actor_id: NotRequired[AgentId]
    target_id: NotRequired[AgentId]
    value: NotRequired[float]


class TacticalObservation(TypedDict):
    schema_version: Literal["cpc-common-v0"]
    episode_id: str
    step: int
    agent_id: AgentId
    team_id: TeamId
    mode: BattleMode
    self: SelfObservation
    vector: List[float]
    vector_keys: List[str]
    visible_enemies: List[EntityObservation]
    visible_enemies_mask: List[bool]
    visible_allies: List[EntityObservation]
    visible_allies_mask: List[bool]
    visible_obstacles: List[ObstacleObservation]
    visible_obstacles_mask: List[bool]
    recent_events: List[EventObservation]
    recent_events_mask: List[bool]


class AgentSnapshot(TypedDict):
    agent_id: AgentId
    team_id: TeamId
    position: Vec2
    hp: float
    alive: bool
    facing: NotRequired[Vec2]
    aim: NotRequired[Vec2]


class ObstacleSnapshot(TypedDict):
    obstacle_id: str
    position: Vec2
    width: float
    height: float
    blocks_movement: bool
    blocks_line_of_sight: bool


class MapSnapshot(TypedDict):
    width: float
    height: float
    obstacles: NotRequired[List[ObstacleSnapshot]]


class SafeZoneSnapshot(TypedDict):
    center: Vec2
    radius: float
    damage_per_step: float


BattleEventType = Literal[
    "damage",
    "death",
    "fire",
    "move",
    "spawn",
    "support_response",
    "teammate_under_pressure",
    "isolation",
]


class BattleEvent(TypedDict):
    event_id: str
    step: int
    event_type: BattleEventType
    actor_id: NotRequired[AgentId]
    target_id: NotRequired[AgentId]
    team_id: NotRequired[TeamId]
    position: NotRequired[Vec2]
    value: NotRequired[float]
    metadata: NotRequired[Dict[str, Any]]


class BattleSnapshot(TypedDict):
    schema_version: Literal["cpc-common-v0"]
    episode_id: str
    step: int
    mode: BattleMode
    agent_ids: List[AgentId]
    team_ids: List[TeamId]
    agent_team_map: Dict[AgentId, TeamId]
    map: MapSnapshot
    safe_zone: NotRequired[SafeZoneSnapshot]
    agents: Dict[AgentId, AgentSnapshot]
    events: List[BattleEvent]


class BattleActionBody(TypedDict):
    move_x: float
    move_y: float
    aim_x: float
    aim_y: float
    fire: float


class ActionSource(TypedDict):
    policy_type: Literal["random", "user_controlled", "linear_model", "future_policy"]
    policy_id: NotRequired[str]


class BattleAction(TypedDict):
    schema_version: Literal["cpc-common-v0"]
    episode_id: str
    step: int
    agent_id: AgentId
    action: BattleActionBody
    intent: NotRequired[str]
    source: NotRequired[ActionSource]


class MultiAgentAction(TypedDict):
    schema_version: Literal["cpc-common-v0"]
    episode_id: str
    step: int
    actions: Dict[AgentId, BattleAction]


class PlayerMetrics(TypedDict):
    agent_id: AgentId
    team_id: TeamId
    combat: Dict[str, float]
    survival: Dict[str, Any]
    cooperation: Dict[str, Any]
    movement: NotRequired[Dict[str, Any]]


class StepInfo(TypedDict):
    snapshot: BattleSnapshot
    events: List[BattleEvent]
    metrics: NotRequired[Dict[AgentId, PlayerMetrics]]


class MultiAgentStep(TypedDict):
    schema_version: Literal["cpc-common-v0"]
    episode_id: str
    step: int
    observations: Dict[AgentId, TacticalObservation]
    actions: Dict[AgentId, BattleAction]
    rewards: NotRequired[Dict[AgentId, float]]
    terminated: bool
    truncated: bool
    info: StepInfo


class EpisodeTrajectory(TypedDict):
    schema_version: Literal["cpc-common-v0"]
    episode_id: str
    config: BattleConfig
    steps: List[MultiAgentStep]
    final_metrics: NotRequired[Dict[AgentId, PlayerMetrics]]

