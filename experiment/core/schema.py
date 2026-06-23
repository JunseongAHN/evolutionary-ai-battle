from typing import Any, Dict, List, Literal, NotRequired, TypedDict

SCHEMA_VERSION = "cpc-common-v0"

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

