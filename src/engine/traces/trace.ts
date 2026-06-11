export const TRAJECTORY_SCHEMA_VERSION = '0.1.0';

export interface TeamDescriptor {
    teamId: string;
    playerIds: string[];
}

export interface PlayerDescriptor {
    id: string;
    teamId: string;
    tacticId?: string | null;
    policyId?: string | null;
    genomeId?: string | null;
}

export interface TrajectoryMetadata {
    trajectoryId: string;
    schemaVersion?: string;
    scenarioId: string;
    seed: number | null;
    createdAt?: string;
    teams: TeamDescriptor[];
    players: PlayerDescriptor[];
}

export interface EnvironmentState {
    width: number;
    height: number;
    bounds?: {
        minX: number;
        minY: number;
        maxX: number;
        maxY: number;
    };
    activeProjectileCount?: number;
    projectiles?: ProjectileStateSnapshot[];
    remainingTeams?: string[];
}

export interface ProjectileStateSnapshot {
    id: string;
    shooterId: string;
    shooterTeamId: string;
    positionX: number;
    positionY: number;
    headingX: number;
    headingY: number;
}

export interface PlayerInitialState {
    id: string;
    teamId: string;
    positionX: number;
    positionY: number;
    headingX: number;
    headingY: number;
    hp: number;
    alive: boolean;
    weaponCooldownSteps: number;
}

export interface InitialState {
    environment: EnvironmentState;
    players: PlayerInitialState[];
}

export interface PlayerStateSnapshot {
    positionX: number;
    positionY: number;
    headingX: number;
    headingY: number;
    velocityX: number;
    velocityY: number;
    hp: number;
    alive: boolean;
    weaponCooldownSteps: number;
}

export interface PlayerAction {
    moveX?: number;
    moveY?: number;
    aimX?: number;
    aimY?: number;
    fire?: number;
    // Compatibility for older evaluation fixtures; TraceRecorder emits canonical fields.
    dx?: number;
    dy?: number;
    dh?: number;
    ds?: boolean;
}

export interface DecisionReason {
    source: string;
    label: string;
    evidence: Record<string, unknown>;
}

export interface PlayerMeasurements {
    nearestAllyDistance: number;
    nearestEnemyDistance: number;
    canFire: boolean;
    didFire: boolean;
    targetId?: string | null;
    targetDistance?: number | null;
    aimTargetAlignment?: number | null;
    damageDealt: number;
    damageTaken: number;
    // Compatibility fields for existing metrics; newly recorded traces keep these in state.
    positionX?: number;
    positionY?: number;
    hp?: number;
}

export interface PlayerStepRecord {
    step?: number;
    actorId: string;
    actorTeamId: string;
    // Runtime trajectory validation requires state.
    state?: PlayerStateSnapshot;
    action: PlayerAction;
    reason: DecisionReason;
    measurements: PlayerMeasurements;
}

export interface TrajectoryEvent {
    type: string;
    actorId?: string | null;
    actorTeamId?: string | null;
    targetId?: string | null;
    targetTeamId?: string | null;
    damage?: number;
    [key: string]: unknown;
}

export interface TrajectoryStep {
    step: number;
    timeMs: number;
    environment?: EnvironmentState;
    players: PlayerStepRecord[];
    events?: TrajectoryEvent[];
}

export interface TrajectoryResult {
    winnerTeamId: string | null;
    endStep: number;
    endReason: string;
}

export interface Trajectory {
    trajectoryId: string;
    schemaVersion: string;
    scenarioId: string;
    seed: number | null;
    createdAt: string;
    teams: TeamDescriptor[];
    players: PlayerDescriptor[];
    initialState?: InitialState | null;
    steps: TrajectoryStep[];
    result: TrajectoryResult | null;
}

export interface ReplayPlayerState {
    id: string;
    teamId: string;
    positionX: number;
    positionY: number;
    headingX: number;
    headingY: number;
    hp: number;
    alive: boolean;
    weaponCooldownSteps: number;
    lastAction: PlayerAction;
    reason: DecisionReason;
    measurements?: PlayerMeasurements;
}

export interface ReplayState {
    step: number;
    timeMs: number;
    environment: EnvironmentState;
    players: ReplayPlayerState[];
}

export function createDefaultDecisionReason(): DecisionReason {
    return { source: 'unknown', label: 'not_recorded', evidence: {} };
}

export function createEmptyTrajectory(metadata: TrajectoryMetadata): Trajectory {
    return {
        trajectoryId: metadata.trajectoryId,
        schemaVersion: metadata.schemaVersion || TRAJECTORY_SCHEMA_VERSION,
        scenarioId: metadata.scenarioId,
        seed: metadata.seed,
        createdAt: metadata.createdAt || new Date().toISOString(),
        teams: metadata.teams,
        players: metadata.players,
        initialState: null,
        steps: [],
        result: null
    };
}

export type TrajectoryTeam = TeamDescriptor;
export type TrajectoryPlayer = PlayerDescriptor;
