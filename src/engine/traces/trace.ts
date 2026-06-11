export interface PlayerAction {
    dx: number;
    dy: number;
    dh: number;
    ds: boolean;
}

export interface DecisionReason {
    source: string;
    label: string;
    evidence: Record<string, unknown>;
}

export interface PlayerMeasurements {
    positionX: number;
    positionY: number;
    hp: number;
    nearestAllyDistance: number;
    nearestEnemyDistance: number;
    damageDealt: number;
    damageTaken: number;
}

export interface PlayerStepRecord {
    step: number;
    actorId: string;
    actorTeamId: string;
    action: PlayerAction;
    reason: DecisionReason;
    measurements: PlayerMeasurements;
}

export interface TrajectoryStep {
    step: number;
    timeMs: number;
    players: PlayerStepRecord[];
}

export interface TrajectoryTeam {
    teamId: string;
    playerIds: string[];
}

export interface TrajectoryPlayer {
    id: string;
    teamId: string;
    tacticId?: string | null;
    policyId?: string | null;
}

export interface TrajectoryResult {
    winnerTeamId: string | null;
    endStep: number;
    endReason: string;
}

export interface TrajectoryMetadata {
    trajectoryId: string;
    schemaVersion: string;
    scenarioId: string;
    seed: number | null;
    createdAt: string;
    teams: TrajectoryTeam[];
    players: TrajectoryPlayer[];
}

export interface Trajectory extends TrajectoryMetadata {
    steps: TrajectoryStep[];
    result: TrajectoryResult | null;
}

export function createDefaultDecisionReason(): DecisionReason {
    return {
        source: 'unknown',
        label: 'not_recorded',
        evidence: {}
    };
}

export function createEmptyTrajectory(metadata: TrajectoryMetadata): Trajectory {
    return {
        ...metadata,
        steps: [],
        result: null
    };
}
