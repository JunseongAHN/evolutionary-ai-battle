import {
    PlayerMeasurements,
    PlayerStepRecord,
    Trajectory,
    TrajectoryPlayer,
    TrajectoryStep
} from '../../engine/traces/trace';

export interface PlayerMetric {
    playerId: string;
    teamId: string;
    damageDealt: number;
    damageTaken: number;
    kills: number;
    survivalSteps: number;
}

export interface PlayerMetricMap {
    [playerId: string]: PlayerMetric;
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null;
}

function toFiniteNumber(value: unknown, fallback = 0): number {
    return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function getStringValue(value: unknown): string | null {
    return typeof value === 'string' && value.length > 0 ? value : null;
}

function getPlayerMetadata(trajectory: Trajectory | null | undefined, playerId: string) {
    if (!trajectory || !Array.isArray(trajectory.players)) {
        return null;
    }

    return trajectory.players.find((player) => player && typeof player.id === 'string' && player.id === playerId) || null;
}

export function getEmptyPlayerMetric(player: { playerId: string; teamId?: string | null } | null | undefined): PlayerMetric {
    return {
        playerId: player && typeof player.playerId === 'string' ? player.playerId : '',
        teamId: player && typeof player.teamId === 'string' && player.teamId.length > 0 ? player.teamId : 'unknown',
        damageDealt: 0,
        damageTaken: 0,
        kills: 0,
        survivalSteps: 0
    };
}

export function ensurePlayerMetric(metrics: PlayerMetricMap, playerId: string, teamId?: string | null): PlayerMetric {
    if (!playerId) {
        throw new Error('Cannot ensure player metric without a playerId');
    }

    if (!metrics[playerId]) {
        metrics[playerId] = getEmptyPlayerMetric({
            playerId,
            teamId: teamId || null
        });
    }

    if (teamId) {
        metrics[playerId].teamId = teamId;
    }

    return metrics[playerId];
}

function collectExplicitKillEvents(trajectory: Trajectory): Array<Record<string, unknown>> {
    const eventCandidates: Array<Record<string, unknown>> = [];

    const maybePush = (value: unknown) => {
        if (Array.isArray(value)) {
            value.forEach((entry) => {
                if (isRecord(entry)) {
                    eventCandidates.push(entry);
                }
            });
        }
    };

    if (isRecord(trajectory)) {
        maybePush((trajectory as Record<string, unknown>).events);
        maybePush((trajectory as Record<string, unknown>).killEvents);
        maybePush((trajectory as Record<string, unknown>).deathEvents);
    }

    trajectory.steps.forEach((step) => {
        if (!isRecord(step)) {
            return;
        }
        maybePush(step.events);
        maybePush(step.killEvents);
        maybePush(step.deathEvents);
        step.players.forEach((player) => {
            if (!isRecord(player)) {
                return;
            }
            maybePush(player.events);
            maybePush(player.killEvents);
            maybePush(player.deathEvents);
        });
    });

    return eventCandidates;
}

function inferKillsFromExplicitEvents(trajectory: Trajectory, metrics: PlayerMetricMap) {
    const killEvents = collectExplicitKillEvents(trajectory);

    killEvents.forEach((event) => {
        const eventType = getStringValue(event.type) || getStringValue(event.kind) || '';
        const isKillEvent = eventType.includes('kill') || eventType.includes('death') || eventType.includes('eliminate');
        if (!isKillEvent) {
            return;
        }

        const actorId = getStringValue(event.actorId)
            || getStringValue(event.killerId)
            || getStringValue(event.killerActorId)
            || getStringValue(event.sourceActorId);
        const victimId = getStringValue(event.targetId)
            || getStringValue(event.victimId)
            || getStringValue(event.targetActorId)
            || getStringValue(event.victimActorId);

        if (!actorId || !victimId) {
            return;
        }

        const actorMetric = ensurePlayerMetric(metrics, actorId, getStringValue(event.actorTeamId) || getStringValue(event.killerTeamId));
        if (actorMetric.playerId === victimId) {
            return;
        }
        actorMetric.kills += 1;
    });
}

function inferKillsFromMeasurements(trajectory: Trajectory, metrics: PlayerMetricMap) {
    trajectory.steps.forEach((step) => {
        step.players.forEach((player) => {
            if (!isRecord(player)) {
                return;
            }
            const actorId = getStringValue(player.actorId);
            if (!actorId) {
                return;
            }

            const measurements = isRecord(player.measurements) ? player.measurements : null;
            const explicitKillCount = toFiniteNumber(measurements && (measurements.kills as unknown));
            if (explicitKillCount > 0) {
                const actorMetric = ensurePlayerMetric(metrics, actorId, getStringValue(player.actorTeamId));
                actorMetric.kills += explicitKillCount;
            }
        });
    });
}

export function computePlayerMetrics(trajectory: Trajectory | null | undefined): PlayerMetricMap {
    const metrics: PlayerMetricMap = {};

    if (!trajectory || typeof trajectory !== 'object') {
        return metrics;
    }

    if (!Array.isArray(trajectory.steps)) {
        return metrics;
    }

    if (Array.isArray(trajectory.players)) {
        trajectory.players.forEach((player) => {
            if (!isRecord(player)) {
                return;
            }
            const playerId = getStringValue(player.id);
            if (!playerId) {
                return;
            }
            ensurePlayerMetric(metrics, playerId, getStringValue(player.teamId));
        });
    }

    trajectory.steps.forEach((step) => {
        if (!isRecord(step) || !Array.isArray(step.players)) {
            return;
        }

        step.players.forEach((player) => {
            if (!isRecord(player)) {
                return;
            }
            const playerId = getStringValue(player.actorId);
            if (!playerId) {
                return;
            }

            const metric = ensurePlayerMetric(metrics, playerId, getStringValue(player.actorTeamId));
            const measurements = isRecord(player.measurements) ? player.measurements : null;

            metric.damageDealt += toFiniteNumber(measurements && (measurements.damageDealt as unknown));
            metric.damageTaken += toFiniteNumber(measurements && (measurements.damageTaken as unknown));

            const hp = toFiniteNumber(measurements && (measurements.hp as unknown));
            if (hp > 0) {
                metric.survivalSteps += 1;
            }
        });
    });

    inferKillsFromExplicitEvents(trajectory, metrics);
    if (!Object.values(metrics).some((metric) => metric.kills > 0)) {
        inferKillsFromMeasurements(trajectory, metrics);
    }

    return metrics;
}

export function aggregateTeamPlayerMetrics(playerMetrics: PlayerMetricMap) {
    return Object.values(playerMetrics).reduce((aggregates, metric) => {
        const teamId = metric.teamId || 'unknown';
        if (!aggregates[teamId]) {
            aggregates[teamId] = {
                teamId,
                playerCount: 0,
                damageDealt: 0,
                damageTaken: 0,
                kills: 0,
                survivalSteps: 0
            };
        }

        aggregates[teamId].playerCount += 1;
        aggregates[teamId].damageDealt += metric.damageDealt;
        aggregates[teamId].damageTaken += metric.damageTaken;
        aggregates[teamId].kills += metric.kills;
        aggregates[teamId].survivalSteps += metric.survivalSteps;

        return aggregates;
    }, {} as Record<string, { teamId: string; playerCount: number; damageDealt: number; damageTaken: number; kills: number; survivalSteps: number }>);
}

export function smokeTestPlayerMetrics() {
    const syntheticTrajectory: Trajectory = {
        trajectoryId: 'smoke-trajectory',
        schemaVersion: '0.1.0',
        scenarioId: 'smoke-scenario',
        seed: 1,
        createdAt: '2026-01-01T00:00:00.000Z',
        teams: [
            { teamId: 'team-a', playerIds: ['player-a-1', 'player-a-2'] },
            { teamId: 'team-b', playerIds: ['player-b-1', 'player-b-2'] }
        ],
        players: [
            { id: 'player-a-1', teamId: 'team-a' },
            { id: 'player-a-2', teamId: 'team-a' },
            { id: 'player-b-1', teamId: 'team-b' },
            { id: 'player-b-2', teamId: 'team-b' }
        ],
        steps: [
            {
                step: 0,
                timeMs: 0,
                players: [
                    {
                        step: 0,
                        actorId: 'player-a-1',
                        actorTeamId: 'team-a',
                        action: { dx: 0, dy: 0, dh: 0, ds: false },
                        reason: { source: 'unknown', label: 'not_recorded', evidence: {} },
                        measurements: { positionX: 0, positionY: 0, hp: 100, nearestAllyDistance: 1, nearestEnemyDistance: 2, damageDealt: 10, damageTaken: 0 }
                    },
                    {
                        step: 0,
                        actorId: 'player-b-1',
                        actorTeamId: 'team-b',
                        action: { dx: 0, dy: 0, dh: 0, ds: false },
                        reason: { source: 'unknown', label: 'not_recorded', evidence: {} },
                        measurements: { positionX: 0, positionY: 0, hp: 80, nearestAllyDistance: 1, nearestEnemyDistance: 2, damageDealt: 0, damageTaken: 5 }
                    }
                ]
            },
            {
                step: 1,
                timeMs: 50,
                players: [
                    {
                        step: 1,
                        actorId: 'player-a-1',
                        actorTeamId: 'team-a',
                        action: { dx: 1, dy: 0, dh: 0, ds: false },
                        reason: { source: 'unknown', label: 'not_recorded', evidence: {} },
                        measurements: { positionX: 1, positionY: 0, hp: 70, nearestAllyDistance: 1, nearestEnemyDistance: 2, damageDealt: 8, damageTaken: 0 }
                    },
                    {
                        step: 1,
                        actorId: 'player-b-1',
                        actorTeamId: 'team-b',
                        action: { dx: 0, dy: 1, dh: 0, ds: false },
                        reason: { source: 'unknown', label: 'not_recorded', evidence: {} },
                        measurements: { positionX: 0, positionY: 1, hp: 0, nearestAllyDistance: 1, nearestEnemyDistance: 2, damageDealt: 0, damageTaken: 8 }
                    }
                ]
            }
        ],
        result: {
            winnerTeamId: 'team-a',
            endStep: 2,
            endReason: 'team_eliminated'
        }
    };

    const metrics = computePlayerMetrics(syntheticTrajectory);

    if (metrics['player-a-1'].damageDealt !== 18) {
        throw new Error('Expected damageDealt to be 18 for player-a-1');
    }
    if (metrics['player-b-1'].damageTaken !== 13) {
        throw new Error('Expected damageTaken to be 13 for player-b-1');
    }
    if (metrics['player-a-1'].survivalSteps !== 2) {
        throw new Error('Expected survivalSteps to count alive hp > 0 steps');
    }
    if (metrics['player-b-1'].survivalSteps !== 1) {
        throw new Error('Expected survivalSteps to stop counting at hp 0');
    }

    return metrics;
}
