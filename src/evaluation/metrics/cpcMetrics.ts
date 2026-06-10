import {
    PlayerMeasurements,
    PlayerStepRecord,
    Trajectory,
    TrajectoryStep
} from '../../engine/traces/trace';

export interface CpcMetric {
    playerId: string;
    teamId: string;
    avgAllyDistance: number;
    isolationRate: number;
    teammateUnderPressureEvents: number;
    teammateUnderPressureResponses: number;
    teammateUnderPressureResponseRate: number;
    friendlyFireDamage: number;
    harmEvents: number;
}

export interface CpcMetricMap {
    [playerId: string]: CpcMetric;
}

export interface CpcMetricOptions {
    isolationDistanceThreshold?: number;
    pressureHpThreshold?: number;
    pressureEnemyDistanceThreshold?: number;
    responseWindowSteps?: number;
    supportDistanceThreshold?: number;
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null;
}

function toFiniteNumber(value: unknown): number | null {
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function getStringValue(value: unknown): string | null {
    return typeof value === 'string' && value.length > 0 ? value : null;
}

function getMeasurements(record: PlayerStepRecord | null | undefined): PlayerMeasurements | null {
    if (!record || !isRecord(record.measurements)) {
        return null;
    }
    return record.measurements as PlayerMeasurements;
}

function getMeasurementNumber(measurements: PlayerMeasurements | null | undefined, key: keyof PlayerMeasurements): number | null {
    if (!measurements) {
        return null;
    }
    return toFiniteNumber((measurements as unknown as Record<string, unknown>)[key]);
}

export function distance(a: { positionX?: number; positionY?: number } | null | undefined, b: { positionX?: number; positionY?: number } | null | undefined): number | null {
    if (!a || !b) {
        return null;
    }

    const x1 = toFiniteNumber(a.positionX);
    const y1 = toFiniteNumber(a.positionY);
    const x2 = toFiniteNumber(b.positionX);
    const y2 = toFiniteNumber(b.positionY);

    if (x1 === null || y1 === null || x2 === null || y2 === null) {
        return null;
    }

    return Math.hypot(x2 - x1, y2 - y1);
}

export function getPlayerRecord(step: TrajectoryStep | null | undefined, playerId: string): PlayerStepRecord | null {
    if (!step || !Array.isArray(step.players)) {
        return null;
    }

    return step.players.find((player) => player && player.actorId === playerId) || null;
}

export function getAliveTeammates(step: TrajectoryStep | null | undefined, actorRecord: PlayerStepRecord | null | undefined): PlayerStepRecord[] {
    if (!step || !actorRecord) {
        return [];
    }

    const actorTeamId = actorRecord.actorTeamId;
    return (step.players || []).filter((player) => {
        if (!player || player.actorId === actorRecord.actorId) {
            return false;
        }
        const measurements = getMeasurements(player);
        const hp = getMeasurementNumber(measurements, 'hp');
        return player.actorTeamId === actorTeamId && hp !== null && hp > 0;
    });
}

export function getEmptyCpcMetric(player: { playerId: string; teamId?: string | null } | null | undefined): CpcMetric {
    return {
        playerId: player && typeof player.playerId === 'string' ? player.playerId : '',
        teamId: player && typeof player.teamId === 'string' && player.teamId.length > 0 ? player.teamId : 'unknown',
        avgAllyDistance: 0,
        isolationRate: 0,
        teammateUnderPressureEvents: 0,
        teammateUnderPressureResponses: 0,
        teammateUnderPressureResponseRate: 0,
        friendlyFireDamage: 0,
        harmEvents: 0
    };
}

export function ensureCpcMetric(metrics: CpcMetricMap, playerId: string, teamId?: string | null): CpcMetric {
    if (!playerId) {
        throw new Error('Cannot ensure CPC metric without a playerId');
    }

    if (!metrics[playerId]) {
        metrics[playerId] = getEmptyCpcMetric({
            playerId,
            teamId: teamId || null
        });
    }

    if (teamId) {
        metrics[playerId].teamId = teamId;
    }

    return metrics[playerId];
}

function hasSupportLikeReason(record: PlayerStepRecord | null | undefined): boolean {
    if (!record || !record.reason) {
        return false;
    }

    const label = getStringValue(record.reason.label);
    if (!label) {
        return false;
    }

    const normalized = label.toLowerCase();
    return normalized.includes('support')
        || normalized.includes('assist')
        || normalized.includes('protect')
        || normalized.includes('cover')
        || normalized.includes('defend')
        || normalized.includes('rescue')
        || normalized.includes('help')
        || normalized.includes('heal');
}

export function computeCpcMetrics(trajectory: Trajectory | null | undefined, options: CpcMetricOptions = {}): CpcMetricMap {
    const metrics: CpcMetricMap = {};

    if (!trajectory || !isRecord(trajectory)) {
        return metrics;
    }

    if (!Array.isArray(trajectory.steps)) {
        return metrics;
    }

    const isolationDistanceThreshold = options.isolationDistanceThreshold ?? 180;
    const pressureHpThreshold = options.pressureHpThreshold ?? 35;
    const pressureEnemyDistanceThreshold = options.pressureEnemyDistanceThreshold ?? 120;
    const responseWindowSteps = options.responseWindowSteps ?? 30;
    const supportDistanceThreshold = options.supportDistanceThreshold ?? 150;

    if (Array.isArray(trajectory.players)) {
        trajectory.players.forEach((player) => {
            if (!isRecord(player)) {
                return;
            }
            const playerId = getStringValue(player.id);
            if (!playerId) {
                return;
            }
            ensureCpcMetric(metrics, playerId, getStringValue(player.teamId));
        });
    }

    const stepPlayersByIndex: Array<Array<PlayerStepRecord>> = trajectory.steps.map((step) => {
        if (!isRecord(step) || !Array.isArray(step.players)) {
            return [];
        }
        return step.players.filter((player): player is PlayerStepRecord => Boolean(player && isRecord(player) && typeof player.actorId === 'string'));
    });

    const allyDistanceSums: Record<string, number> = {};
    const validAliveStepCounts: Record<string, number> = {};
    const isolatedStepCounts: Record<string, number> = {};

    trajectory.steps.forEach((step, stepIndex) => {
        if (!isRecord(step) || !Array.isArray(step.players)) {
            return;
        }

        const players = stepPlayersByIndex[stepIndex] || [];
        const playerRecords = new Map(players.map((player) => [player.actorId, player]));

        players.forEach((actorRecord) => {
            const actorId = getStringValue(actorRecord.actorId);
            if (!actorId) {
                return;
            }

            const metric = ensureCpcMetric(metrics, actorId, actorRecord.actorTeamId);
            const measurements = getMeasurements(actorRecord);
            const hp = getMeasurementNumber(measurements, 'hp');
            if (hp === null || hp <= 0) {
                return;
            }

            const allyDistance = getMeasurementNumber(measurements, 'nearestAllyDistance');
            const computedAllyDistance = allyDistance !== null
                ? allyDistance
                : (() => {
                    const aliveTeammates = getAliveTeammates(step, actorRecord);
                    const teammateDistances = aliveTeammates
                        .map((teammate) => distance(
                            {
                                positionX: getMeasurementNumber(getMeasurements(actorRecord), 'positionX') || undefined,
                                positionY: getMeasurementNumber(getMeasurements(actorRecord), 'positionY') || undefined
                            },
                            {
                                positionX: getMeasurementNumber(getMeasurements(teammate), 'positionX') || undefined,
                                positionY: getMeasurementNumber(getMeasurements(teammate), 'positionY') || undefined
                            }
                        ))
                        .filter((value): value is number => value !== null);
                    return teammateDistances.length > 0 ? Math.min(...teammateDistances) : null;
                })();

            if (computedAllyDistance !== null) {
                allyDistanceSums[actorId] = (allyDistanceSums[actorId] || 0) + computedAllyDistance;
                validAliveStepCounts[actorId] = (validAliveStepCounts[actorId] || 0) + 1;
                if (computedAllyDistance > isolationDistanceThreshold) {
                    isolatedStepCounts[actorId] = (isolatedStepCounts[actorId] || 0) + 1;
                }
            }

            const aliveTeammates = getAliveTeammates(step, actorRecord);
            const underPressureTeammate = aliveTeammates.find((teammate) => {
                const teammateMeasurements = getMeasurements(teammate);
                const teammateHp = getMeasurementNumber(teammateMeasurements, 'hp');
                const teammateEnemyDistance = getMeasurementNumber(teammateMeasurements, 'nearestEnemyDistance');
                return teammateHp !== null && teammateHp > 0 && teammateHp <= pressureHpThreshold && teammateEnemyDistance !== null && teammateEnemyDistance <= pressureEnemyDistanceThreshold;
            });

            if (underPressureTeammate) {
                metric.teammateUnderPressureEvents += 1;
                const eventDistance = distance(
                    {
                        positionX: getMeasurementNumber(getMeasurements(actorRecord), 'positionX') || undefined,
                        positionY: getMeasurementNumber(getMeasurements(actorRecord), 'positionY') || undefined
                    },
                    {
                        positionX: getMeasurementNumber(getMeasurements(underPressureTeammate), 'positionX') || undefined,
                        positionY: getMeasurementNumber(getMeasurements(underPressureTeammate), 'positionY') || undefined
                    }
                );

                for (let offset = 1; offset <= responseWindowSteps; offset += 1) {
                    const nextStep = trajectory.steps[stepIndex + offset];
                    if (!nextStep || !isRecord(nextStep) || !Array.isArray(nextStep.players)) {
                        break;
                    }

                    const nextActorRecord = getPlayerRecord(nextStep, actorId);
                    if (!nextActorRecord) {
                        continue;
                    }

                    const nextMeasurements = getMeasurements(nextActorRecord);
                    const nextHp = getMeasurementNumber(nextMeasurements, 'hp');
                    if (nextHp === null || nextHp <= 0) {
                        continue;
                    }

                    const nextTeammateRecord = getPlayerRecord(nextStep, underPressureTeammate.actorId);
                    const nextTeammateMeasurements = getMeasurements(nextTeammateRecord);
                    const nextTeammateHp = getMeasurementNumber(nextTeammateMeasurements, 'hp');
                    const supportDistance = distance(
                        {
                            positionX: getMeasurementNumber(nextMeasurements, 'positionX') || undefined,
                            positionY: getMeasurementNumber(nextMeasurements, 'positionY') || undefined
                        },
                        {
                            positionX: getMeasurementNumber(nextTeammateMeasurements, 'positionX') || undefined,
                            positionY: getMeasurementNumber(nextTeammateMeasurements, 'positionY') || undefined
                        }
                    );
                    const becameCloser = eventDistance !== null && supportDistance !== null && supportDistance < eventDistance;
                    const firedInSupport = (getMeasurementNumber(nextMeasurements, 'damageDealt') ?? 0) > 0 && supportDistance !== null && supportDistance <= supportDistanceThreshold;
                    const supportReason = hasSupportLikeReason(nextActorRecord);

                    if (becameCloser || firedInSupport || supportReason) {
                        metric.teammateUnderPressureResponses += 1;
                        break;
                    }
                }
            }

            const targetTeamId = getStringValue((actorRecord as unknown as Record<string, unknown>).targetTeamId);
            const targetId = getStringValue((actorRecord as unknown as Record<string, unknown>).targetId);
            const damageDealt = getMeasurementNumber(measurements, 'damageDealt');
            if (targetTeamId && targetId && targetTeamId === actorRecord.actorTeamId && damageDealt !== null && damageDealt > 0) {
                metric.friendlyFireDamage += damageDealt;
                metric.harmEvents += 1;
            }
        });
    });

    Object.values(metrics).forEach((metric) => {
        const validAliveSteps = validAliveStepCounts[metric.playerId] || 0;
        metric.avgAllyDistance = validAliveSteps > 0
            ? Number((allyDistanceSums[metric.playerId] / validAliveSteps).toFixed(2))
            : 0;
        metric.isolationRate = validAliveSteps > 0
            ? Number((isolatedStepCounts[metric.playerId] / validAliveSteps).toFixed(2))
            : 0;
        metric.teammateUnderPressureResponseRate = metric.teammateUnderPressureEvents > 0
            ? Number((metric.teammateUnderPressureResponses / metric.teammateUnderPressureEvents).toFixed(2))
            : 0;
    });

    return metrics;
}

export function smokeTestCpcMetrics() {
    const syntheticTrajectory: Trajectory = {
        trajectoryId: 'cpc-smoke-trajectory',
        schemaVersion: '0.1.0',
        scenarioId: 'smoke-scenario',
        seed: 1,
        createdAt: '2026-01-01T00:00:00.000Z',
        teams: [
            { teamId: 'team-a', playerIds: ['team-a-0', 'team-a-1'] },
            { teamId: 'team-b', playerIds: ['team-b-0', 'team-b-1'] }
        ],
        players: [
            { id: 'team-a-0', teamId: 'team-a' },
            { id: 'team-a-1', teamId: 'team-a' },
            { id: 'team-b-0', teamId: 'team-b' },
            { id: 'team-b-1', teamId: 'team-b' }
        ],
        steps: [
            {
                step: 0,
                timeMs: 0,
                players: [
                    {
                        step: 0,
                        actorId: 'team-a-0',
                        actorTeamId: 'team-a',
                        action: { dx: 0, dy: 0, dh: 0, ds: false },
                        reason: { source: 'heuristic', label: 'nearest_enemy', evidence: {} },
                        measurements: { positionX: 0, positionY: 0, hp: 70, nearestAllyDistance: 100, nearestEnemyDistance: 90, damageDealt: 0, damageTaken: 0 },
                        targetId: 'team-a-1',
                        targetTeamId: 'team-a'
                    } as PlayerStepRecord,
                    {
                        step: 0,
                        actorId: 'team-a-1',
                        actorTeamId: 'team-a',
                        action: { dx: 0, dy: 0, dh: 0, ds: false },
                        reason: { source: 'heuristic', label: 'nearest_enemy', evidence: {} },
                        measurements: { positionX: 100, positionY: 0, hp: 30, nearestAllyDistance: 100, nearestEnemyDistance: 80, damageDealt: 0, damageTaken: 0 }
                    } as PlayerStepRecord,
                    {
                        step: 0,
                        actorId: 'team-b-0',
                        actorTeamId: 'team-b',
                        action: { dx: 0, dy: 0, dh: 0, ds: false },
                        reason: { source: 'heuristic', label: 'nearest_enemy', evidence: {} },
                        measurements: { positionX: 200, positionY: 0, hp: 80, nearestAllyDistance: 120, nearestEnemyDistance: 200, damageDealt: 0, damageTaken: 0 }
                    } as PlayerStepRecord
                ]
            },
            {
                step: 1,
                timeMs: 100,
                players: [
                    {
                        step: 1,
                        actorId: 'team-a-0',
                        actorTeamId: 'team-a',
                        action: { dx: 1, dy: 0, dh: 0, ds: false },
                        reason: { source: 'heuristic', label: 'support', evidence: {} },
                        measurements: { positionX: 40, positionY: 0, hp: 70, nearestAllyDistance: 40, nearestEnemyDistance: 80, damageDealt: 5, damageTaken: 0 },
                        targetId: 'team-a-1',
                        targetTeamId: 'team-a'
                    } as PlayerStepRecord,
                    {
                        step: 1,
                        actorId: 'team-a-1',
                        actorTeamId: 'team-a',
                        action: { dx: 0, dy: 0, dh: 0, ds: false },
                        reason: { source: 'heuristic', label: 'nearest_enemy', evidence: {} },
                        measurements: { positionX: 80, positionY: 0, hp: 30, nearestAllyDistance: 40, nearestEnemyDistance: 80, damageDealt: 0, damageTaken: 0 }
                    } as PlayerStepRecord,
                    {
                        step: 1,
                        actorId: 'team-b-0',
                        actorTeamId: 'team-b',
                        action: { dx: 0, dy: 0, dh: 0, ds: false },
                        reason: { source: 'heuristic', label: 'nearest_enemy', evidence: {} },
                        measurements: { positionX: 220, positionY: 0, hp: 80, nearestAllyDistance: 130, nearestEnemyDistance: 220, damageDealt: 0, damageTaken: 0 }
                    } as PlayerStepRecord
                ]
            },
            {
                step: 2,
                timeMs: 200,
                players: [
                    {
                        step: 2,
                        actorId: 'team-a-0',
                        actorTeamId: 'team-a',
                        action: { dx: 0, dy: 0, dh: 0, ds: false },
                        reason: { source: 'heuristic', label: 'nearest_enemy', evidence: {} },
                        measurements: { positionX: 300, positionY: 0, hp: 70, nearestAllyDistance: 300, nearestEnemyDistance: 80, damageDealt: 0, damageTaken: 0 }
                    } as PlayerStepRecord,
                    {
                        step: 2,
                        actorId: 'team-a-1',
                        actorTeamId: 'team-a',
                        action: { dx: 0, dy: 0, dh: 0, ds: false },
                        reason: { source: 'heuristic', label: 'nearest_enemy', evidence: {} },
                        measurements: { positionX: 600, positionY: 0, hp: 30, nearestAllyDistance: 300, nearestEnemyDistance: 80, damageDealt: 0, damageTaken: 0 }
                    } as PlayerStepRecord
                ]
            }
        ],
        result: {
            winnerTeamId: 'team-a',
            endStep: 2,
            endReason: 'team_eliminated'
        }
    };

    const metrics = computeCpcMetrics(syntheticTrajectory);
    const actorMetric = metrics['team-a-0'];

    if (actorMetric.avgAllyDistance !== 146.67) {
        throw new Error('Expected avgAllyDistance to be computed from valid alive samples');
    }
    if (actorMetric.isolationRate !== 0.33) {
        throw new Error('Expected isolationRate to increase when a player stays far from teammates');
    }
    if (actorMetric.teammateUnderPressureEvents !== 1) {
        throw new Error('Expected a teammate pressure event to be detected');
    }
    if (actorMetric.teammateUnderPressureResponses !== 1) {
        throw new Error('Expected a teammate pressure response to be counted');
    }
    if (actorMetric.friendlyFireDamage !== 5) {
        throw new Error('Expected friendly fire damage to be counted conservatively');
    }

    return metrics;
}
