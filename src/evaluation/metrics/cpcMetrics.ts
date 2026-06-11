import { PlayerStepRecord, Trajectory, TrajectoryStep } from '../../engine/traces/trace';

export interface CpcMetric {
    playerId: string;
    teamId: string;
    teammateUnderPressureEvents: number;
    teammateUnderPressureResponses: number;
    teammateResponseRate: number;
    isolationRate: number;
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
    fireThreshold?: number;
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

function getMeasurements(record: PlayerStepRecord | null | undefined): Record<string, unknown> | null {
    if (!record || !isRecord(record.measurements)) {
        return null;
    }
    return record.measurements;
}

function getStateNumber(record: PlayerStepRecord | null | undefined, key: string): number | null {
    if (!record || !isRecord(record.state)) {
        return null;
    }
    return toFiniteNumber(record.state[key]);
}

function getMeasurementNumber(record: PlayerStepRecord | null | undefined, key: string): number | null {
    const measurements = getMeasurements(record);
    if (!measurements) {
        return null;
    }
    return toFiniteNumber(measurements[key]);
}

function getPosition(record: PlayerStepRecord | null | undefined): { positionX: number; positionY: number } | null {
    const positionX = getStateNumber(record, 'positionX');
    const positionY = getStateNumber(record, 'positionY');
    if (positionX !== null && positionY !== null) {
        return { positionX, positionY };
    }

    const measurementX = getMeasurementNumber(record, 'positionX');
    const measurementY = getMeasurementNumber(record, 'positionY');
    if (measurementX === null || measurementY === null) {
        return null;
    }

    return { positionX: measurementX, positionY: measurementY };
}

function getHp(record: PlayerStepRecord | null | undefined): number | null {
    const stateHp = getStateNumber(record, 'hp');
    if (stateHp !== null) {
        return stateHp;
    }
    return getMeasurementNumber(record, 'hp');
}

function isAlive(record: PlayerStepRecord | null | undefined): boolean {
    if (!record || !isRecord(record.state)) {
        return false;
    }
    if (typeof record.state.alive === 'boolean') {
        return record.state.alive;
    }
    const hp = getHp(record);
    return hp !== null ? hp > 0 : false;
}

function distance(a: { positionX: number; positionY: number } | null, b: { positionX: number; positionY: number } | null): number | null {
    if (!a || !b) {
        return null;
    }
    return Math.hypot(b.positionX - a.positionX, b.positionY - a.positionY);
}

function getPlayerRecord(step: TrajectoryStep | null | undefined, playerId: string): PlayerStepRecord | null {
    if (!step || !Array.isArray(step.players)) {
        return null;
    }
    return step.players.find((player) => player && player.actorId === playerId) || null;
}

function getAliveTeammates(step: TrajectoryStep, actorRecord: PlayerStepRecord): PlayerStepRecord[] {
    return (step.players || []).filter((player) => player && player.actorTeamId === actorRecord.actorTeamId && player.actorId !== actorRecord.actorId && isAlive(player));
}

export function getEmptyCpcMetric(player: { playerId: string; teamId?: string | null } | null | undefined): CpcMetric {
    return {
        playerId: player && typeof player.playerId === 'string' ? player.playerId : '',
        teamId: player && typeof player.teamId === 'string' && player.teamId.length > 0 ? player.teamId : 'unknown',
        teammateUnderPressureEvents: 0,
        teammateUnderPressureResponses: 0,
        teammateResponseRate: 0,
        isolationRate: 0
    };
}

export function ensureCpcMetric(metrics: CpcMetricMap, playerId: string, teamId?: string | null): CpcMetric {
    if (!playerId) {
        throw new Error('Cannot ensure CPC metric without a playerId');
    }

    if (!metrics[playerId]) {
        metrics[playerId] = getEmptyCpcMetric({ playerId, teamId: teamId || null });
    }

    if (teamId) {
        metrics[playerId].teamId = teamId;
    }

    return metrics[playerId];
}

function hasSupportLikeReason(record: PlayerStepRecord | null | undefined): boolean {
    const label = record?.reason?.label;
    return typeof label === 'string' && /support|assist|protect/i.test(label);
}

export function computeCpcMetrics(trajectory: Trajectory | null | undefined, options: CpcMetricOptions = {}): CpcMetricMap {
    const metrics: CpcMetricMap = {};

    if (!trajectory || !Array.isArray(trajectory.steps)) {
        return metrics;
    }

    const isolationDistanceThreshold = options.isolationDistanceThreshold ?? 180;
    const pressureHpThreshold = options.pressureHpThreshold ?? 35;
    const pressureEnemyDistanceThreshold = options.pressureEnemyDistanceThreshold ?? 120;
    const responseWindowSteps = options.responseWindowSteps ?? 30;
    const supportDistanceThreshold = options.supportDistanceThreshold ?? 150;
    const fireThreshold = options.fireThreshold ?? 0;
    const validAliveStepsByPlayer: Record<string, number> = {};
    const isolatedStepsByPlayer: Record<string, number> = {};

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

    trajectory.steps.forEach((step, stepIndex) => {
        if (!isRecord(step) || !Array.isArray(step.players)) {
            return;
        }

        const players = step.players.filter((player): player is PlayerStepRecord => Boolean(player && isRecord(player) && typeof player.actorId === 'string'));

        players.forEach((actorRecord) => {
            const actorId = actorRecord.actorId;
            const actorMetric = ensureCpcMetric(metrics, actorId, actorRecord.actorTeamId);

            if (!isAlive(actorRecord)) {
                return;
            }

            const actorPosition = getPosition(actorRecord);
            const nearestAllyDistance = getMeasurementNumber(actorRecord, 'nearestAllyDistance');
            const computedAllyDistance = nearestAllyDistance !== null
                ? nearestAllyDistance
                : (() => {
                    const teammates = getAliveTeammates(step, actorRecord);
                    const teammateDistances = teammates
                        .map((teammate) => distance(actorPosition, getPosition(teammate)))
                        .filter((value): value is number => value !== null);
                    return teammateDistances.length ? Math.min(...teammateDistances) : null;
                })();

            if (computedAllyDistance !== null && computedAllyDistance > isolationDistanceThreshold) {
                isolatedStepsByPlayer[actorId] = (isolatedStepsByPlayer[actorId] || 0) + 1;
            }
            if (computedAllyDistance !== null) {
                validAliveStepsByPlayer[actorId] = (validAliveStepsByPlayer[actorId] || 0) + 1;
            }

            const pressureTeammate = getAliveTeammates(step, actorRecord).find((teammate) => {
                const teammateHp = getHp(teammate);
                const teammateEnemyDistance = getMeasurementNumber(teammate, 'nearestEnemyDistance');
                return teammateHp !== null
                    && teammateHp > 0
                    && teammateHp <= pressureHpThreshold
                    && teammateEnemyDistance !== null
                    && teammateEnemyDistance <= pressureEnemyDistanceThreshold;
            });

            if (!pressureTeammate) {
                return;
            }

            actorMetric.teammateUnderPressureEvents += 1;
            const pressureDistance = distance(actorPosition, getPosition(pressureTeammate));

            for (let offset = 1; offset <= responseWindowSteps; offset += 1) {
                const nextStep = trajectory.steps[stepIndex + offset];
                if (!isRecord(nextStep) || !Array.isArray(nextStep.players)) {
                    break;
                }

                const nextActorRecord = getPlayerRecord(nextStep, actorId);
                if (!nextActorRecord || !isAlive(nextActorRecord)) {
                    continue;
                }

                const nextTeammateRecord = getPlayerRecord(nextStep, pressureTeammate.actorId);
                const nextActorPosition = getPosition(nextActorRecord);
                const nextTeammatePosition = getPosition(nextTeammateRecord);
                const supportDistance = distance(nextActorPosition, nextTeammatePosition);
                const movedCloser = pressureDistance !== null && supportDistance !== null && supportDistance < pressureDistance;
                const fired = toFiniteNumber((nextActorRecord.action as unknown as Record<string, unknown>)?.fire) !== null
                    && toFiniteNumber((nextActorRecord.action as unknown as Record<string, unknown>)?.fire) > fireThreshold
                    && supportDistance !== null
                    && supportDistance <= supportDistanceThreshold;
                const supportReason = hasSupportLikeReason(nextActorRecord);

                if (movedCloser || fired || supportReason) {
                    actorMetric.teammateUnderPressureResponses += 1;
                    break;
                }
            }
        });
    });

    Object.values(metrics).forEach((metric) => {
        const validAliveSteps = validAliveStepsByPlayer[metric.playerId] || 0;
        metric.isolationRate = validAliveSteps > 0
            ? Number(((isolatedStepsByPlayer[metric.playerId] || 0) / validAliveSteps).toFixed(2))
            : 0;
        metric.teammateResponseRate = metric.teammateUnderPressureEvents > 0
            ? Number((metric.teammateUnderPressureResponses / metric.teammateUnderPressureEvents).toFixed(2))
            : 0;
    });

    return metrics;
}
