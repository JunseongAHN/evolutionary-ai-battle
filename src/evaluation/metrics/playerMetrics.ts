import { Trajectory } from '../../engine/traces/trace';

export interface PlayerMetric {
    playerId: string;
    teamId: string;
    damageDealt: number;
    damageTaken: number;
    survivalSteps: number;
}

export interface PlayerMetricMap {
    [playerId: string]: PlayerMetric;
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null;
}

function toFiniteNumber(value: unknown): number {
    return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function getStringValue(value: unknown): string | null {
    return typeof value === 'string' && value.length > 0 ? value : null;
}

export function getEmptyPlayerMetric(player: { playerId: string; teamId?: string | null } | null | undefined): PlayerMetric {
    return {
        playerId: player && typeof player.playerId === 'string' ? player.playerId : '',
        teamId: player && typeof player.teamId === 'string' && player.teamId.length > 0 ? player.teamId : 'unknown',
        damageDealt: 0,
        damageTaken: 0,
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

export function computePlayerMetrics(trajectory: Trajectory | null | undefined): PlayerMetricMap {
    const metrics: PlayerMetricMap = {};

    if (!trajectory || !Array.isArray(trajectory.steps)) {
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
            const state = isRecord(player.state) ? player.state : null;

            metric.damageDealt += toFiniteNumber(measurements?.damageDealt);
            metric.damageTaken += toFiniteNumber(measurements?.damageTaken);
            if (state && state.alive === true && toFiniteNumber(state.hp) > 0) {
                metric.survivalSteps += 1;
            }
        });
    });

    return metrics;
}
