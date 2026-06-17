import { Trajectory } from '../engine/traces/trace';
import { validateReplayableTrajectory } from '../engine/traces/trajectoryReplay';
import { computeCpcMetrics, CpcMetricMap, CpcMetricOptions } from './metrics/cpcMetrics';
import { computePlayerMetrics, PlayerMetricMap } from './metrics/playerMetrics';

export interface EvaluationPlayerSummary {
    playerId: string;
    teamId: string;
    player: {
        damageDealt: number;
        damageTaken: number;
        survivalSteps: number;
    };
    cpc: {
        applicable: boolean;
        teammateUnderPressureEvents: number;
        teammateUnderPressureResponses: number;
        teammateResponseRate: number;
        isolatedSteps: number;
        isolationRate: number;
        avgAllyDistance: number | null;
    };
    evaluationScore: number;
}

export interface EvaluationTeamSummary {
    teamId: string;
    playerIds: string[];
    damageDealt: number;
    damageTaken: number;
    survivalSteps: number;
    avgTeammateResponseRate: number;
    avgIsolationRate: number;
    avgEvaluationScore: number;
}

export interface TrajectoryEvaluation {
    trajectoryId: string;
    schemaVersion: string;
    players: Record<string, EvaluationPlayerSummary>;
    teams: Record<string, EvaluationTeamSummary>;
}

function getEvaluationScore(player: EvaluationPlayerSummary['player'], cpc: EvaluationPlayerSummary['cpc']): number {
    // This is an initial heuristic score.
    // It is only for comparison and future policy tuning.
    // Raw metrics remain the primary output.
    return Number((
        player.damageDealt
        - (0.5 * player.damageTaken)
        - (0.1 * player.survivalSteps)
        - (100 * (cpc.applicable ? cpc.teammateResponseRate : 0))
        + (50 * (cpc.applicable ? cpc.isolationRate : 0))
    ).toFixed(2));
}

export function evaluateTrajectory(trajectory: Trajectory | null | undefined, options: CpcMetricOptions = {}): TrajectoryEvaluation {
    const errors = validateReplayableTrajectory(trajectory);
    if (errors.length) {
        throw new Error(`Invalid trajectory: ${errors.join('; ')}`);
    }

    const playerMetrics: PlayerMetricMap = computePlayerMetrics(trajectory);
    const cpcMetrics: CpcMetricMap = computeCpcMetrics(trajectory, options);
    const players: Record<string, EvaluationPlayerSummary> = {};
    const teams: Record<string, EvaluationTeamSummary & { _responseRateSum: number; _isolationRateSum: number; _scoreSum: number; _playerCount: number; _cpcPlayerCount: number }> = {};

    Object.keys(playerMetrics).forEach((playerId) => {
        const playerMetric = playerMetrics[playerId];
        const cpcMetric = cpcMetrics[playerId];
        if (!playerMetric || !cpcMetric) {
            return;
        }

        const player = {
            damageDealt: playerMetric.damageDealt,
            damageTaken: playerMetric.damageTaken,
            survivalSteps: playerMetric.survivalSteps
        };

        const cpc = {
            applicable: cpcMetric.applicable,
            teammateUnderPressureEvents: cpcMetric.teammateUnderPressureEvents,
            teammateUnderPressureResponses: cpcMetric.teammateUnderPressureResponses,
            teammateResponseRate: cpcMetric.teammateResponseRate,
            isolatedSteps: cpcMetric.isolatedSteps,
            isolationRate: cpcMetric.isolationRate,
            avgAllyDistance: cpcMetric.avgAllyDistance
        };

        const evaluationScore = getEvaluationScore(player, cpc);

        players[playerId] = {
            playerId,
            teamId: playerMetric.teamId || cpcMetric.teamId,
            player,
            cpc,
            evaluationScore
        };

        const teamId = playerMetric.teamId || cpcMetric.teamId || 'unknown';
        if (!teams[teamId]) {
            teams[teamId] = {
                teamId,
                playerIds: [],
                damageDealt: 0,
                damageTaken: 0,
                survivalSteps: 0,
                avgTeammateResponseRate: 0,
                avgIsolationRate: 0,
                avgEvaluationScore: 0,
                _responseRateSum: 0,
                _isolationRateSum: 0,
                _scoreSum: 0,
                _playerCount: 0,
                _cpcPlayerCount: 0
            };
        }

        const team = teams[teamId];
        team.playerIds.push(playerId);
        team.damageDealt += player.damageDealt;
        team.damageTaken += player.damageTaken;
        team.survivalSteps += player.survivalSteps;
        if (cpc.applicable) {
            team._responseRateSum += cpc.teammateResponseRate;
            team._isolationRateSum += cpc.isolationRate;
            team._cpcPlayerCount += 1;
        }
        team._scoreSum += evaluationScore;
        team._playerCount += 1;
    });

    Object.values(teams).forEach((team) => {
        team.avgTeammateResponseRate = team._cpcPlayerCount > 0 ? Number((team._responseRateSum / team._cpcPlayerCount).toFixed(2)) : 0;
        team.avgIsolationRate = team._cpcPlayerCount > 0 ? Number((team._isolationRateSum / team._cpcPlayerCount).toFixed(2)) : 0;
        team.avgEvaluationScore = team._playerCount > 0 ? Number((team._scoreSum / team._playerCount).toFixed(2)) : 0;
        delete team._responseRateSum;
        delete team._isolationRateSum;
        delete team._scoreSum;
        delete team._playerCount;
        delete team._cpcPlayerCount;
    });

    return {
        trajectoryId: trajectory.trajectoryId,
        schemaVersion: trajectory.schemaVersion,
        players,
        teams
    };
}
