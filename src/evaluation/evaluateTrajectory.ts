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
        teammateUnderPressureEvents: number;
        teammateUnderPressureResponses: number;
        teammateResponseRate: number;
        isolationRate: number;
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
        - (100 * cpc.teammateResponseRate)
        + (50 * cpc.isolationRate)
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
    const teams: Record<string, EvaluationTeamSummary & { _responseRateSum: number; _isolationRateSum: number; _scoreSum: number; _playerCount: number }> = {};

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
            teammateUnderPressureEvents: cpcMetric.teammateUnderPressureEvents,
            teammateUnderPressureResponses: cpcMetric.teammateUnderPressureResponses,
            teammateResponseRate: cpcMetric.teammateResponseRate,
            isolationRate: cpcMetric.isolationRate
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
                _playerCount: 0
            };
        }

        const team = teams[teamId];
        team.playerIds.push(playerId);
        team.damageDealt += player.damageDealt;
        team.damageTaken += player.damageTaken;
        team.survivalSteps += player.survivalSteps;
        team._responseRateSum += cpc.teammateResponseRate;
        team._isolationRateSum += cpc.isolationRate;
        team._scoreSum += evaluationScore;
        team._playerCount += 1;
    });

    Object.values(teams).forEach((team) => {
        team.avgTeammateResponseRate = team._playerCount > 0 ? Number((team._responseRateSum / team._playerCount).toFixed(2)) : 0;
        team.avgIsolationRate = team._playerCount > 0 ? Number((team._isolationRateSum / team._playerCount).toFixed(2)) : 0;
        team.avgEvaluationScore = team._playerCount > 0 ? Number((team._scoreSum / team._playerCount).toFixed(2)) : 0;
        delete team._responseRateSum;
        delete team._isolationRateSum;
        delete team._scoreSum;
        delete team._playerCount;
    });

    return {
        trajectoryId: trajectory.trajectoryId,
        schemaVersion: trajectory.schemaVersion,
        players,
        teams
    };
}
