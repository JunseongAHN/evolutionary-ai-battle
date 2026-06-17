import { Trajectory } from '../engine/traces/trace';
import { EvaluationPlayerSummary, TrajectoryEvaluation } from './evaluateTrajectory';

export type BaselinePolicyType = 'random' | 'user-controlled' | 'future-policy';

export interface BaselineRunConfig {
    mode: 'solo';
    seed: number;
    policyType: 'random' | 'user-controlled';
    playerCount: number;
    maxSteps: number;
    runLabel?: string;
}

export interface BaselineComparisonRow {
    seed: number;
    runId: string;
    runLabel?: string;
    playerId: string;
    teamId: string;
    policyType: BaselinePolicyType;
    damageDealt: number;
    damageTaken: number;
    survivalSteps: number;
    aliveAtEnd: boolean;
    kills?: number;
    deaths?: number;
    wastefulFireCount?: number;
    cooperation: {
        applicable: boolean;
    };
}

export interface BaselineComparisonSummary {
    policyType: string;
    runCount: number;
    avgDamageDealt: number;
    avgDamageTaken: number;
    avgSurvivalSteps: number;
    survivalRate: number;
    seeds: number[];
}

export interface BaselineRunResult {
    runId: string;
    runLabel?: string;
    seed: number;
    mode: 'solo';
    playerCount: number;
    policyType: BaselinePolicyType;
    createdAt: string;
    rows: BaselineComparisonRow[];
    summaries: BaselineComparisonSummary[];
}

function round(value: number): number {
    return Number(value.toFixed(2));
}

function countEvents(trajectory: Trajectory, playerId: string, type: string, key: 'actorId' | 'targetId'): number {
    return (trajectory.steps || []).reduce((total, step) => {
        return total + (step.events || []).filter((event) => event.type === type && event[key] === playerId).length;
    }, 0);
}

function countWastefulFire(trajectory: Trajectory, playerId: string): number {
    return (trajectory.steps || []).reduce((total, step) => {
        const player = (step.players || []).find((record) => record.actorId === playerId);
        if (!player) {
            return total;
        }
        const fired = (player.action?.fire || 0) > 0 || player.measurements?.didFire === true;
        const enemyInRange = player.measurements?.enemyInRange === true;
        return total + (fired && !enemyInRange ? 1 : 0);
    }, 0);
}

function getAliveAtEnd(trajectory: Trajectory, playerId: string): boolean {
    const lastStep = (trajectory.steps || [])[Math.max((trajectory.steps || []).length - 1, 0)];
    if (!lastStep) {
        const initialPlayer = trajectory.initialState?.players?.find((player) => player.id === playerId);
        return initialPlayer ? initialPlayer.alive : false;
    }

    const player = (lastStep.players || []).find((record) => record.actorId === playerId);
    return player ? player.state?.alive === true && (player.state?.hp || 0) > 0 : false;
}

function getPlayerPolicyType(
    trajectory: Trajectory,
    playerId: string,
    fallbackPolicyType: BaselinePolicyType
): BaselinePolicyType {
    const policyId = trajectory.players?.find((player) => player.id === playerId)?.policyId;
    if (policyId === 'random') {
        return 'random';
    }
    if (policyId === 'user_controlled') {
        return 'user-controlled';
    }
    if (policyId === 'future-policy') {
        return 'future-policy';
    }
    return fallbackPolicyType;
}

export function createBaselineComparisonRow({
    trajectory,
    playerSummary,
    seed,
    runId,
    runLabel,
    policyType
}: {
    trajectory: Trajectory;
    playerSummary: EvaluationPlayerSummary;
    seed: number;
    runId: string;
    runLabel?: string;
    policyType: BaselinePolicyType;
}): BaselineComparisonRow {
    const playerId = playerSummary.playerId;
    return {
        seed,
        runId,
        runLabel,
        playerId,
        teamId: playerSummary.teamId,
        policyType: getPlayerPolicyType(trajectory, playerId, policyType),
        damageDealt: playerSummary.player.damageDealt,
        damageTaken: playerSummary.player.damageTaken,
        survivalSteps: playerSummary.player.survivalSteps,
        aliveAtEnd: getAliveAtEnd(trajectory, playerId),
        kills: countEvents(trajectory, playerId, 'elimination', 'actorId'),
        deaths: countEvents(trajectory, playerId, 'elimination', 'targetId'),
        wastefulFireCount: countWastefulFire(trajectory, playerId),
        cooperation: {
            applicable: playerSummary.cpc.applicable
        }
    };
}

export function summarizeBaselineRows(rows: BaselineComparisonRow[]): BaselineComparisonSummary[] {
    const groups = rows.reduce<Record<string, BaselineComparisonRow[]>>((nextGroups, row) => {
        nextGroups[row.policyType] = [...(nextGroups[row.policyType] || []), row];
        return nextGroups;
    }, {});

    return Object.entries(groups).map(([policyType, policyRows]) => {
        const runCount = policyRows.length;
        return {
            policyType,
            runCount,
            avgDamageDealt: round(policyRows.reduce((sum, row) => sum + row.damageDealt, 0) / runCount),
            avgDamageTaken: round(policyRows.reduce((sum, row) => sum + row.damageTaken, 0) / runCount),
            avgSurvivalSteps: round(policyRows.reduce((sum, row) => sum + row.survivalSteps, 0) / runCount),
            survivalRate: round(policyRows.filter((row) => row.aliveAtEnd).length / runCount),
            seeds: Array.from(new Set(policyRows.map((row) => row.seed))).sort((a, b) => a - b)
        };
    });
}

export function groupBaselineRowsBySeed(rows: BaselineComparisonRow[]): Record<number, BaselineComparisonRow[]> {
    return rows.reduce<Record<number, BaselineComparisonRow[]>>((groups, row) => {
        groups[row.seed] = [...(groups[row.seed] || []), row];
        return groups;
    }, {});
}

export function createBaselineRunResult({
    config,
    trajectory,
    evaluation,
    runId
}: {
    config: BaselineRunConfig;
    trajectory: Trajectory;
    evaluation: TrajectoryEvaluation;
    runId?: string;
}): BaselineRunResult {
    const resolvedRunId = runId || `baseline-${config.policyType}-${config.seed}-${Date.now()}`;
    const rows = Object.values(evaluation.players).map((playerSummary) => createBaselineComparisonRow({
        trajectory,
        playerSummary,
        seed: config.seed,
        runId: resolvedRunId,
        runLabel: config.runLabel,
        policyType: config.policyType
    }));

    return {
        runId: resolvedRunId,
        runLabel: config.runLabel,
        seed: config.seed,
        mode: 'solo',
        playerCount: config.playerCount,
        policyType: config.policyType,
        createdAt: new Date().toISOString(),
        rows,
        summaries: summarizeBaselineRows(rows)
    };
}

