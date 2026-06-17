import assert from 'node:assert/strict';
import { test } from 'node:test';
import { synthetic2v2Trajectory } from '../../engine/traces/fixtures/synthetic2v2Trajectory';
import { evaluateTrajectory } from '../evaluateTrajectory';
import {
    BaselineComparisonRow,
    createBaselineRunResult,
    groupBaselineRowsBySeed,
    summarizeBaselineRows
} from '../baselineComparison';

function createSoloTrajectory() {
    const trajectory = JSON.parse(JSON.stringify(synthetic2v2Trajectory));
    trajectory.battleConfig = { mode: 'solo', teamCount: 4, playersPerTeam: 1 };
    trajectory.teams = ['team-0-0', 'team-1-0', 'team-2-0', 'team-3-0'].map((playerId, index) => ({
        teamId: `team-${index}`,
        playerIds: [playerId]
    }));
    trajectory.players = trajectory.teams.map((team, index) => ({
        id: team.playerIds[0],
        teamId: team.teamId,
        policyId: index === 0 ? 'user_controlled' : 'random'
    }));
    trajectory.initialState.players.forEach((player, index) => {
        player.id = trajectory.players[index].id;
        player.teamId = trajectory.players[index].teamId;
    });
    trajectory.steps.forEach((step) => {
        step.players.forEach((player, index) => {
            player.actorId = trajectory.players[index].id;
            player.actorTeamId = trajectory.players[index].teamId;
        });
        step.events = [];
    });
    return trajectory;
}

test('random baseline produces result rows', () => {
    const trajectory = createSoloTrajectory();
    trajectory.players.forEach((player) => {
        player.policyId = 'random';
    });
    const evaluation = evaluateTrajectory(trajectory);
    const result = createBaselineRunResult({
        config: { mode: 'solo', seed: 42, policyType: 'random', playerCount: 4, maxSteps: 100 },
        trajectory,
        evaluation,
        runId: 'random-42'
    });

    assert.equal(result.rows.length, 4);
    assert.equal(result.rows[0].policyType, 'random');
    assert.equal(result.rows[0].cooperation.applicable, false);
});

test('user-controlled baseline result shape can be represented', () => {
    const trajectory = createSoloTrajectory();
    const evaluation = evaluateTrajectory(trajectory);
    const result = createBaselineRunResult({
        config: { mode: 'solo', seed: 7, policyType: 'user-controlled', playerCount: 4, maxSteps: 100 },
        trajectory,
        evaluation,
        runId: 'human-7'
    });

    const humanRow = result.rows.find((row) => row.policyType === 'user-controlled');
    assert.ok(humanRow);
    assert.equal(humanRow?.seed, 7);
    assert.equal(humanRow?.runId, 'human-7');
});

test('baseline comparison summary computes averages', () => {
    const rows: BaselineComparisonRow[] = [
        { seed: 1, runId: 'a', playerId: 'p1', teamId: 't1', policyType: 'random', damageDealt: 2, damageTaken: 4, survivalSteps: 10, aliveAtEnd: true, cooperation: { applicable: false } },
        { seed: 2, runId: 'b', playerId: 'p2', teamId: 't2', policyType: 'random', damageDealt: 4, damageTaken: 2, survivalSteps: 20, aliveAtEnd: false, cooperation: { applicable: false } }
    ];

    const summary = summarizeBaselineRows(rows)[0];

    assert.equal(summary.policyType, 'random');
    assert.equal(summary.runCount, 2);
    assert.equal(summary.avgDamageDealt, 3);
    assert.equal(summary.avgDamageTaken, 3);
    assert.equal(summary.avgSurvivalSteps, 15);
    assert.equal(summary.survivalRate, 0.5);
    assert.deepEqual(summary.seeds, [1, 2]);
});

test('same-seed comparison groups runs by seed', () => {
    const rows: BaselineComparisonRow[] = [
        { seed: 5, runId: 'random-5', playerId: 'p1', teamId: 't1', policyType: 'random', damageDealt: 1, damageTaken: 3, survivalSteps: 5, aliveAtEnd: false, cooperation: { applicable: false } },
        { seed: 5, runId: 'human-5', playerId: 'p2', teamId: 't2', policyType: 'user-controlled', damageDealt: 3, damageTaken: 1, survivalSteps: 8, aliveAtEnd: true, cooperation: { applicable: false } },
        { seed: 6, runId: 'random-6', playerId: 'p3', teamId: 't3', policyType: 'random', damageDealt: 0, damageTaken: 4, survivalSteps: 4, aliveAtEnd: false, cooperation: { applicable: false } }
    ];

    const groups = groupBaselineRowsBySeed(rows);

    assert.equal(groups[5].length, 2);
    assert.equal(groups[6].length, 1);
});

