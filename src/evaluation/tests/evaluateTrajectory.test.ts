import assert from 'node:assert/strict';
import { test } from 'node:test';
import { synthetic2v2Trajectory } from '../../engine/traces/fixtures/synthetic2v2Trajectory';
import { evaluateTrajectory } from '../evaluateTrajectory';

test('evaluateTrajectory combines player and CPC metrics', () => {
    const evaluation = evaluateTrajectory(synthetic2v2Trajectory);

    assert.equal(evaluation.trajectoryId, synthetic2v2Trajectory.trajectoryId);
    assert.equal(evaluation.schemaVersion, synthetic2v2Trajectory.schemaVersion);
    assert.equal(Object.keys(evaluation.players).length, 4);
    assert.equal(Object.keys(evaluation.teams).length, 2);
});

test('evaluateTrajectory includes evaluationScore', () => {
    const evaluation = evaluateTrajectory(synthetic2v2Trajectory);

    const player = evaluation.players['team-a-0'];
    assert.equal(typeof player.evaluationScore, 'number');
});

test('evaluateTrajectory includes team aggregation', () => {
    const evaluation = evaluateTrajectory(synthetic2v2Trajectory);

    const team = evaluation.teams['team-a'];
    assert.deepEqual(team.playerIds.sort(), ['team-a-0', 'team-a-1']);
    assert.equal(team.damageDealt, 1);
    assert.equal(team.damageTaken, 2);
    assert.equal(team.survivalSteps, 4);
});

test('evaluateTrajectory output is JSON stringify compatible', () => {
    const evaluation = evaluateTrajectory(synthetic2v2Trajectory);
    assert.doesNotThrow(() => JSON.stringify(evaluation));
});
