import assert from 'node:assert/strict';
import { test } from 'node:test';
import { synthetic2v2Trajectory } from '../../../engine/traces/fixtures/synthetic2v2Trajectory';
import { computePlayerMetrics } from '../playerMetrics';

function cloneTrajectory(trajectory) {
    return JSON.parse(JSON.stringify(trajectory));
}

test('computePlayerMetrics returns metrics for all 4 players', () => {
    const metrics = computePlayerMetrics(synthetic2v2Trajectory);

    assert.equal(Object.keys(metrics).length, 4);
    assert.equal(metrics['team-a-0'].playerId, 'team-a-0');
    assert.equal(metrics['team-b-1'].teamId, 'team-b');
});

test('damageDealt sums correctly', () => {
    const metrics = computePlayerMetrics(synthetic2v2Trajectory);
    assert.equal(metrics['team-a-0'].damageDealt, 1);
});

test('damageTaken sums correctly', () => {
    const metrics = computePlayerMetrics(synthetic2v2Trajectory);
    assert.equal(metrics['team-a-1'].damageTaken, 2);
});

test('survivalSteps counts only state.alive true and state.hp > 0', () => {
    const trajectory = cloneTrajectory(synthetic2v2Trajectory);
    trajectory.steps[1].players[0].state.alive = false;

    const metrics = computePlayerMetrics(trajectory);

    assert.equal(metrics['team-a-0'].survivalSteps, 1);
});

test('missing damage fields default to 0', () => {
    const trajectory = cloneTrajectory(synthetic2v2Trajectory);
    delete trajectory.steps[0].players[0].measurements.damageDealt;
    delete trajectory.steps[0].players[0].measurements.damageTaken;

    const metrics = computePlayerMetrics(trajectory);

    assert.equal(metrics['team-a-0'].damageDealt, 1);
    assert.equal(metrics['team-a-0'].damageTaken, 0);
});
