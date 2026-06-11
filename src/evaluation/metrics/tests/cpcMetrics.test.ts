import assert from 'node:assert/strict';
import { test } from 'node:test';
import { synthetic2v2Trajectory } from '../../../engine/traces/fixtures/synthetic2v2Trajectory';
import { computeCpcMetrics } from '../cpcMetrics';

function cloneTrajectory(trajectory) {
    return JSON.parse(JSON.stringify(trajectory));
}

test('computeCpcMetrics returns metrics for all 4 players', () => {
    const metrics = computeCpcMetrics(synthetic2v2Trajectory);

    assert.equal(Object.keys(metrics).length, 4);
    assert.equal(metrics['team-a-0'].playerId, 'team-a-0');
});

test('isolationRate uses measurements.nearestAllyDistance when available', () => {
    const metrics = computeCpcMetrics(synthetic2v2Trajectory);
    assert.equal(metrics['team-a-0'].isolationRate, 0);
});

test('isolationRate can fallback to player state positions', () => {
    const trajectory = cloneTrajectory(synthetic2v2Trajectory);
    delete trajectory.steps[1].players[0].measurements.nearestAllyDistance;

    const metrics = computeCpcMetrics(trajectory);

    assert.equal(metrics['team-a-0'].isolationRate, 0);
});

test('isolationRate increases when teammate distance exceeds threshold', () => {
    const trajectory = cloneTrajectory(synthetic2v2Trajectory);
    delete trajectory.steps[1].players[0].measurements.nearestAllyDistance;
    trajectory.steps[1].players[0].state.positionX = 300;
    trajectory.steps[1].players[0].state.positionY = 0;
    trajectory.steps[1].players[1].state.positionX = 600;
    trajectory.steps[1].players[1].state.positionY = 0;

    const metrics = computeCpcMetrics(trajectory);

    assert.equal(metrics['team-a-0'].isolationRate, 0.5);
});

test('teammateUnderPressureEvents are detected from teammate hp and nearestEnemyDistance', () => {
    const trajectory = cloneTrajectory(synthetic2v2Trajectory);
    trajectory.steps[0].players[1].state.hp = 20;
    trajectory.steps[0].players[1].measurements.nearestEnemyDistance = 100;
    trajectory.steps[1].players[0].reason.label = 'support';

    const metrics = computeCpcMetrics(trajectory);

    assert.equal(metrics['team-a-0'].teammateUnderPressureEvents, 1);
});

test('teammateResponseRate returns 0 when there are no pressure events', () => {
    const metrics = computeCpcMetrics(synthetic2v2Trajectory);

    assert.equal(metrics['team-a-0'].teammateResponseRate, 0);
});

test('teammateUnderPressureResponses are counted conservatively', () => {
    const trajectory = cloneTrajectory(synthetic2v2Trajectory);
    trajectory.steps[0].players[1].state.hp = 20;
    trajectory.steps[0].players[1].measurements.nearestEnemyDistance = 100;
    trajectory.steps[1].players[0].reason.label = 'support';

    const metrics = computeCpcMetrics(trajectory);

    assert.equal(metrics['team-a-0'].teammateUnderPressureResponses, 1);
    assert.equal(metrics['team-a-0'].teammateResponseRate, 1);
});
