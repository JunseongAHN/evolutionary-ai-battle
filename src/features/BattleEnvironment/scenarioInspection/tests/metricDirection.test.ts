import assert from 'node:assert/strict';
import { test } from 'node:test';
import { synthetic2v2Trajectory } from '../../../../engine/traces/fixtures/synthetic2v2Trajectory';
import { scenarioCatalog } from '../scenarioCatalog';
import { computeFiveStepMetricDirection } from '../metricDirection';

test('computeFiveStepMetricDirection returns a conservative summary', () => {
    const result = computeFiveStepMetricDirection(synthetic2v2Trajectory, scenarioCatalog[0], 0, 'team-a-0');

    assert.equal(result.windowSteps, 5);
    assert.ok(['improved', 'worsened', 'unchanged', 'unknown'].includes(result.isolationTrend));
    assert.equal(typeof result.damageDealtDelta, 'number');
    assert.equal(typeof result.damageTakenDelta, 'number');
    assert.equal(typeof result.teammateResponseTriggered, 'boolean');
    assert.ok(Array.isArray(result.notes));
});

test('computeFiveStepMetricDirection does not crash on support-like scenarios', () => {
    const trajectory = JSON.parse(JSON.stringify(synthetic2v2Trajectory));
    trajectory.steps[0].players[0].measurements.nearestAllyDistance = 6;
    trajectory.steps[1].players[0].measurements.nearestAllyDistance = 3;

    const result = computeFiveStepMetricDirection(trajectory, scenarioCatalog[1], 0, 'team-a-0');
    assert.equal(result.windowSteps, 5);
});
