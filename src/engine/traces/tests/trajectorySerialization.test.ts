import assert from 'node:assert/strict';
import { test } from 'node:test';
import { synthetic2v2Trajectory } from '../fixtures/synthetic2v2Trajectory';
import { parseTrajectoryJsonString, trajectoryToJsonString } from '../trajectorySerialization';

test('trajectoryToJsonString returns readable JSON', () => {
    const json = trajectoryToJsonString(synthetic2v2Trajectory);

    assert.match(json, /\n  "trajectoryId": "synthetic-2v2-001",\n/);
    assert.match(json, /\n  "steps": \[\n/);
    assert.doesNotThrow(() => JSON.parse(json));
});

test('JSON round trip preserves trajectory identity and steps length', () => {
    const json = trajectoryToJsonString(synthetic2v2Trajectory);
    const parsed = JSON.parse(json);

    assert.equal(parsed.trajectoryId, synthetic2v2Trajectory.trajectoryId);
    assert.equal(parsed.steps.length, synthetic2v2Trajectory.steps.length);
});

test('parseTrajectoryJsonString accepts the synthetic 2v2 trajectory fixture', () => {
    const json = trajectoryToJsonString(synthetic2v2Trajectory);
    const parsed = parseTrajectoryJsonString(json);

    assert.equal(parsed.trajectoryId, synthetic2v2Trajectory.trajectoryId);
    assert.equal(parsed.steps.length, synthetic2v2Trajectory.steps.length);
});

test('parseTrajectoryJsonString rejects invalid JSON with a clear error', () => {
    assert.throws(() => parseTrajectoryJsonString('{'), /Failed to parse trajectory JSON:/);
});

test('parseTrajectoryJsonString rejects JSON missing schemaVersion', () => {
    const badJson = JSON.stringify({
        ...synthetic2v2Trajectory,
        schemaVersion: ''
    });

    assert.throws(() => parseTrajectoryJsonString(badJson), /Invalid trajectory JSON: .*schemaVersion is required/);
});

test('parseTrajectoryJsonString rejects JSON missing steps', () => {
    const { steps, ...rest } = synthetic2v2Trajectory;
    const badJson = JSON.stringify(rest);

    assert.throws(() => parseTrajectoryJsonString(badJson), /Invalid trajectory JSON: .*steps must be an array/);
});

test('parseTrajectoryJsonString rejects a step without players', () => {
    const badJson = JSON.stringify({
        ...synthetic2v2Trajectory,
        steps: [
            {
                ...synthetic2v2Trajectory.steps[0],
                players: undefined
            }
        ]
    });

    assert.throws(() => parseTrajectoryJsonString(badJson), /Invalid trajectory JSON: .*steps\[0\]\.players must be an array/);
});
