import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createReplayStateFromStep, getStepFrame, loadTrajectoryFromObject, validateTrajectory } from '../trajectoryReplay';
import { synthetic2v2Trajectory } from '../fixtures/synthetic2v2Trajectory';

function assertValidationFails(raw: unknown, expectedMessageFragment: string): void {
    const errors = validateTrajectory(raw);
    assert.ok(errors.length > 0, 'expected validation to fail');
    const combined = errors.join('; ');
    assert.match(combined, new RegExp(expectedMessageFragment.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));
}

test('validateTrajectory accepts a valid synthetic 2v2 trajectory', () => {
    const errors = validateTrajectory(synthetic2v2Trajectory);
    assert.deepEqual(errors, []);
});

test('trajectory is JSON-compatible and can be revalidated', () => {
    const roundTrip = JSON.parse(JSON.stringify(synthetic2v2Trajectory));
    assert.doesNotThrow(() => JSON.stringify(synthetic2v2Trajectory));
    const errors = validateTrajectory(roundTrip);
    assert.deepEqual(errors, []);
});

test('getStepFrame returns the requested step frame', () => {
    assert.deepEqual(getStepFrame(synthetic2v2Trajectory, 0)?.step, 0);
    assert.deepEqual(getStepFrame(synthetic2v2Trajectory, 1)?.step, 1);
    assert.equal(getStepFrame(synthetic2v2Trajectory, 99), null);
});

test('createReplayStateFromStep creates drawable replay state', () => {
    const replayState = createReplayStateFromStep(synthetic2v2Trajectory.steps[1]);

    assert.ok(replayState.environment);
    assert.ok(Array.isArray(replayState.players));
    assert.equal(replayState.players.length, 4);

    const firstPlayer = replayState.players[0];
    assert.equal(firstPlayer.id, 'team-a-0');
    assert.equal(firstPlayer.teamId, 'team-a');
    assert.equal(firstPlayer.positionX, -10);
    assert.equal(firstPlayer.positionY, 2);
    assert.equal(firstPlayer.hp, 100);
    assert.equal(firstPlayer.alive, true);
    assert.deepEqual(firstPlayer.lastAction, synthetic2v2Trajectory.steps[1].players[0].action);
    assert.deepEqual(firstPlayer.reason, synthetic2v2Trajectory.steps[1].players[0].reason);
    assert.deepEqual(firstPlayer.measurements, synthetic2v2Trajectory.steps[1].players[0].measurements);
});

test('invalid trajectories fail validation with clear errors', () => {
    assertValidationFails({ schemaVersion: '0.1.0' }, 'initialState is required');
    assertValidationFails({ schemaVersion: '0.1.0', initialState: {} }, 'steps must be an array');
    assertValidationFails({
        schemaVersion: '0.1.0',
        initialState: {},
        steps: [{}]
    }, 'steps[0].players must be an array');

    assertValidationFails({
        schemaVersion: '0.1.0',
        initialState: {},
        steps: [{ players: [{}] }]
    }, 'steps[0].players[0].actorId is required');

    assertValidationFails({
        schemaVersion: '0.1.0',
        initialState: {},
        steps: [{ players: [{ actorId: 'team-a-0', action: {} }] }]
    }, 'steps[0].players[0].state is required');

    assertValidationFails({
        schemaVersion: '0.1.0',
        initialState: {},
        steps: [{ players: [{ actorId: 'team-a-0', state: {} }] }]
    }, 'steps[0].players[0].action is required');

    assertValidationFails({
        schemaVersion: '0.1.0',
        initialState: {},
        steps: [{}]
    }, 'steps[0].players must be an array');
});

test('loadTrajectoryFromObject preserves a valid object and rejects invalid input', () => {
    const loaded = loadTrajectoryFromObject(synthetic2v2Trajectory);
    assert.equal(loaded.trajectoryId, synthetic2v2Trajectory.trajectoryId);

    assert.throws(() => loadTrajectoryFromObject({ schemaVersion: '0.1.0' }), /Invalid trajectory/);
});

test('replay utilities stay state-playback only and do not depend on policy or simulation modules', () => {
    const replayStep = synthetic2v2Trajectory.steps[0];
    const replayState = createReplayStateFromStep(replayStep);
    assert.ok(replayState);
    assert.equal(replayState.players[0].id, 'team-a-0');
    assert.equal(replayState.players[0].lastAction.moveX, 0.2);
    assert.equal(replayState.players[0].reason.label, 'advance');
});
