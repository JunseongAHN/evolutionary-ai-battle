import assert from 'node:assert/strict';
import { test } from 'node:test';
import { synthetic2v2Trajectory } from '../fixtures/synthetic2v2Trajectory';
import { validateReplayableTrajectory } from '../trajectoryReplay';

function assertValidationFails(raw: unknown, expectedMessageFragment: string): void {
    const errors = validateReplayableTrajectory(raw);
    assert.ok(errors.length > 0, 'expected validation to fail');
    const combined = errors.join('; ');
    assert.match(combined, new RegExp(expectedMessageFragment.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));
}

test('validateReplayableTrajectory accepts the synthetic 2v2 trajectory fixture', () => {
    assert.deepEqual(validateReplayableTrajectory(synthetic2v2Trajectory), []);
});

test('validateReplayableTrajectory rejects trajectories missing replay-required fields', () => {
    assertValidationFails({ initialState: {}, steps: [] }, 'schemaVersion is required');
    assertValidationFails({ schemaVersion: '0.1.0', steps: [] }, 'initialState is required');
    assertValidationFails({ schemaVersion: '0.1.0', initialState: {} }, 'steps must be an array');
    assertValidationFails({ schemaVersion: '0.1.0', initialState: {}, steps: [{}] }, 'steps[0].players must be an array');
});

test('validateReplayableTrajectory rejects malformed player records', () => {
    assertValidationFails({
        schemaVersion: '0.1.0',
        initialState: {},
        steps: [{
            step: 0,
            players: [{}]
        }]
    }, 'steps[0].players[0].actorId is required');

    assertValidationFails({
        schemaVersion: '0.1.0',
        initialState: {},
        steps: [{
            step: 0,
            players: [{
                actorId: 'team-a-0'
            }]
        }]
    }, 'steps[0].players[0].actorTeamId is required');

    assertValidationFails({
        schemaVersion: '0.1.0',
        initialState: {},
        steps: [{
            step: 0,
            players: [{
                actorId: 'team-a-0',
                actorTeamId: 'team-a'
            }]
        }]
    }, 'steps[0].players[0].state is required');
});

test('validateReplayableTrajectory rejects malformed replay state, action, reason, and measurements fields', () => {
    assertValidationFails({
        schemaVersion: '0.1.0',
        initialState: {},
        steps: [{
            step: 0,
            players: [{
                actorId: 'team-a-0',
                actorTeamId: 'team-a',
                state: {
                    positionY: 0,
                    hp: 100,
                    alive: true
                },
                action: {
                    moveX: 0,
                    moveY: 0,
                    aimX: 0,
                    aimY: 0,
                    fire: 0
                },
                reason: {
                    source: 'policy',
                    label: 'advance',
                    evidence: {}
                },
                measurements: {}
            }]
        }]
    }, 'steps[0].players[0].state.positionX is required');

    assertValidationFails({
        schemaVersion: '0.1.0',
        initialState: {},
        steps: [{
            step: 0,
            players: [{
                actorId: 'team-a-0',
                actorTeamId: 'team-a',
                state: {
                    positionX: 0,
                    positionY: 0,
                    hp: 100,
                    alive: true
                },
                action: {},
                reason: {
                    source: 'policy',
                    label: 'advance',
                    evidence: {}
                },
                measurements: {}
            }]
        }]
    }, 'steps[0].players[0].action.moveX is required');

    assertValidationFails({
        schemaVersion: '0.1.0',
        initialState: {},
        steps: [{
            step: 0,
            players: [{
                actorId: 'team-a-0',
                actorTeamId: 'team-a',
                state: {
                    positionX: 0,
                    positionY: 0,
                    hp: 100,
                    alive: true
                },
                action: {
                    moveX: 0,
                    moveY: 0,
                    aimX: 0,
                    aimY: 0,
                    fire: 0
                },
                reason: {
                    label: 'advance',
                    evidence: {}
                },
                measurements: {}
            }]
        }]
    }, 'steps[0].players[0].reason.source is required');

    assertValidationFails({
        schemaVersion: '0.1.0',
        initialState: {},
        steps: [{
            step: 0,
            players: [{
                actorId: 'team-a-0',
                actorTeamId: 'team-a',
                state: {
                    positionX: 0,
                    positionY: 0,
                    hp: 100,
                    alive: true
                },
                action: {
                    moveX: 0,
                    moveY: 0,
                    aimX: 0,
                    aimY: 0,
                    fire: 0
                },
                reason: {
                    source: 'policy',
                    label: 'advance',
                    evidence: {}
                }
            }]
        }]
    }, 'steps[0].players[0].measurements is required');
});
