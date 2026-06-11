import assert from 'node:assert/strict';
import { test } from 'node:test';
import { computeActionAlignment } from '../actionAlignment';
import { scenarioCatalog } from '../scenarioCatalog';

function createStepFrame() {
    return {
        players: [
            {
                actorId: 'team-a-0',
                actorTeamId: 'team-a',
                state: {
                    positionX: 0,
                    positionY: 0,
                    hp: 100,
                    alive: true
                },
                action: {
                    moveX: 1,
                    moveY: 0,
                    aimX: 1,
                    aimY: 0,
                    fire: 1
                },
                reason: {
                    source: 'policy',
                    label: 'support',
                    evidence: {}
                },
                measurements: {
                    canFire: true,
                    didFire: true,
                    damageDealt: 0,
                    damageTaken: 0
                }
            },
            {
                actorId: 'team-b-0',
                actorTeamId: 'team-b',
                state: {
                    positionX: 10,
                    positionY: 0,
                    hp: 100,
                    alive: true
                }
            },
            {
                actorId: 'team-a-1',
                actorTeamId: 'team-a',
                state: {
                    positionX: 2,
                    positionY: 0,
                    hp: 100,
                    alive: true
                }
            }
        ]
    };
}

test('computeActionAlignment scores movement and aim toward the target highly', () => {
    const stepFrame = createStepFrame();
    const result = computeActionAlignment(stepFrame.players[0], scenarioCatalog[1], stepFrame);

    assert.ok((result.moveAlignment ?? 0) > 0.9);
    assert.ok((result.aimAlignment ?? 0) > 0.9);
    assert.equal(result.reasonMatch, true);
});

test('computeActionAlignment scores movement away from the target negatively', () => {
    const stepFrame = createStepFrame();
    stepFrame.players[0].action.moveX = -1;

    const result = computeActionAlignment(stepFrame.players[0], scenarioCatalog[0], stepFrame);

    assert.ok((result.moveAlignment ?? 0) < -0.9);
});

test('computeActionAlignment respects fire intent and availability', () => {
    const stepFrame = createStepFrame();
    const scenario = scenarioCatalog[1];

    const fireResult = computeActionAlignment(stepFrame.players[0], scenario, stepFrame);
    assert.equal(fireResult.fireMatch, true);

    const blocked = JSON.parse(JSON.stringify(stepFrame));
    blocked.players[0].measurements.canFire = false;
    blocked.players[0].measurements.didFire = false;
    blocked.players[0].action.fire = 1;

    const blockedResult = computeActionAlignment(blocked.players[0], scenario, blocked);
    assert.equal(blockedResult.fireMatch, true);
    assert.ok(blockedResult.details.notes.includes('fire was intended but weapon was unavailable'));
});

test('computeActionAlignment passes reason labels for support-like intent', () => {
    const stepFrame = createStepFrame();
    const result = computeActionAlignment(stepFrame.players[0], scenarioCatalog[1], stepFrame);
    assert.equal(result.reasonMatch, true);
});
