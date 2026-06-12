import assert from 'node:assert/strict';
import { test } from 'node:test';
import Bot from '../bot';
import {
    BOT_POLICY_TYPES,
    formatBotPolicy,
    requiresLinearIntentModel,
    setAllBotPolicies
} from '../../../pages/simulation/botPolicyConfig';

test('none policy returns no movement, rotation, or fire action', () => {
    const bot = new Bot(1, 'team-a') as any;
    bot.setPolicyMode(BOT_POLICY_TYPES.none);
    bot.updateNetwork = () => assert.fail('none policy must not update the genome network');
    bot.calculateWeights = () => assert.fail('none policy must not calculate genome weights');

    assert.deepEqual(bot.update({}), {
        dx: 0,
        dy: 0,
        dh: 0,
        ds: false
    });
    assert.equal(bot.lastDecision, null);
    assert.equal(bot.lastTrajectoryAction, null);
});

test('none policy config is displayable and does not require the linear intent model', () => {
    const config = setAllBotPolicies(BOT_POLICY_TYPES.none);

    assert.equal(formatBotPolicy(BOT_POLICY_TYPES.none), 'None');
    assert.equal(requiresLinearIntentModel(config), false);
    assert.deepEqual(Object.values(config), ['none', 'none', 'none', 'none']);
});
