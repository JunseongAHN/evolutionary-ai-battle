import {
    LinearIntentModelJson,
    LinearIntentScenarioDefinition,
    LinearIntentScenarioResult
} from './linearIntentTypes';
import { buildLinearIntentDecision } from './linearIntentPolicy';
import { mapLinearIntentToAction } from './linearIntentActionMapper';

export function getLinearIntentDemoScenarios(): LinearIntentScenarioDefinition[] {
    return [
        {
            scenarioId: 'direct_enemy_contact',
            expectedIntent: 'attack_nearest_enemy',
            state: {
                scenarioId: 'direct_enemy_contact',
                gridWidth: 20,
                gridHeight: 20,
                self: { hp: 90, x: 5, y: 5, weaponCooldownSteps: 0 },
                ally: { hp: 90, x: 8, y: 5 },
                enemy0: { hp: 90, x: 9, y: 5 },
                enemy1: { hp: 90, x: 17, y: 17 }
            }
        },
        {
            scenarioId: 'teammate_under_pressure',
            expectedIntent: 'support_teammate_under_pressure',
            state: {
                scenarioId: 'teammate_under_pressure',
                gridWidth: 20,
                gridHeight: 20,
                self: { hp: 90, x: 4, y: 5, weaponCooldownSteps: 0 },
                ally: { hp: 25, x: 10, y: 5 },
                enemy0: { hp: 90, x: 12, y: 5 },
                enemy1: { hp: 90, x: 16, y: 12 }
            }
        },
        {
            scenarioId: 'isolated_teammate',
            expectedIntent: 'reduce_isolation',
            state: {
                scenarioId: 'isolated_teammate',
                gridWidth: 20,
                gridHeight: 20,
                self: { hp: 90, x: 2, y: 2, weaponCooldownSteps: 0 },
                ally: { hp: 90, x: 15, y: 15 },
                enemy0: { hp: 90, x: 17, y: 3 },
                enemy1: { hp: 90, x: 14, y: 18 }
            }
        },
        {
            scenarioId: 'self_low_hp',
            expectedIntent: 'retreat_when_low_hp',
            state: {
                scenarioId: 'self_low_hp',
                gridWidth: 20,
                gridHeight: 20,
                self: { hp: 20, x: 5, y: 5, weaponCooldownSteps: 0 },
                ally: { hp: 90, x: 9, y: 7 },
                enemy0: { hp: 90, x: 7, y: 5 },
                enemy1: { hp: 90, x: 16, y: 16 }
            }
        }
    ];
}

export function runLinearIntentScenarioDemo(model: LinearIntentModelJson): LinearIntentScenarioResult[] {
    return getLinearIntentDemoScenarios().map((scenario) => {
        const decision = buildLinearIntentDecision(model, scenario.state);
        const action = mapLinearIntentToAction(decision.intent, scenario.state);
        return {
            scenarioId: scenario.scenarioId,
            expectedIntent: scenario.expectedIntent,
            predictedIntent: decision.intent,
            passed: decision.intent === scenario.expectedIntent,
            featureVector: decision.featureVector,
            scores: decision.scores,
            probabilities: decision.probabilities,
            action,
            reason: decision.reason,
            intentIndex: decision.intentIndex
        };
    });
}
