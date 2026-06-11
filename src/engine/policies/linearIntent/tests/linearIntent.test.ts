import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { test } from 'node:test';
import { MAX_HP } from '../../../simulation/combatState';
import { mapLinearIntentToAction } from '../linearIntentActionMapper';
import { extractLinearIntentFeatureVector, getLinearIntentFeatureNames } from '../linearIntentFeatures';
import {
    loadLinearIntentModelFromUrl,
    predictLinearIntent,
    softmax,
    validateLinearIntentModel
} from '../linearIntentModel';
import {
    getDefaultLinearIntentModelPath,
    loadLinearIntentModelFromFile
} from '../linearIntentModelNode';
import { buildLinearIntentDecision, decideLinearIntent } from '../linearIntentPolicy';
import { getLinearIntentDemoScenarios, runLinearIntentScenarioDemo } from '../linearIntentScenarioDemo';
import { LINEAR_INTENT_FEATURE_NAMES, LINEAR_INTENT_MODEL_URL } from '../linearIntentTypes';

function loadJson(filePath: string): unknown {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

test('feature order is locked and matches the canonical list', () => {
    assert.deepEqual(getLinearIntentFeatureNames(), LINEAR_INTENT_FEATURE_NAMES);
});

test('extractLinearIntentFeatureVector returns hp, canFire, and distance features only', () => {
    const vector = extractLinearIntentFeatureVector({
        gridWidth: 100,
        gridHeight: 80,
        self: { hp: 75, x: 10, y: 10, weaponCooldownSteps: 0 },
        ally: { hp: 50, x: 10, y: 30 },
        enemy0: { hp: 90, x: 40, y: 10 },
        enemy1: { hp: 20, x: 90, y: 60 }
    });

    assert.equal(vector.length, 8);
    assert.equal(vector[0], 0.75);
    assert.equal(vector[1], 1);
    assert.equal(vector[2], 0.5);
    assert.equal(vector[3], 0.2);
    assert.equal(vector[4], 0.9);
    assert.equal(vector[5], 0.3);
    assert.equal(vector[6], 0.2);
    assert.equal(vector[7], 1);
    assert.equal(MAX_HP, 100);
});

test('validateLinearIntentModel accepts the exported artifact', () => {
    const model = loadLinearIntentModelFromFile(getDefaultLinearIntentModelPath());
    assert.equal(model.schemaVersion, 'linear-intent-model-v0.1');
    assert.deepEqual(model.input.featureNames, LINEAR_INTENT_FEATURE_NAMES);
});

test('validateLinearIntentModel rejects incorrect feature order and invalid shapes', () => {
    const model = loadLinearIntentModelFromFile(getDefaultLinearIntentModelPath());

    assert.throws(() => validateLinearIntentModel({
        ...model,
        input: {
            ...model.input,
            featureNames: [...model.input.featureNames].reverse()
        }
    }), /feature order/);

    assert.throws(() => validateLinearIntentModel({
        ...model,
        weights: model.weights.slice(0, 3)
    }), /weights must have 4 rows/);

    assert.throws(() => validateLinearIntentModel({
        ...model,
        weights: model.weights.map((row, rowIndex) => row.map((value, featureIndex) => rowIndex === 0 && featureIndex === 0 ? Number.NaN : value))
    }), /weights\[0\] must have 8 finite numbers/);
});

test('softmax remains numerically stable', () => {
    const probabilities = softmax([1000, 999, 998]);
    assert.ok(probabilities[0] > probabilities[1]);
    assert.ok(Math.abs(probabilities.reduce((total, value) => total + value, 0) - 1) < 1e-12);
});

test('predictLinearIntent matches the exported parity cases', () => {
    const model = loadLinearIntentModelFromFile(getDefaultLinearIntentModelPath());
    const casesPath = path.resolve(process.cwd(), 'experiment/linear_intent_inference_cases.json');
    const cases = loadJson(casesPath) as Array<{
        name: string;
        featureVector: number[];
        expectedIntent: string;
        expectedOutput: number;
        expectedScores: number[];
        expectedProbabilities: number[];
        predictedOutput: number;
        predictedIntent: string;
    }>;

    cases.forEach((testCase) => {
        const prediction = predictLinearIntent(model, testCase.featureVector);
        assert.equal(prediction.intentIndex, testCase.predictedOutput);
        assert.equal(prediction.intent, testCase.predictedIntent);
        assert.equal(testCase.expectedIntent, model.output.labels[testCase.expectedOutput]);
        prediction.scores.forEach((score, index) => {
            assert.ok(Math.abs(score - testCase.expectedScores[index]) < 1e-9, `${testCase.name} score[${index}]`);
        });
        prediction.probabilities.forEach((probability, index) => {
            assert.ok(Math.abs(probability - testCase.expectedProbabilities[index]) < 1e-9, `${testCase.name} probability[${index}]`);
        });
        assert.ok(Math.abs(prediction.probabilities.reduce((total, value) => total + value, 0) - 1) < 1e-12);
    });
});

test('buildLinearIntentDecision returns a replay-friendly decision reason', () => {
    const model = loadLinearIntentModelFromFile(getDefaultLinearIntentModelPath());
    const decision = buildLinearIntentDecision(model, {
        gridWidth: 100,
        gridHeight: 100,
        self: { hp: 20, x: 10, y: 10, weaponCooldownSteps: 3 },
        ally: { hp: 80, x: 20, y: 20 },
        enemy0: { hp: 90, x: 50, y: 50 },
        enemy1: { hp: 90, x: 60, y: 60 }
    });

    assert.equal(decision.reason.source, 'linear_intent_model');
    assert.equal(decision.reason.label, decision.intent);
    assert.deepEqual(decision.reason.evidence.featureNames, LINEAR_INTENT_FEATURE_NAMES);
    assert.equal(decision.featureVector[1], 0);
});

test('loadLinearIntentModelFromUrl loads and validates the browser asset', async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (url: RequestInfo | URL) => {
        assert.equal(String(url), LINEAR_INTENT_MODEL_URL);
        const body = fs.readFileSync(path.resolve(process.cwd(), 'public/models/linear_intent_model.json'), 'utf8');
        return new Response(body, { status: 200, headers: { 'content-type': 'application/json' } });
    };

    try {
        const model = await loadLinearIntentModelFromUrl();
        assert.equal(model.schemaVersion, 'linear-intent-model-v0.1');
    } finally {
        globalThis.fetch = originalFetch;
    }
});

test('mapLinearIntentToAction follows the v0 action mapping rules', () => {
    const attack = mapLinearIntentToAction('attack_nearest_enemy', {
        gridWidth: 20,
        gridHeight: 20,
        self: { hp: 90, x: 5, y: 5, weaponCooldownSteps: 0 },
        ally: { hp: 90, x: 8, y: 5 },
        enemy0: { hp: 90, x: 9, y: 5 },
        enemy1: { hp: 90, x: 17, y: 17 }
    });
    assert.equal(attack.fire, 1);
    assert.equal(attack.moveX, 1);
    assert.equal(attack.moveY, 0);

    const support = mapLinearIntentToAction('support_teammate_under_pressure', {
        gridWidth: 20,
        gridHeight: 20,
        self: { hp: 90, x: 4, y: 5, weaponCooldownSteps: 0 },
        ally: { hp: 25, x: 10, y: 5 },
        enemy0: { hp: 90, x: 12, y: 5 },
        enemy1: { hp: 90, x: 16, y: 12 }
    });
    assert.equal(support.fire, 1);
    assert.equal(support.moveX, 1);
    assert.equal(support.moveY, 0);
    assert.equal(support.aimX, 1);
    assert.equal(support.aimY, 0);

    const reduceIsolation = mapLinearIntentToAction('reduce_isolation', {
        gridWidth: 20,
        gridHeight: 20,
        self: { hp: 90, x: 2, y: 2, weaponCooldownSteps: 0 },
        ally: { hp: 90, x: 15, y: 15 },
        enemy0: { hp: 90, x: 17, y: 3 },
        enemy1: { hp: 90, x: 14, y: 18 }
    });
    assert.equal(reduceIsolation.fire, 0);
    assert.equal(reduceIsolation.moveX > 0, true);
    assert.equal(reduceIsolation.moveY > 0, true);

    const retreat = mapLinearIntentToAction('retreat_when_low_hp', {
        gridWidth: 20,
        gridHeight: 20,
        self: { hp: 20, x: 5, y: 5, weaponCooldownSteps: 0 },
        ally: { hp: 90, x: 9, y: 7 },
        enemy0: { hp: 90, x: 7, y: 5 },
        enemy1: { hp: 90, x: 16, y: 16 }
    });
    assert.equal(retreat.fire, 1);
    assert.ok(retreat.moveX > 0);
    assert.ok(retreat.moveY > 0);
});

test('decideLinearIntent and the scenario demo produce the expected intents', () => {
    const model = loadLinearIntentModelFromFile(getDefaultLinearIntentModelPath());
    const scenarios = getLinearIntentDemoScenarios();
    const demoResults = runLinearIntentScenarioDemo(model);

    assert.equal(scenarios.length, 4);
    assert.equal(demoResults.length, 4);
    demoResults.forEach((result) => {
        assert.equal(result.passed, true);
        const expectedScenario = scenarios.find((scenario) => scenario.scenarioId === result.scenarioId);
        assert.ok(expectedScenario);
        assert.equal(result.expectedIntent, expectedScenario.expectedIntent);
        assert.equal(result.predictedIntent, expectedScenario.expectedIntent);
    });

    const directDecision = decideLinearIntent(model, scenarios[0].state);
    assert.equal(directDecision.intent, 'attack_nearest_enemy');
});
