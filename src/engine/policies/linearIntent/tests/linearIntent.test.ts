import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { test } from 'node:test';

import { linearIntentActionToBattleAction, mapLinearIntentToAction } from '../linearIntentActionMapper';
import { buildLinearIntentFeatureStateForBot } from '../linearIntentBattleAdapter';
import {
    buildLinearIntentFeatureSnapshot,
    extractLinearIntentFeatureVector,
    LINEAR_INTENT_ATTACK_RANGE,
    LINEAR_INTENT_MAX_HP
} from '../linearIntentFeatures';
import { predictLinearIntent } from '../linearIntentModel';
import { getDefaultLinearIntentModelPath, loadLinearIntentModelFromFile } from '../linearIntentModelNode';
import { buildLinearIntentDecision } from '../linearIntentPolicy';
import { LINEAR_INTENT_FEATURE_NAMES } from '../linearIntentTypes';

type CsvRow = Record<string, string>;

function parseCsv(text: string): CsvRow[] {
    const lines = text
        .trim()
        .split(/\r?\n/)
        .filter(Boolean);

    assert.ok(lines.length >= 2, 'CSV must contain header and at least one row');

    const headers = lines[0].split(',').map((value) => value.trim());

    return lines.slice(1).map((line) => {
        const values = line.split(',').map((value) => value.trim());
        const row: CsvRow = {};

        headers.forEach((header, index) => {
            row[header] = values[index] ?? '';
        });

        return row;
    });
}

function findLabelColumn(row: CsvRow): string | null {
    const candidates = ['intent', 'label', 'expectedIntent', 'target', 'y'];
    return candidates.find((candidate) => candidate in row) ?? null;
}

function rowToFeatureVector(row: CsvRow): number[] {
    return LINEAR_INTENT_FEATURE_NAMES.map((featureName) => {
        assert.ok(featureName in row, `CSV row is missing feature column: ${featureName}`);

        const value = Number(row[featureName]);
        assert.ok(Number.isFinite(value), `Feature ${featureName} must be finite. Received: ${row[featureName]}`);

        return value;
    });
}

function assertProbabilities(probabilities: number[]): void {
    probabilities.forEach((probability) => {
        assert.ok(Number.isFinite(probability));
        assert.ok(probability >= 0);
        assert.ok(probability <= 1);
    });

    const total = probabilities.reduce((sum, value) => sum + value, 0);
    assert.ok(Math.abs(total - 1) < 1e-9, `probabilities must sum to 1. Received: ${total}`);
}

function assertLinearIntentAction(action: {
    moveX: number;
    moveY: number;
    aimX: number;
    aimY: number;
    fire: number;
}): void {
    assert.ok(Number.isFinite(action.moveX));
    assert.ok(Number.isFinite(action.moveY));
    assert.ok(Number.isFinite(action.aimX));
    assert.ok(Number.isFinite(action.aimY));
    assert.ok(action.fire === 0 || action.fire === 1);
}

test('linear intent model loads with the expected I/O schema', () => {
    const model = loadLinearIntentModelFromFile(getDefaultLinearIntentModelPath());

    assert.equal(model.schemaVersion, 'linear-intent-model-v0.2');
    assert.deepEqual(model.input.featureNames, LINEAR_INTENT_FEATURE_NAMES);
    assert.equal(model.input.featureNames.length, 7);
    assert.equal(model.input.featureNames.includes('canFire' as never), false);
    assert.equal(model.output.labels.length, 4);
});

test('runtime preprocessing preserves tactical HP, distance, and enemy ordering without canFire', () => {
    const baseState = {
        gridWidth: 1000,
        gridHeight: 500,
        self: { hp: LINEAR_INTENT_MAX_HP, x: 0, y: 0, weaponCooldownSteps: 0 },
        ally: { hp: LINEAR_INTENT_MAX_HP, x: 350, y: 0 },
        enemy0: { hp: LINEAR_INTENT_MAX_HP, x: 690, y: 0 },
        enemy1: { hp: LINEAR_INTENT_MAX_HP, x: 700, y: 0 }
    };

    const farSnapshot = buildLinearIntentFeatureSnapshot(baseState);
    assert.equal(farSnapshot.featureVector[0], 1);
    assert.equal(farSnapshot.featureVector.length, 7);
    assert.notEqual(farSnapshot.featureVector[2], farSnapshot.featureVector[4]);
    assert.ok(farSnapshot.featureVector.every(Number.isFinite));
    assert.equal(LINEAR_INTENT_FEATURE_NAMES.includes('canFire' as never), false);
    assert.equal(farSnapshot.rawFeatures.weaponReady, true);
    assert.equal(farSnapshot.rawFeatures.enemyInRange, false);
    assert.equal(mapLinearIntentToAction('attack_nearest_enemy', baseState).fire, 1);

    const inRangeState = {
        ...baseState,
        enemy0: { ...baseState.enemy0, x: LINEAR_INTENT_ATTACK_RANGE - 1 }
    };
    const inRangeSnapshot = buildLinearIntentFeatureSnapshot(inRangeState);
    assert.equal(inRangeSnapshot.rawFeatures.enemyInRange, true);
    assert.deepEqual(inRangeSnapshot.featureVector.length, farSnapshot.featureVector.length);
    assert.equal(mapLinearIntentToAction('attack_nearest_enemy', inRangeState).fire, 1);

    const missingEnemySnapshot = buildLinearIntentFeatureSnapshot({
        ...baseState,
        enemy1: { hp: 0, x: 0, y: 0, missing: true }
    });
    assert.equal(missingEnemySnapshot.featureVector[5], 0);
    assert.equal(missingEnemySnapshot.featureVector[6], 1);
    assert.ok(Object.values(missingEnemySnapshot.rawFeatures).every((value) => typeof value === 'boolean' || Number.isFinite(value)));

    const actor = { teamId: 'team-b', xPos: 901, yPos: 470, lives: 5 };
    const ally = { teamId: 'team-b', xPos: 961, yPos: 68, lives: 5 };
    const fartherEnemy = { teamId: 'team-a', xPos: 215, yPos: 224, lives: 5 };
    const nearerEnemy = { teamId: 'team-a', xPos: 220, yPos: 391, lives: 5 };
    const state = buildLinearIntentFeatureStateForBot(actor, {
        bots: [actor, ally, fartherEnemy, nearerEnemy],
        weaponCooldownSteps: [0, 0, 0, 0],
        environment: { width: 1000, height: 500 }
    });
    const snapshot = buildLinearIntentFeatureSnapshot(state);

    assert.equal(state.enemy0.x, nearerEnemy.xPos);
    assert.deepEqual(snapshot.featureVector.filter((_, index) => [0, 1, 3, 5].includes(index)), [1, 1, 1, 1]);
    assert.ok(snapshot.featureVector.slice(2).some((value) => value !== 1));
});

test('intent action rotates toward its aim target before firing', () => {
    const aimUpAction = {
        moveX: 0,
        moveY: 0,
        aimX: 0,
        aimY: 1,
        fire: 1
    };

    const turning = linearIntentActionToBattleAction(aimUpAction, 0);
    assert.ok(turning.dh > 0);
    assert.equal(turning.ds, false);

    const aligned = linearIntentActionToBattleAction(aimUpAction, 90);
    assert.equal(aligned.dh, 0);
    assert.equal(aligned.ds, true);

    const wrapAround = linearIntentActionToBattleAction({
        ...aimUpAction,
        aimX: Math.cos(-10 * Math.PI / 180),
        aimY: Math.sin(-10 * Math.PI / 180)
    }, 350);
    assert.ok(Math.abs(wrapAround.dh) < 1e-9);

    const attackWhileTurning = linearIntentActionToBattleAction({
        ...aimUpAction,
        fireWhileAiming: true
    }, 0);
    assert.ok(attackWhileTurning.dh > 0);
    assert.equal(attackWhileTurning.ds, true);
});

test('eval_intent_dataset.csv rows can be preprocessed, predicted, and postprocessed', () => {
    const model = loadLinearIntentModelFromFile(getDefaultLinearIntentModelPath());
    const csvPath = path.resolve(process.cwd(), 'experiment/eval_intent_dataset_v0_2.csv');
    const rows = parseCsv(fs.readFileSync(csvPath, 'utf8'));
    assert.equal('canFire' in rows[0], false);

    let correct = 0;
    let evaluated = 0;

    const preview: Array<{
        row: number;
        expected?: string;
        predicted: string;
        confidence: number;
    }> = [];

    rows.forEach((row, rowIndex) => {
        const featureVector = rowToFeatureVector(row);

        assert.equal(featureVector.length, LINEAR_INTENT_FEATURE_NAMES.length);
        featureVector.forEach((value) => assert.ok(Number.isFinite(value)));

        const prediction = predictLinearIntent(model, featureVector);

        assert.ok(model.output.labels.includes(prediction.intent));
        assert.equal(prediction.intent, model.output.labels[prediction.intentIndex]);
        assertProbabilities(prediction.probabilities);

        const labelColumn = findLabelColumn(row);
        const expectedIntent = labelColumn ? row[labelColumn] : undefined;

        if (expectedIntent) {
            evaluated += 1;
            if (prediction.intent === expectedIntent) {
                correct += 1;
            }
        }

        if (rowIndex < 10) {
            preview.push({
                row: rowIndex,
                expected: expectedIntent,
                predicted: prediction.intent,
                confidence: Math.max(...prediction.probabilities)
            });
        }
    });

    console.table(preview);

    if (evaluated > 0) {
        const accuracy = correct / evaluated;
        console.log(`eval_intent_dataset.csv accuracy: ${correct}/${evaluated} = ${accuracy.toFixed(4)}`);

        // This is smoke/parity checking, not a strict training evaluation.
        assert.ok(accuracy >= 0.9, `Expected smoke accuracy >= 0.9, received ${accuracy}`);
    }
});

test('environment-like states go through preprocess, model predict, and postprocess', () => {
    const model = loadLinearIntentModelFromFile(getDefaultLinearIntentModelPath());

    const scenarios = [
        {
            scenarioId: 'default_2v2_trajectory_like',
            state: {
                gridWidth: 1000,
                gridHeight: 500,
                self: { hp: 5, x: 901, y: 470, weaponCooldownSteps: 0 },
                ally: { hp: 5, x: 961, y: 68 },
                enemy0: { hp: 5, x: 220, y: 391 },
                enemy1: { hp: 5, x: 215, y: 224 }
            }
        },
        {
            scenarioId: 'direct_enemy_contact',
            state: {
                gridWidth: 1000,
                gridHeight: 500,
                self: { hp: 5, x: 100, y: 100, weaponCooldownSteps: 0 },
                ally: { hp: 5, x: 350, y: 100 },
                enemy0: { hp: 5, x: 300, y: 100 },
                enemy1: { hp: 5, x: 900, y: 400 }
            }
        },
        {
            scenarioId: 'teammate_under_pressure',
            state: {
                gridWidth: 1000,
                gridHeight: 500,
                self: { hp: 5, x: 100, y: 100, weaponCooldownSteps: 0 },
                ally: { hp: 1, x: 450, y: 100 },
                enemy0: { hp: 5, x: 500, y: 100 },
                enemy1: { hp: 5, x: 900, y: 400 }
            }
        },
        {
            scenarioId: 'isolated_teammate',
            state: {
                gridWidth: 1000,
                gridHeight: 500,
                self: { hp: 5, x: 50, y: 50, weaponCooldownSteps: 0 },
                ally: { hp: 5, x: 950, y: 450 },
                enemy0: { hp: 5, x: 800, y: 100 },
                enemy1: { hp: 5, x: 850, y: 400 }
            }
        },
        {
            scenarioId: 'self_low_hp',
            state: {
                gridWidth: 1000,
                gridHeight: 500,
                self: { hp: 1, x: 300, y: 250, weaponCooldownSteps: 0 },
                ally: { hp: 5, x: 500, y: 300 },
                enemy0: { hp: 5, x: 400, y: 250 },
                enemy1: { hp: 5, x: 900, y: 400 }
            }
        }
    ];

    const summary = scenarios.map((scenario) => {
        const featureVector = extractLinearIntentFeatureVector(scenario.state);
        const decision = buildLinearIntentDecision(model, scenario.state);
        const action = mapLinearIntentToAction(decision.intent, scenario.state);
        const battleAction = linearIntentActionToBattleAction(action);

        assert.equal(featureVector.length, LINEAR_INTENT_FEATURE_NAMES.length);
        featureVector.forEach((value) => assert.ok(Number.isFinite(value)));

        assert.equal(decision.reason.source, 'linear_intent_model');
        assert.equal(decision.reason.label, decision.intent);
        assert.deepEqual(decision.reason.evidence.featureNames, LINEAR_INTENT_FEATURE_NAMES);
        assert.equal(decision.reason.evidence.schemaVersion, 'linear-intent-model-v0.2');
        assert.equal((decision.reason.evidence.featureVector as number[]).length, 7);
        assert.ok(decision.reason.evidence.rawFeatures);
        assert.equal(typeof (decision.reason.evidence.rawFeatures as any).weaponReady, 'boolean');
        assert.equal(typeof (decision.reason.evidence.rawFeatures as any).enemyInRange, 'boolean');
        assert.ok(model.output.labels.includes(decision.intent));
        assertProbabilities(decision.probabilities);

        assertLinearIntentAction(action);

        assert.ok(Number.isFinite(battleAction.dx));
        assert.ok(Number.isFinite(battleAction.dy));
        assert.ok(Number.isFinite(battleAction.dh));
        assert.equal(typeof battleAction.ds, 'boolean');

        return {
            scenarioId: scenario.scenarioId,
            intent: decision.intent,
            confidence: Math.max(...decision.probabilities),
            featureVector: JSON.stringify(featureVector),
            rawFeatures: JSON.stringify(decision.reason.evidence.rawFeatures),
            action: JSON.stringify(action),
            battleAction: JSON.stringify(battleAction)
        };
    });

    console.table(summary);
});
