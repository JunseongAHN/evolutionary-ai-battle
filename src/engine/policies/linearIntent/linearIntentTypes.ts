import { DecisionReason } from '../../traces/trace';

export const LINEAR_INTENT_SCHEMA_VERSION = 'linear-intent-model-v0.1';
export const LINEAR_INTENT_MODEL_URL = '/models/linear_intent_model.json';

export const LINEAR_INTENT_FEATURE_NAMES = [
    'selfHpNorm',
    'canFire',
    'allyHpNorm',
    'allyDistanceNorm',
    'enemy0HpNorm',
    'enemy0DistanceNorm',
    'enemy1HpNorm',
    'enemy1DistanceNorm'
] as const;

export const LINEAR_INTENT_LABELS = [
    'attack_nearest_enemy',
    'support_teammate_under_pressure',
    'reduce_isolation',
    'retreat_when_low_hp'
] as const;

export type LinearIntentLabel = typeof LINEAR_INTENT_LABELS[number];

export interface LinearIntentTrainingMetadata {
    trainDataset: string;
    evalDataset: string;
    epochs: number;
    learningRate: number;
    l2: number;
    seed: number;
    trainSize: number;
    evalSize: number;
    trainAccuracy: number;
    evalAccuracy: number;
}

export interface LinearIntentModelInputSchema {
    featureNames: typeof LINEAR_INTENT_FEATURE_NAMES;
    featureCount: number;
    featureOrderLocked: boolean;
}

export interface LinearIntentModelOutputSchema {
    labels: typeof LINEAR_INTENT_LABELS;
    labelToIndex: Record<LinearIntentLabel, number>;
    indexToLabel: Record<string, LinearIntentLabel>;
}

export interface LinearIntentModelJson {
    schemaVersion: string;
    modelType: 'softmax_linear_classifier';
    input: LinearIntentModelInputSchema;
    output: LinearIntentModelOutputSchema;
    weights: number[][];
    bias: number[];
    training: LinearIntentTrainingMetadata;
}

export interface LinearIntentEntityState {
    hp: number;
    x: number;
    y: number;
    weaponCooldownSteps?: number;
}

export interface LinearIntentFeatureInput {
    gridWidth: number;
    gridHeight: number;
    self: LinearIntentEntityState;
    ally: LinearIntentEntityState;
    enemy0: LinearIntentEntityState;
    enemy1: LinearIntentEntityState;
}

export interface LinearIntentPrediction {
    intentIndex: number;
    intent: LinearIntentLabel;
    scores: number[];
    probabilities: number[];
}

export interface LinearIntentDecision {
    intentIndex: number;
    intent: LinearIntentLabel;
    scores: number[];
    probabilities: number[];
    featureNames: typeof LINEAR_INTENT_FEATURE_NAMES;
    featureVector: number[];
    reason: DecisionReason;
}

export interface LinearIntentAction {
    moveX: number;
    moveY: number;
    aimX: number;
    aimY: number;
    fire: number;
}

export interface LinearIntentScenarioState extends LinearIntentFeatureInput {
    scenarioId: string;
}

export interface LinearIntentScenarioDefinition {
    scenarioId: string;
    expectedIntent: LinearIntentLabel;
    state: LinearIntentScenarioState;
}

export interface LinearIntentScenarioResult {
    scenarioId: string;
    expectedIntent: LinearIntentLabel;
    predictedIntent: LinearIntentLabel;
    passed: boolean;
    featureVector: number[];
    scores: number[];
    probabilities: number[];
    action: LinearIntentAction;
    reason: DecisionReason;
    intentIndex: number;
}
