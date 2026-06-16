import {
    LINEAR_INTENT_FEATURE_NAMES,
    LinearIntentDecision,
    LinearIntentFeatureInput,
    LinearIntentAction,
    LinearIntentLabel,
    LinearIntentModelJson
} from './linearIntentTypes';
import { buildLinearIntentFeatureSnapshot } from './linearIntentFeatures';
import { predictLinearIntent } from './linearIntentModel';
import { mapLinearIntentToAction } from './linearIntentActionMapper';

export function buildLinearIntentDecision(
    model: LinearIntentModelJson,
    input: LinearIntentFeatureInput
): LinearIntentDecision {
    const { featureVector, rawFeatures } = buildLinearIntentFeatureSnapshot(input);
    const prediction = predictLinearIntent(model, featureVector);

    return {
        intentIndex: prediction.intentIndex,
        intent: prediction.intent,
        scores: prediction.scores,
        probabilities: prediction.probabilities,
        featureNames: LINEAR_INTENT_FEATURE_NAMES,
        featureVector,
        reason: {
            source: 'linear_intent_model',
            label: prediction.intent,
            evidence: {
                schemaVersion: model.schemaVersion,
                intentIndex: prediction.intentIndex,
                intent: prediction.intent,
                scores: prediction.scores,
                probabilities: prediction.probabilities,
                featureNames: LINEAR_INTENT_FEATURE_NAMES,
                featureVector,
                rawFeatures
            }
        }
    };
}

export const decideLinearIntent = buildLinearIntentDecision;

export function decideLinearIntentAction(
    model: LinearIntentModelJson,
    input: LinearIntentFeatureInput
): LinearIntentDecision & { action: LinearIntentAction } {
    const decision = buildLinearIntentDecision(model, input);
    return {
        ...decision,
        action: mapLinearIntentToAction(decision.intent, input)
    };
}

export function getLinearIntentLabelFromIndex(model: LinearIntentModelJson, intentIndex: number): LinearIntentLabel {
    const intent = model.output.labels[intentIndex];
    if (!intent) {
        throw new Error(`Invalid linear intent index: ${intentIndex}`);
    }
    return intent;
}
