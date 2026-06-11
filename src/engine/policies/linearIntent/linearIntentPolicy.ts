import {
    LINEAR_INTENT_FEATURE_NAMES,
    LinearIntentDecision,
    LinearIntentFeatureInput,
    LinearIntentLabel,
    LinearIntentModelJson
} from './linearIntentTypes';
import { extractLinearIntentFeatureVector } from './linearIntentFeatures';
import { predictLinearIntent } from './linearIntentModel';

export function buildLinearIntentDecision(
    model: LinearIntentModelJson,
    input: LinearIntentFeatureInput
): LinearIntentDecision {
    const featureVector = extractLinearIntentFeatureVector(input);
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
                scores: prediction.scores,
                probabilities: prediction.probabilities,
                featureNames: LINEAR_INTENT_FEATURE_NAMES,
                featureVector
            }
        }
    };
}

export const decideLinearIntent = buildLinearIntentDecision;

export function getLinearIntentLabelFromIndex(model: LinearIntentModelJson, intentIndex: number): LinearIntentLabel {
    const intent = model.output.labels[intentIndex];
    if (!intent) {
        throw new Error(`Invalid linear intent index: ${intentIndex}`);
    }
    return intent;
}
