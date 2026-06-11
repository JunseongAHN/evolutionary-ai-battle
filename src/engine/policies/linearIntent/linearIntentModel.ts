import {
    LINEAR_INTENT_FEATURE_NAMES,
    LINEAR_INTENT_LABELS,
    LINEAR_INTENT_MODEL_URL,
    LINEAR_INTENT_SCHEMA_VERSION,
    LinearIntentLabel,
    LinearIntentModelJson,
    LinearIntentPrediction
} from './linearIntentTypes';

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null;
}

function isFiniteNumber(value: unknown): value is number {
    return typeof value === 'number' && Number.isFinite(value);
}

function isNumberArray(value: unknown): value is number[] {
    return Array.isArray(value) && value.every((item) => isFiniteNumber(item));
}

function arraysEqual<T>(left: readonly T[], right: readonly T[]): boolean {
    return left.length === right.length && left.every((item, index) => item === right[index]);
}

function validateWeightMatrix(weights: unknown, expectedRows: number, expectedCols: number, errors: string[]): weights is number[][] {
    if (!Array.isArray(weights) || weights.length !== expectedRows) {
        errors.push(`weights must have ${expectedRows} rows`);
        return false;
    }

    weights.forEach((row, rowIndex) => {
        if (!isNumberArray(row) || row.length !== expectedCols) {
            errors.push(`weights[${rowIndex}] must have ${expectedCols} finite numbers`);
        }
    });

    return errors.length === 0;
}

function collectModelValidationErrors(raw: unknown): string[] {
    if (!isRecord(raw)) {
        return ['model must be an object'];
    }

    const errors: string[] = [];
    if (raw.schemaVersion !== LINEAR_INTENT_SCHEMA_VERSION) {
        errors.push(`schemaVersion must be ${LINEAR_INTENT_SCHEMA_VERSION}`);
    }
    if (raw.modelType !== 'softmax_linear_classifier') {
        errors.push('modelType must be softmax_linear_classifier');
    }

    if (!isRecord(raw.input)) {
        errors.push('input is required');
    } else {
        const input = raw.input as Record<string, unknown>;
        if (!Array.isArray(input.featureNames) || !input.featureNames.every((value) => typeof value === 'string')) {
            errors.push('input.featureNames is required');
        } else if (!arraysEqual(input.featureNames, LINEAR_INTENT_FEATURE_NAMES)) {
            errors.push('input.featureNames must match the locked feature order');
        }
        if (input.featureCount !== LINEAR_INTENT_FEATURE_NAMES.length) {
            errors.push(`input.featureCount must be ${LINEAR_INTENT_FEATURE_NAMES.length}`);
        }
        if (input.featureOrderLocked !== true) {
            errors.push('input.featureOrderLocked must be true');
        }
    }

    if (!isRecord(raw.output)) {
        errors.push('output is required');
    } else {
        const output = raw.output as Record<string, unknown>;
        if (!Array.isArray(output.labels) || !output.labels.every((value) => typeof value === 'string')) {
            errors.push('output.labels is required');
        } else if (!arraysEqual(output.labels, LINEAR_INTENT_LABELS)) {
            errors.push('output.labels must match the canonical label order');
        }
        if (!isRecord(output.labelToIndex)) {
            errors.push('output.labelToIndex is required');
        } else {
            const labelToIndex = output.labelToIndex as Record<string, unknown>;
            LINEAR_INTENT_LABELS.forEach((label, index) => {
                if (labelToIndex[label] !== index) {
                    errors.push(`output.labelToIndex.${label} must be ${index}`);
                }
            });
        }
        if (!isRecord(output.indexToLabel)) {
            errors.push('output.indexToLabel is required');
        } else {
            const indexToLabel = output.indexToLabel as Record<string, unknown>;
            LINEAR_INTENT_LABELS.forEach((label, index) => {
                if (indexToLabel[String(index)] !== label) {
                    errors.push(`output.indexToLabel.${index} must be ${label}`);
                }
            });
        }
    }

    if (!validateWeightMatrix(raw.weights, LINEAR_INTENT_LABELS.length, LINEAR_INTENT_FEATURE_NAMES.length, errors)) {
        // errors already collected
    }
    if (!isNumberArray(raw.bias) || raw.bias.length !== LINEAR_INTENT_LABELS.length) {
        errors.push(`bias must have ${LINEAR_INTENT_LABELS.length} finite numbers`);
    }

    if (!isRecord(raw.training)) {
        errors.push('training is required');
    }

    return errors;
}

export function validateLinearIntentModel(raw: unknown): LinearIntentModelJson {
    const errors = collectModelValidationErrors(raw);
    if (errors.length) {
        throw new Error(`Invalid linear intent model: ${errors.join('; ')}`);
    }
    return raw as LinearIntentModelJson;
}

export function parseLinearIntentModelJsonString(jsonString: string): LinearIntentModelJson {
    let parsed: unknown;
    try {
        parsed = JSON.parse(jsonString);
    } catch (error) {
        const message = error instanceof Error ? error.message : 'Unknown parse error';
        throw new Error(`Failed to parse linear intent model JSON: ${message}`);
    }
    return validateLinearIntentModel(parsed);
}

export async function loadLinearIntentModelFromUrl(
    url: string = LINEAR_INTENT_MODEL_URL
): Promise<LinearIntentModelJson> {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Failed to load linear intent model from ${url}: ${response.status} ${response.statusText}`);
    }

    const raw = await response.json();
    return validateLinearIntentModel(raw);
}

export function softmax(values: number[]): number[] {
    if (!values.length) {
        return [];
    }
    const maxValue = Math.max(...values);
    const exps = values.map((value) => Math.exp(value - maxValue));
    const sum = exps.reduce((total, value) => total + value, 0);
    return exps.map((value) => value / sum);
}

export function predictLinearIntent(model: LinearIntentModelJson, featureVector: number[]): LinearIntentPrediction {
    if (featureVector.length !== LINEAR_INTENT_FEATURE_NAMES.length) {
        throw new Error(`featureVector must have ${LINEAR_INTENT_FEATURE_NAMES.length} values`);
    }

    const scores = model.weights.map((row, rowIndex) => {
        const score = row.reduce((total, weight, featureIndex) => total + (weight * featureVector[featureIndex]), model.bias[rowIndex]);
        if (!Number.isFinite(score)) {
            throw new Error('Linear intent score calculation produced a non-finite value');
        }
        return score;
    });

    let intentIndex = 0;
    for (let index = 1; index < scores.length; index += 1) {
        if (scores[index] > scores[intentIndex]) {
            intentIndex = index;
        }
    }

    const probabilities = softmax(scores);
    return {
        intentIndex,
        intent: model.output.labels[intentIndex] as LinearIntentLabel,
        scores,
        probabilities
    };
}
