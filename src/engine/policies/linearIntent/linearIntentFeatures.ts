import { MAX_HP } from '../../simulation/combatState';
import {
    LINEAR_INTENT_FEATURE_NAMES,
    LinearIntentFeatureInput
} from './linearIntentTypes';

function clamp01(value: number): number {
    if (!Number.isFinite(value)) {
        return 0;
    }
    if (value < 0) return 0;
    if (value > 1) return 1;
    return value;
}

function toNormHp(hp: number): number {
    return clamp01(hp / MAX_HP);
}

function manhattanDistance(a: { x: number; y: number }, b: { x: number; y: number }): number {
    return Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
}

function distanceNorm(
    gridWidth: number,
    gridHeight: number,
    a: { x: number; y: number },
    b: { x: number; y: number }
): number {
    const denominator = Math.max(gridWidth, gridHeight);
    if (!Number.isFinite(denominator) || denominator <= 0) {
        return 0;
    }
    return clamp01(manhattanDistance(a, b) / denominator);
}

export function getLinearIntentFeatureNames(): typeof LINEAR_INTENT_FEATURE_NAMES {
    return LINEAR_INTENT_FEATURE_NAMES;
}

export function extractLinearIntentFeatureVector(input: LinearIntentFeatureInput): number[] {
    const self = { x: input.self.x, y: input.self.y };
    return [
        toNormHp(input.self.hp),
        input.self.weaponCooldownSteps !== undefined && input.self.weaponCooldownSteps <= 0 ? 1 : 0,
        toNormHp(input.ally.hp),
        distanceNorm(input.gridWidth, input.gridHeight, self, input.ally),
        toNormHp(input.enemy0.hp),
        distanceNorm(input.gridWidth, input.gridHeight, self, input.enemy0),
        toNormHp(input.enemy1.hp),
        distanceNorm(input.gridWidth, input.gridHeight, self, input.enemy1)
    ];
}

export function normalizeVector(dx: number, dy: number): { x: number; y: number } {
    const length = Math.hypot(dx, dy);
    if (!Number.isFinite(length) || length === 0) {
        return { x: 0, y: 0 };
    }
    return { x: dx / length, y: dy / length };
}
