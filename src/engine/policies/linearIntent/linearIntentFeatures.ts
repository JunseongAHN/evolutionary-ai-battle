import config from '../../../../config/default.json';
import {
    LINEAR_INTENT_FEATURE_NAMES,
    LinearIntentEntityState,
    LinearIntentFeatureInput,
    LinearIntentRawFeatures
} from './linearIntentTypes';

// Runtime battles use lives as HP. Model features always receive normalized HP.
export const LINEAR_INTENT_MAX_HP = config.startingLives;
// The dataset fire range is 5 cells on a 20-cell-wide grid; runtime cells are 50 px.
export const LINEAR_INTENT_ATTACK_RANGE = config.neuralNetworkSquareSize * 5;

function clamp01(value: number): number {
    if (!Number.isFinite(value)) {
        return 0;
    }
    if (value < 0) return 0;
    if (value > 1) return 1;
    return value;
}

function toNormHp(hp: number): number {
    return clamp01(hp / LINEAR_INTENT_MAX_HP);
}

export function linearIntentDistance(a: { x: number; y: number }, b: { x: number; y: number }): number {
    return Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
}

function arenaDiagonal(input: LinearIntentFeatureInput): number {
    const diagonal = Math.hypot(input.gridWidth, input.gridHeight);
    return Number.isFinite(diagonal) && diagonal > 0 ? diagonal : 1;
}

function distanceNormalizationScale(input: LinearIntentFeatureInput): number {
    // Match the trained dataset contract: Manhattan distance divided by the largest arena dimension.
    const scale = Math.max(input.gridWidth, input.gridHeight);
    return Number.isFinite(scale) && scale > 0 ? scale : 1;
}

function entityDistance(self: LinearIntentEntityState, entity: LinearIntentEntityState): number {
    return entity.missing ? Number.POSITIVE_INFINITY : linearIntentDistance(self, entity);
}

function normalizeEntityDistance(distance: number, scale: number): number {
    return Number.isFinite(distance) ? clamp01(distance / scale) : 1;
}

export function buildLinearIntentFeatureSnapshot(input: LinearIntentFeatureInput): {
    featureVector: number[];
    rawFeatures: LinearIntentRawFeatures;
} {
    const diagonal = arenaDiagonal(input);
    const normalizationScale = distanceNormalizationScale(input);
    const allyDistance = input.ally.missing ? normalizationScale : entityDistance(input.self, input.ally);
    const enemy0Distance = input.enemy0.missing ? normalizationScale : entityDistance(input.self, input.enemy0);
    const enemy1Distance = input.enemy1.missing ? normalizationScale : entityDistance(input.self, input.enemy1);
    const nearestEnemyDistance = Math.min(enemy0Distance, enemy1Distance);
    const weaponReady = input.self.weaponCooldownSteps !== undefined && input.self.weaponCooldownSteps <= 0;
    const enemyInRange = nearestEnemyDistance <= LINEAR_INTENT_ATTACK_RANGE;

    return {
        featureVector: [
            toNormHp(input.self.hp),
            input.ally.missing ? 0 : toNormHp(input.ally.hp),
            normalizeEntityDistance(allyDistance, normalizationScale),
            input.enemy0.missing ? 0 : toNormHp(input.enemy0.hp),
            normalizeEntityDistance(enemy0Distance, normalizationScale),
            input.enemy1.missing ? 0 : toNormHp(input.enemy1.hp),
            normalizeEntityDistance(enemy1Distance, normalizationScale)
        ],
        rawFeatures: {
            selfHp: input.self.hp,
            maxHp: LINEAR_INTENT_MAX_HP,
            allyHp: input.ally.missing ? 0 : input.ally.hp,
            allyDistance,
            enemy0Hp: input.enemy0.missing ? 0 : input.enemy0.hp,
            enemy0Distance,
            enemy1Hp: input.enemy1.missing ? 0 : input.enemy1.hp,
            enemy1Distance,
            weaponReady,
            enemyInRange,
            nearestEnemyDistance,
            attackRange: LINEAR_INTENT_ATTACK_RANGE,
            arenaDiagonal: diagonal,
            distanceNormalizationScale: normalizationScale
        }
    };
}

export function getLinearIntentFeatureNames(): typeof LINEAR_INTENT_FEATURE_NAMES {
    return LINEAR_INTENT_FEATURE_NAMES;
}

export function extractLinearIntentFeatureVector(input: LinearIntentFeatureInput): number[] {
    return buildLinearIntentFeatureSnapshot(input).featureVector;
}

export function normalizeVector(dx: number, dy: number): { x: number; y: number } {
    const length = Math.hypot(dx, dy);
    if (!Number.isFinite(length) || length === 0) {
        return { x: 0, y: 0 };
    }
    return { x: dx / length, y: dy / length };
}
