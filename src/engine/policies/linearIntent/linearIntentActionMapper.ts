import { LinearIntentAction, LinearIntentFeatureInput, LinearIntentLabel } from './linearIntentTypes';
import { normalizeVector } from './linearIntentFeatures';

function manhattanDistance(a: { x: number; y: number }, b: { x: number; y: number }): number {
    return Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
}

function clampFire(value: boolean): number {
    return value ? 1 : 0;
}

function getSelf(input: LinearIntentFeatureInput): { x: number; y: number } {
    return { x: input.self.x, y: input.self.y };
}

function getEnemyPositions(input: LinearIntentFeatureInput): Array<{ hp: number; x: number; y: number }> {
    return [input.enemy0, input.enemy1];
}

function getNearestEnemy(input: LinearIntentFeatureInput): { hp: number; x: number; y: number } {
    const self = getSelf(input);
    return getEnemyPositions(input).reduce((best, enemy) => {
        if (!best) return enemy;
        return manhattanDistance(self, enemy) < manhattanDistance(self, best) ? enemy : best;
    }, null as { hp: number; x: number; y: number } | null) || input.enemy0;
}

function getEnemyClosestToAlly(input: LinearIntentFeatureInput): { hp: number; x: number; y: number } {
    const ally = input.ally;
    return getEnemyPositions(input).reduce((best, enemy) => {
        if (!best) return enemy;
        return manhattanDistance(ally, enemy) < manhattanDistance(ally, best) ? enemy : best;
    }, null as { hp: number; x: number; y: number } | null) || input.enemy0;
}

function buildVector(from: { x: number; y: number }, to: { x: number; y: number }): { x: number; y: number } {
    return normalizeVector(to.x - from.x, to.y - from.y);
}

export function mapLinearIntentToAction(intent: LinearIntentLabel, input: LinearIntentFeatureInput): LinearIntentAction {
    const self = getSelf(input);
    const ally = input.ally;
    const nearestEnemy = getNearestEnemy(input);
    const enemyClosestToAlly = getEnemyClosestToAlly(input);
    const canFire = input.self.weaponCooldownSteps !== undefined && input.self.weaponCooldownSteps <= 0;

    switch (intent) {
        case 'attack_nearest_enemy': {
            const move = buildVector(self, nearestEnemy);
            const aim = buildVector(self, nearestEnemy);
            return {
                moveX: move.x,
                moveY: move.y,
                aimX: aim.x,
                aimY: aim.y,
                fire: clampFire(canFire)
            };
        }
        case 'support_teammate_under_pressure': {
            const move = buildVector(self, ally);
            const aim = buildVector(self, enemyClosestToAlly);
            return {
                moveX: move.x,
                moveY: move.y,
                aimX: aim.x,
                aimY: aim.y,
                fire: clampFire(canFire)
            };
        }
        case 'reduce_isolation': {
            const move = buildVector(self, ally);
            const aim = buildVector(self, nearestEnemy);
            return {
                moveX: move.x,
                moveY: move.y,
                aimX: aim.x,
                aimY: aim.y,
                fire: 0
            };
        }
        case 'retreat_when_low_hp': {
            const move = buildVector(self, ally);
            const aim = buildVector(self, nearestEnemy);
            return {
                moveX: move.x,
                moveY: move.y,
                aimX: aim.x,
                aimY: aim.y,
                fire: clampFire(canFire)
            };
        }
        default:
            return {
                moveX: 0,
                moveY: 0,
                aimX: 0,
                aimY: 0,
                fire: 0
            };
    }
}
