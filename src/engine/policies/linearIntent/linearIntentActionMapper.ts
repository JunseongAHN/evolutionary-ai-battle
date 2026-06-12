import { LinearIntentAction, LinearIntentFeatureInput, LinearIntentLabel } from './linearIntentTypes';
import { linearIntentDistance, normalizeVector } from './linearIntentFeatures';
import config from '../../../../config/default.json';

const AIM_FIRE_TOLERANCE_DEGREES = 10;

function getSelf(input: LinearIntentFeatureInput): { x: number; y: number } {
    return { x: input.self.x, y: input.self.y };
}

function getEnemyPositions(input: LinearIntentFeatureInput): Array<{ hp: number; x: number; y: number }> {
    return [input.enemy0, input.enemy1].filter((enemy) => !enemy.missing && enemy.hp > 0);
}

function getNearestEnemy(input: LinearIntentFeatureInput): { hp: number; x: number; y: number } {
    const self = getSelf(input);
    return getEnemyPositions(input).reduce((best, enemy) => {
        if (!best) return enemy;
        return linearIntentDistance(self, enemy) < linearIntentDistance(self, best) ? enemy : best;
    }, null as { hp: number; x: number; y: number } | null) || input.enemy0;
}

function getEnemyClosestToAlly(input: LinearIntentFeatureInput): { hp: number; x: number; y: number } {
    const ally = input.ally;
    return getEnemyPositions(input).reduce((best, enemy) => {
        if (!best) return enemy;
        return linearIntentDistance(ally, enemy) < linearIntentDistance(ally, best) ? enemy : best;
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
    switch (intent) {
        case 'attack_nearest_enemy': {
            const move = buildVector(self, nearestEnemy);
            const aim = buildVector(self, nearestEnemy);
            return {
                moveX: move.x,
                moveY: move.y,
                aimX: aim.x,
                aimY: aim.y,
                // action.fire is policy intent. The battle system decides whether a shot is emitted.
                fire: 1,
                fireWhileAiming: true
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
                fire: 1
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
            const towardEnemy = buildVector(self, nearestEnemy);
            const aim = buildVector(self, nearestEnemy);
            return {
                moveX: -towardEnemy.x,
                moveY: -towardEnemy.y,
                aimX: aim.x,
                aimY: aim.y,
                fire: 1
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

function shortestAngleDelta(currentDegrees: number, targetDegrees: number): number {
    return ((targetDegrees - currentDegrees + 540) % 360) - 180;
}

export function linearIntentActionToBattleAction(
    action: LinearIntentAction,
    currentRotation: number = 0
): { dx: number; dy: number; dh: number; ds: boolean } {
    const scale = Number.isFinite(config.maxSpeed) ? config.maxSpeed : 1;
    const clampFinite = (value: number): number => (Number.isFinite(value) ? value : 0);
    const aimX = clampFinite(action.aimX);
    const aimY = clampFinite(action.aimY);
    const hasAim = Math.hypot(aimX, aimY) > 0;
    const targetRotation = hasAim ? Math.atan2(aimY, aimX) * 180 / Math.PI : currentRotation;
    const angleDelta = hasAim ? shortestAngleDelta(clampFinite(currentRotation), targetRotation) : 0;
    const rotationSpeed = Math.max(Math.min(angleDelta, scale), -scale);
    const aimAligned = hasAim && Math.abs(angleDelta) <= AIM_FIRE_TOLERANCE_DEGREES;

    return {
        dx: clampFinite(action.moveX) * scale,
        dy: clampFinite(action.moveY) * scale,
        dh: rotationSpeed,
        ds: action.fire > 0 && (action.fireWhileAiming === true || aimAligned)
    };
}
