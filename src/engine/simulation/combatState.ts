import config from '../../../config/default.json';

export const MAX_HP = 100;
export const BULLET_DAMAGE = config.bulletDamage;
export const FIRE_COOLDOWN_STEPS = config.fireCooldownSteps;
export const FIRE_THRESHOLD = 0.5;

export function advanceWeaponCooldown(currentCooldown: number): number {
    if (!Number.isFinite(currentCooldown)) {
        return 0;
    }
    return Math.max(currentCooldown - 1, 0);
}

export function resolveFireAttempt({
    alive,
    attemptedFire,
    weaponCooldownSteps,
    systemAllowed = true
}: {
    alive: boolean;
    attemptedFire: number;
    weaponCooldownSteps: number;
    systemAllowed?: boolean;
}): { canFire: boolean; didFire: boolean; nextWeaponCooldownSteps: number } {
    const canFire = alive && attemptedFire > FIRE_THRESHOLD && weaponCooldownSteps <= 0 && systemAllowed;
    return {
        canFire,
        didFire: canFire,
        nextWeaponCooldownSteps: canFire ? FIRE_COOLDOWN_STEPS : weaponCooldownSteps
    };
}
