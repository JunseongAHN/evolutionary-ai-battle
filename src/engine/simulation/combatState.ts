export const MAX_HP = 100;
export const BULLET_DAMAGE = 10;
export const FIRE_COOLDOWN_STEPS = 5;
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
    weaponCooldownSteps
}: {
    alive: boolean;
    attemptedFire: number;
    weaponCooldownSteps: number;
}): { canFire: boolean; didFire: boolean; nextWeaponCooldownSteps: number } {
    const canFire = alive && attemptedFire > FIRE_THRESHOLD && weaponCooldownSteps <= 0;
    return {
        canFire,
        didFire: canFire,
        nextWeaponCooldownSteps: canFire ? FIRE_COOLDOWN_STEPS : weaponCooldownSteps
    };
}
