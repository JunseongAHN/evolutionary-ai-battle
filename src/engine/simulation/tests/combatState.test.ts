import assert from 'node:assert/strict';
import { test } from 'node:test';
import { advanceWeaponCooldown, FIRE_COOLDOWN_STEPS, resolveFireAttempt } from '../combatState';

test('advanceWeaponCooldown decreases cooldown and clamps at zero', () => {
    assert.equal(advanceWeaponCooldown(3), 2);
    assert.equal(advanceWeaponCooldown(1), 0);
    assert.equal(advanceWeaponCooldown(0), 0);
});

test('resolveFireAttempt allows firing only when alive, intent is present, and cooldown is clear', () => {
    const readyShot = resolveFireAttempt({
        alive: true,
        attemptedFire: 1,
        weaponCooldownSteps: 0
    });

    assert.equal(readyShot.canFire, true);
    assert.equal(readyShot.didFire, true);
    assert.equal(readyShot.nextWeaponCooldownSteps, FIRE_COOLDOWN_STEPS);

    const blockedShot = resolveFireAttempt({
        alive: true,
        attemptedFire: 1,
        weaponCooldownSteps: 2
    });

    assert.equal(blockedShot.canFire, false);
    assert.equal(blockedShot.didFire, false);
    assert.equal(blockedShot.nextWeaponCooldownSteps, 2);
});

test('resolveFireAttempt keeps attempted fire separate from actual fire', () => {
    const noIntent = resolveFireAttempt({
        alive: true,
        attemptedFire: 0,
        weaponCooldownSteps: 0
    });

    assert.equal(noIntent.canFire, false);
    assert.equal(noIntent.didFire, false);

    const deadPlayer = resolveFireAttempt({
        alive: false,
        attemptedFire: 1,
        weaponCooldownSteps: 0
    });

    assert.equal(deadPlayer.canFire, false);
    assert.equal(deadPlayer.didFire, false);
});

test('resolveFireAttempt treats system permission separately from policy fire intent', () => {
    const blockedBySystem = resolveFireAttempt({
        alive: true,
        attemptedFire: 1,
        weaponCooldownSteps: 0,
        systemAllowed: false
    });

    assert.equal(blockedBySystem.canFire, false);
    assert.equal(blockedBySystem.didFire, false);
});
