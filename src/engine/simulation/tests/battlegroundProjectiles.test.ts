import assert from 'node:assert/strict';
import { test } from 'node:test';
import Battleground from '../battleground';

test('projectile is removed on its first swept collision even when its shooter is dead', () => {
    const battleground = new Battleground() as any;
    const shooter = {
        id: 1,
        teamId: 'team-a',
        lives: 0,
        bullets: [{
            id: 'projectile-test',
            shooterId: 'team-a-0',
            shooterTeamId: 'team-a',
            xPos: 50,
            yPos: 100,
            rotation: 0
        }]
    };
    const target = {
        id: 2,
        teamId: 'team-b',
        lives: 5,
        bullets: [],
        xPos: 100,
        yPos: 100
    };

    battleground.bots = [shooter, target];
    battleground.stepDamage = {};
    battleground.updateBulletsForBot(shooter, 0.3, 1000 / 75);

    assert.equal(shooter.bullets.length, 0);
    assert.equal(target.lives, 4);
});

test('projectile disappears when it collides with a friendly bot without damaging it', () => {
    const battleground = new Battleground() as any;
    const shooter = {
        id: 1,
        teamId: 'team-a',
        lives: 5,
        bullets: [{
            id: 'projectile-friendly-test',
            shooterId: 'team-a-0',
            shooterTeamId: 'team-a',
            xPos: 50,
            yPos: 100,
            rotation: 0
        }],
        xPos: 50,
        yPos: 100
    };
    const teammate = {
        id: 2,
        teamId: 'team-a',
        lives: 5,
        bullets: [],
        xPos: 100,
        yPos: 100
    };

    battleground.bots = [shooter, teammate];
    battleground.stepDamage = {};
    battleground.updateBulletsForBot(shooter, 0.3, 1000 / 75);

    assert.equal(shooter.bullets.length, 0);
    assert.equal(teammate.lives, 5);
});

test('projectile disappears when it collides with a dead bot body', () => {
    const battleground = new Battleground() as any;
    const shooter = {
        id: 1,
        teamId: 'team-a',
        lives: 5,
        bullets: [{
            id: 'projectile-dead-body-test',
            shooterId: 'team-a-0',
            shooterTeamId: 'team-a',
            xPos: 50,
            yPos: 100,
            rotation: 0
        }],
        xPos: 50,
        yPos: 100
    };
    const deadTarget = {
        id: 2,
        teamId: 'team-b',
        lives: 0,
        bullets: [],
        xPos: 100,
        yPos: 100
    };

    battleground.bots = [shooter, deadTarget];
    battleground.stepDamage = {};
    battleground.updateBulletsForBot(shooter, 0.3, 1000 / 75);

    assert.equal(shooter.bullets.length, 0);
    assert.equal(deadTarget.lives, 0);
});
