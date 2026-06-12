import { LinearIntentFeatureInput } from './linearIntentTypes';

function distance(a: { x: number; y: number }, b: { x: number; y: number }): number {
    return Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
}

function missingEntity(position: { x: number; y: number }) {
    return { hp: 0, x: position.x, y: position.y, missing: true };
}

function getBotPosition(bot: any): { x: number; y: number } {
    return { x: bot.xPos ?? 0, y: bot.yPos ?? 0 };
}

function isAliveBot(bot: any): boolean {
    return (bot?.lives ?? bot?.hp ?? 0) > 0;
}

function getBotHp(bot: any): number {
    const hp = bot?.lives ?? bot?.hp ?? 0;
    return Number.isFinite(hp) ? hp : 0;
}

function getBotCooldown(botIndex: number, battleground: any): number {
    const cooldown = battleground?.weaponCooldownSteps?.[botIndex];
    return Number.isFinite(cooldown) ? cooldown : 0;
}

function getEntityFromBot(bot: any): { hp: number; x: number; y: number } {
    return {
        hp: getBotHp(bot),
        x: bot?.xPos ?? 0,
        y: bot?.yPos ?? 0
    };
}

function chooseNearest(reference: { x: number; y: number }, candidates: any[]): any | null {
    return candidates.reduce((nearest: any | null, candidate: any) => {
        if (!nearest) return candidate;
        return distance(reference, getBotPosition(candidate)) < distance(reference, getBotPosition(nearest))
            ? candidate
            : nearest;
    }, null);
}

export function buildLinearIntentFeatureStateForBot(bot: any, battleground: any): LinearIntentFeatureInput {
    const bots = Array.isArray(battleground?.bots) ? battleground.bots : [];
    const botIndex = bots.indexOf(bot);
    const self = getEntityFromBot(bot);
    const position = getBotPosition(bot);
    const cooldown = getBotCooldown(botIndex, battleground);
    const teamBots = bots.filter((candidate: any) => candidate && candidate.teamId === bot.teamId && candidate !== bot && isAliveBot(candidate));
    const enemyBots = bots.filter((candidate: any) => candidate && candidate.teamId !== bot.teamId && isAliveBot(candidate));

    const nearestAlly = chooseNearest(position, teamBots);
    const sortedEnemies = enemyBots
        .slice()
        .sort((left: any, right: any) => distance(position, getBotPosition(left)) - distance(position, getBotPosition(right)));

    // Conservative fallbacks keep the model running even in degenerate states with missing teammates or enemies.
    const ally = nearestAlly ? getEntityFromBot(nearestAlly) : missingEntity(position);
    const enemy0 = sortedEnemies[0] ? getEntityFromBot(sortedEnemies[0]) : missingEntity(position);
    const enemy1 = sortedEnemies[1] ? getEntityFromBot(sortedEnemies[1]) : missingEntity(position);
    const environment = typeof battleground?.getEnvironmentState === 'function'
        ? battleground.getEnvironmentState()
        : battleground?.environment || { width: 1, height: 1 };

    return {
        gridWidth: Number.isFinite(environment.width) ? environment.width : 1,
        gridHeight: Number.isFinite(environment.height) ? environment.height : 1,
        self: {
            ...self,
            weaponCooldownSteps: cooldown
        },
        ally,
        enemy0,
        enemy1
    };
}
