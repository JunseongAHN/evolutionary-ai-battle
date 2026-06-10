import { createBotBrainInputState } from '../../../engine/agents/botPerception';

export function getNearestEnemy(bot, bots) {
    return bots
        .filter((otherBot) => otherBot !== bot && bot.teamId !== otherBot.teamId && otherBot.lives > 0)
        .reduce((nearestEnemy, enemy) => {
            if (!nearestEnemy) return enemy;
            const nearestDistance = Math.hypot(bot.xPos - nearestEnemy.xPos, bot.yPos - nearestEnemy.yPos);
            const enemyDistance = Math.hypot(bot.xPos - enemy.xPos, bot.yPos - enemy.yPos);
            return enemyDistance < nearestDistance ? enemy : nearestEnemy;
        }, null);
}

export function createLiveBotBoardViewModel(bot, bots) {
    return {
        bot,
        players: bots,
        translatedPositions: createBotBrainInputState(bot, getNearestEnemy(bot, bots))
    };
}

export function createTrajectoryBotBoardViewModel(playerRecord) {
    const players = playerRecord.stepFrame.players.map((player) => ({
        id: player.actorId,
        teamId: player.actorTeamId,
        xPos: player.measurements.positionX,
        yPos: player.measurements.positionY,
        rotation: 0,
        lives: player.measurements.hp,
        bullets: []
    }));
    const bot = players.find((player) => player.id === playerRecord.actorId);

    return {
        bot,
        players,
        translatedPositions: createBotBrainInputState(bot, getNearestEnemy(bot, players)),
        playerRecord
    };
}
