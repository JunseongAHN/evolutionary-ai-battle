import { BRAIN_CANVAS_WIDTH } from '../../../shared/constants';
import { degreesToRadians } from '../../../shared/math';
import config from '../../../../config/default.json';

const BRAIN_WORLD_WIDTH = 2000;
const BRAIN_SCALE = BRAIN_CANVAS_WIDTH / BRAIN_WORLD_WIDTH;
const BRAIN_CENTER = BRAIN_CANVAS_WIDTH / 2;

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

function projectPointForBot(bot, xPos, yPos) {
    const angle = degreesToRadians(-bot.rotation - 90);
    const deltaX = xPos - bot.xPos;
    const deltaY = yPos - bot.yPos;
    const rotatedX = Math.cos(angle) * deltaX - Math.sin(angle) * deltaY;
    const rotatedY = Math.sin(angle) * deltaX + Math.cos(angle) * deltaY;

    return {
        xPos: BRAIN_CENTER + (rotatedX * BRAIN_SCALE),
        yPos: BRAIN_CENTER + (rotatedY * BRAIN_SCALE)
    };
}

function createBotBoardViewModel(bot, bots) {
    return {
        backgroundColor: bot.teamId === 'team-a' ? '#ffdddd' : '#ddddff',
        boundary: [
            projectPointForBot(bot, 0, 0),
            projectPointForBot(bot, config.mapWidth, 0),
            projectPointForBot(bot, config.mapWidth, config.mapHeight),
            projectPointForBot(bot, 0, config.mapHeight)
        ],
        players: bots.map((player) => ({
            ...projectPointForBot(bot, player.xPos, player.yPos),
            color: player.id === bot.id ? '#00aa00' : (player.teamId === 'team-a' ? '#ff0000' : '#0000ff')
        })),
        bullets: bots.flatMap((player) => (player.bullets || []).map((bullet) => ({
            ...projectPointForBot(bot, bullet.xPos, bullet.yPos),
            color: '#000000'
        })))
    };
}

export function createLiveBotBoardViewModel(bot, bots) {
    return createBotBoardViewModel(bot, bots);
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

    return createBotBoardViewModel(bot, players);
}
