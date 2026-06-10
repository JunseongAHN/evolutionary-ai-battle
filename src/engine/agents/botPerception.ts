import config from '../../../config/default.json';
import { degreesToRadians, rotateAroundPoint, translateMatrix } from '../../shared/math';

const MAP_WIDTH = config.mapWidth;
const MAP_HEIGHT = config.mapHeight;
const NN_SQUARE_SIZE = config.neuralNetworkSquareSize;

export function createBotBrainInputState(bot, otherPlayer) {
    if (!otherPlayer) {
        return {
            xPos: bot.xPos,
            yPos: bot.yPos,
            bullets: [],
            walls: []
        };
    }

    const rotationAngle = degreesToRadians(-bot.rotation);
    const translationMatrix = [MAP_WIDTH - bot.xPos, MAP_HEIGHT - bot.yPos];
    const otherPlayerRotated = rotateAroundPoint(bot.xPos, bot.yPos, rotationAngle, [otherPlayer.xPos, otherPlayer.yPos]);
    const otherPlayerTranslated = translateMatrix(translationMatrix, otherPlayerRotated);
    const walls = [];

    for (let i = NN_SQUARE_SIZE / 2; i < MAP_WIDTH; i += NN_SQUARE_SIZE) {
        walls.push({ xPos: i, yPos: -NN_SQUARE_SIZE / 2 });
        walls.push({ xPos: i, yPos: MAP_HEIGHT + (NN_SQUARE_SIZE / 2) });
    }
    for (let i = NN_SQUARE_SIZE / 2; i < MAP_HEIGHT; i += NN_SQUARE_SIZE) {
        walls.push({ xPos: -NN_SQUARE_SIZE / 2, yPos: i });
        walls.push({ xPos: MAP_WIDTH + (NN_SQUARE_SIZE / 2), yPos: i });
    }

    const verticalOffset = MAP_WIDTH - MAP_HEIGHT;
    return {
        xPos: otherPlayerTranslated[0],
        yPos: otherPlayerTranslated[1] + verticalOffset,
        bullets: otherPlayer.bullets.map((bullet) => {
            const rotated = rotateAroundPoint(bot.xPos, bot.yPos, rotationAngle, [bullet.xPos, bullet.yPos]);
            const translated = translateMatrix(translationMatrix, rotated);
            return {
                xPos: translated[0],
                yPos: translated[1] + verticalOffset
            };
        }),
        walls: walls.map((wall) => {
            const rotated = rotateAroundPoint(bot.xPos, bot.yPos, rotationAngle, [wall.xPos, wall.yPos]);
            const translated = translateMatrix(translationMatrix, rotated);
            return {
                xPos: translated[0],
                yPos: translated[1] + verticalOffset
            };
        })
    };
}
