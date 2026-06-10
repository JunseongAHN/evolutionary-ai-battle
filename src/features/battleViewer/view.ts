import config from '../../../config/default.json';
import { BRAIN_CANVAS_SCALE } from '../../shared/constants';

const BOT_RADIUS = config.botSize / 2;
const BULLET_RADIUS = config.bulletSize / 2;

function scaleForBrainView(value) {
    return Math.floor(value / config.neuralNetworkSquareSize) * BRAIN_CANVAS_SCALE;
}

// Owns browser canvas rendering for the battle viewer.
export function createBattleViewer() {
    return {
        drawBattleground(bots) {
            const canvas = document.getElementById('battleground') as HTMLCanvasElement | null;
            if (!canvas || !canvas.getContext) return;

            const ctx = canvas.getContext('2d');
            ctx.fillStyle = "#ddffdd";
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            bots.forEach((bot) => {
                const botColor = bot.id == 1 ? "#ffdddd" : "#ddddff";
                ctx.fillStyle = botColor;
                ctx.beginPath();
                ctx.arc(bot.xPos, bot.yPos, BOT_RADIUS, 0, 2 * Math.PI, false);
                ctx.fill();

                ctx.strokeStyle = "#000000";
                ctx.lineWidth = 1;
                ctx.stroke();

                ctx.beginPath();
                ctx.lineWidth = 3;
                ctx.moveTo(bot.xPos, bot.yPos);
                ctx.lineTo(
                    bot.xPos + (BOT_RADIUS * Math.cos(bot.rotation * Math.PI / 180)),
                    bot.yPos + (BOT_RADIUS * Math.sin(bot.rotation * Math.PI / 180)),
                );
                ctx.stroke();
                ctx.resetTransform();

                ctx.fillStyle = "#000000";
                bot.bullets.forEach((bullet) => {
                    ctx.beginPath();
                    ctx.arc(bullet.xPos, bullet.yPos, BULLET_RADIUS, 0, 2 * Math.PI, false);
                    ctx.fill();
                });
            });
        },

        drawBrain(bot, translatedPositions) {
            const canvas = document.getElementById('bot' + bot.id + 'brain') as HTMLCanvasElement | null;
            if (!canvas || !canvas.getContext) return;

            const ctx = canvas.getContext('2d');
            const playerBGColor = bot.id == 1 ? "#ffdddd" : "#ddddff";
            const playerColor = bot.id == 1 ? "#ff0000" : "#0000ff";
            const enemyColor = bot.id == 1 ? "#0000ff" : "#ff0000";
            ctx.fillStyle = playerBGColor;
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.fillStyle = playerColor;
            ctx.fillRect(
                scaleForBrainView(config.mapWidth),
                scaleForBrainView(config.mapHeight + bot.getVerticalOffset()),
                BRAIN_CANVAS_SCALE,
                BRAIN_CANVAS_SCALE
            );

            ctx.fillStyle = enemyColor;
            ctx.fillRect(
                scaleForBrainView(translatedPositions.xPos),
                scaleForBrainView(translatedPositions.yPos),
                BRAIN_CANVAS_SCALE,
                BRAIN_CANVAS_SCALE
            );

            ctx.fillStyle = "#000000";
            translatedPositions.bullets.concat(translatedPositions.walls).forEach((object) => {
                ctx.fillRect(
                    scaleForBrainView(object.xPos),
                    scaleForBrainView(object.yPos),
                    BRAIN_CANVAS_SCALE,
                    BRAIN_CANVAS_SCALE
                );
            });
        }
    };
}
