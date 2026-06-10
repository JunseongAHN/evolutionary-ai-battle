import config from '../../../config/default.json';
import { BRAIN_CANVAS_SCALE } from '../../shared/constants';
import { degreesToRadians, rotateAroundPoint, translateMatrix } from '../../shared/math';

const BOT_RADIUS = config.botSize / 2;
const BULLET_RADIUS = config.bulletSize / 2;

export const battleEnvironmentTemplate = `
    <section class="battle-environment">
        <canvas id="battleground" width="1000" height="500"></canvas>
        <div class="brain-grid">
            <div class="brain-board"><h4>Bot 1 Brain</h4><canvas id="bot1brain" width="400" height="400"></canvas></div>
            <div class="brain-board"><h4>Bot 2 Brain</h4><canvas id="bot2brain" width="400" height="400"></canvas></div>
            <div class="brain-board"><h4>Bot 3 Brain</h4><canvas id="bot3brain" width="400" height="400"></canvas></div>
            <div class="brain-board"><h4>Bot 4 Brain</h4><canvas id="bot4brain" width="400" height="400"></canvas></div>
        </div>
    </section>
`;

function scaleForBrainView(value) {
    return Math.floor(value / config.neuralNetworkSquareSize) * BRAIN_CANVAS_SCALE;
}

function getCanvas(canvasId) {
    return document.getElementById(canvasId) as HTMLCanvasElement | null;
}

export function createBattleEnvironmentView() {
    return {
        drawBattleGround(canvasId, boardModel) {
            const canvas = getCanvas(canvasId);
            if (!canvas || !canvas.getContext) return;

            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#ddffdd';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            boardModel.bots.forEach((bot) => {
                const botColor = bot.teamId === 'team-a' ? '#ffdddd' : '#ddddff';
                ctx.fillStyle = botColor;
                ctx.beginPath();
                ctx.arc(bot.xPos, bot.yPos, BOT_RADIUS, 0, 2 * Math.PI, false);
                ctx.fill();

                ctx.strokeStyle = '#000000';
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

                ctx.fillStyle = '#000000';
                (bot.bullets || []).forEach((bullet) => {
                    ctx.beginPath();
                    ctx.arc(bullet.xPos, bullet.yPos, BULLET_RADIUS, 0, 2 * Math.PI, false);
                    ctx.fill();
                });
            });
        },

        drawBotBoard(canvasId, botBoardModel) {
            const canvas = getCanvas(canvasId);
            if (!canvas || !canvas.getContext) return;

            const ctx = canvas.getContext('2d');
            const bot = botBoardModel.bot;
            const translatedPositions = botBoardModel.translatedPositions;
            const playerBGColor = bot.teamId === 'team-a' ? '#ffdddd' : '#ddddff';
            const ownerColor = '#00aa00';
            ctx.fillStyle = playerBGColor;
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            const toBrainPoint = (x, y) => {
                const rotationAngle = degreesToRadians(-bot.rotation);
                const translationMatrix = [config.mapWidth - bot.xPos, config.mapHeight - bot.yPos];
                const rotatedPoint = rotateAroundPoint(bot.xPos, bot.yPos, rotationAngle, [x, y]);
                const translatedPoint = translateMatrix(translationMatrix, rotatedPoint);
                return [
                    scaleForBrainView(translatedPoint[0]),
                    scaleForBrainView(translatedPoint[1] + (config.mapWidth - config.mapHeight))
                ];
            };

            const players = (botBoardModel.players || []).length ? botBoardModel.players : [bot];
            players.forEach((player) => {
                const [playerXPos, playerYPos] = toBrainPoint(player.xPos, player.yPos);
                ctx.fillStyle = player.id === bot.id || player.actorId === bot.id || String(player.id) === String(bot.id)
                    ? ownerColor
                    : (player.teamId === 'team-a' ? '#ff0000' : '#0000ff');
                ctx.fillRect(playerXPos, playerYPos, BRAIN_CANVAS_SCALE, BRAIN_CANVAS_SCALE);
            });

            (translatedPositions.bullets || []).forEach((bullet) => {
                const [bulletXPos, bulletYPos] = toBrainPoint(bullet.xPos, bullet.yPos);
                ctx.fillStyle = '#000000';
                ctx.fillRect(bulletXPos, bulletYPos, BRAIN_CANVAS_SCALE, BRAIN_CANVAS_SCALE);
            });

            (translatedPositions.walls || []).forEach((wall) => {
                const [wallXPos, wallYPos] = toBrainPoint(wall.xPos, wall.yPos);
                ctx.fillStyle = '#000000';
                ctx.fillRect(wallXPos, wallYPos, BRAIN_CANVAS_SCALE, BRAIN_CANVAS_SCALE);
            });
        }
    };
}
