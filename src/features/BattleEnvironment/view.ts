import config from '../../../config/default.json';
import { BRAIN_CANVAS_SCALE } from '../../shared/constants';

const BOT_RADIUS = config.botSize / 2;
const BULLET_RADIUS = config.bulletSize / 2;

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

                const hpRatio = Math.max(0, Math.min(1, bot.lives / config.startingLives));
                const hpBarWidth = 54;
                const hpBarHeight = 7;
                const hpBarX = bot.xPos - (hpBarWidth / 2);
                const hpBarY = bot.yPos > 48 ? bot.yPos - BOT_RADIUS - 20 : bot.yPos + BOT_RADIUS + 10;
                ctx.fillStyle = '#333333';
                ctx.fillRect(hpBarX - 1, hpBarY - 1, hpBarWidth + 2, hpBarHeight + 2);
                ctx.fillStyle = hpRatio > 0.5 ? '#2eaf55' : (hpRatio > 0.2 ? '#e2a72e' : '#d13c3c');
                ctx.fillRect(hpBarX, hpBarY, hpBarWidth * hpRatio, hpBarHeight);
                ctx.fillStyle = '#111111';
                ctx.font = 'bold 11px sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText(`Bot ${bot.id} HP ${bot.lives}/${config.startingLives}`, bot.xPos, hpBarY - 4);

                ctx.fillStyle = '#000000';
                (bot.bullets || []).forEach((bullet) => {
                    ctx.beginPath();
                    ctx.arc(bullet.xPos, bullet.yPos, BULLET_RADIUS, 0, 2 * Math.PI, false);
                    ctx.fill();
                });
            });
            ctx.textAlign = 'start';
        },

        drawBotBoard(canvasId, botBoardModel) {
            const canvas = getCanvas(canvasId);
            if (!canvas || !canvas.getContext) return;

            const ctx = canvas.getContext('2d');
            ctx.fillStyle = botBoardModel.backgroundColor;
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            if (botBoardModel.boundary && botBoardModel.boundary.length) {
                ctx.beginPath();
                ctx.moveTo(botBoardModel.boundary[0].xPos, botBoardModel.boundary[0].yPos);
                botBoardModel.boundary.slice(1).forEach((point) => {
                    ctx.lineTo(point.xPos, point.yPos);
                });
                ctx.closePath();
                ctx.strokeStyle = '#333333';
                ctx.lineWidth = 3;
                ctx.stroke();
            }

            botBoardModel.players.forEach((player) => {
                ctx.fillStyle = player.color;
                ctx.fillRect(player.xPos, player.yPos, BRAIN_CANVAS_SCALE, BRAIN_CANVAS_SCALE);
            });

            botBoardModel.bullets.forEach((bullet) => {
                ctx.fillStyle = bullet.color;
                ctx.fillRect(bullet.xPos, bullet.yPos, BRAIN_CANVAS_SCALE, BRAIN_CANVAS_SCALE);
                if (bullet.label) {
                    ctx.font = '9px sans-serif';
                    ctx.fillText(bullet.label, bullet.xPos + BRAIN_CANVAS_SCALE + 2, bullet.yPos + BRAIN_CANVAS_SCALE);
                }
            });
        }
    };
}
