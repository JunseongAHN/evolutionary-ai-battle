import { createBattleEnvironmentView } from './view';
import { createLiveGameBoardViewModel, createTrajectoryGameBoardViewModel } from './logic/gameBoard';
import { createLiveBotBoardViewModel, createTrajectoryBotBoardViewModel } from './logic/botBoard';

export default class BattleEnvironmentController {
    view: ReturnType<typeof createBattleEnvironmentView>;
    canvasIds: any;
    currentFrame: any;
    currentMode: 'live' | 'trajectory' | null;
    animationFrameId: number | null;
    running: boolean;
    renderFrame: FrameRequestCallback;

    constructor(view = createBattleEnvironmentView()) {
        this.view = view;
        this.canvasIds = null;
        this.currentFrame = null;
        this.currentMode = null;
        this.animationFrameId = null;
        this.running = false;
        this.renderFrame = this.render.bind(this);
    }

    start(canvasIds) {
        this.canvasIds = canvasIds;
        if (this.running) return;

        this.running = true;
        this.animationFrameId = window.requestAnimationFrame(this.renderFrame);
    }

    stop() {
        this.running = false;
        if (this.animationFrameId !== null) {
            window.cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
    }

    setLiveFrame(bots) {
        this.currentMode = 'live';
        this.currentFrame = bots;
    }

    setTrajectoryFrame(stepFrame) {
        this.currentMode = 'trajectory';
        this.currentFrame = stepFrame;
    }

    render() {
        if (!this.running) return;

        try {
            if (this.canvasIds && this.currentFrame) {
                if (this.currentMode === 'live') {
                    this.drawLiveBattle(this.currentFrame);
                } else if (this.currentMode === 'trajectory') {
                    this.drawTrajectoryBattle(this.currentFrame);
                }
            }
        } catch (error) {
            console.error('BattleEnvironment render failed', error);
        } finally {
            this.animationFrameId = window.requestAnimationFrame(this.renderFrame);
        }
    }

    drawLiveBattle(bots) {
        this.view.drawBattleGround(this.canvasIds.battle, createLiveGameBoardViewModel(bots));
        bots.forEach((bot) => {
            try {
                const botCanvasId = this.canvasIds.bot(bot.id);
                this.view.drawBotBoard(botCanvasId, createLiveBotBoardViewModel(bot, bots));
            } catch (error) {
                console.error(`Bot ${bot.id} brain render failed`, error);
            }
        });
    }

    drawTrajectoryBattle(stepFrame) {
        this.view.drawBattleGround(this.canvasIds.battle, createTrajectoryGameBoardViewModel(stepFrame));
        stepFrame.players.forEach((player) => {
            try {
                const botCanvasId = this.canvasIds.bot(player.actorId);
                this.view.drawBotBoard(botCanvasId, createTrajectoryBotBoardViewModel({
                    ...player,
                    stepFrame
                }));
            } catch (error) {
                console.error(`Player ${player.actorId} brain replay render failed`, error);
            }
        });
    }
}
