import { createBattleEnvironmentView } from './view';
import { createLiveGameBoardViewModel, createTrajectoryGameBoardViewModel } from './logic/gameBoard';
import { createLiveBotBoardViewModel, createTrajectoryBotBoardViewModel } from './logic/botBoard';

export default class BattleEnvironmentController {
    view: ReturnType<typeof createBattleEnvironmentView>;

    constructor(view = createBattleEnvironmentView()) {
        this.view = view;
    }

    renderLiveBattle(canvasIds, bots) {
        this.view.drawBattleGround(canvasIds.battle, createLiveGameBoardViewModel(bots));
        bots.forEach((bot) => {
            const botCanvasId = canvasIds.bot(bot.id);
            this.view.drawBotBoard(botCanvasId, createLiveBotBoardViewModel(bot, bots));
        });
    }

    renderTrajectoryBattle(canvasIds, stepFrame) {
        this.view.drawBattleGround(canvasIds.battle, createTrajectoryGameBoardViewModel(stepFrame));
        stepFrame.players.forEach((player) => {
            const botCanvasId = canvasIds.bot(player.actorId);
            this.view.drawBotBoard(botCanvasId, {
                ...createTrajectoryBotBoardViewModel({
                    ...player,
                    stepFrame
                })
            });
        });
    }
}
