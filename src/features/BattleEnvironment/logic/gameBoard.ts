export function createLiveGameBoardViewModel(bots) {
    return {
        mode: 'live',
        bots: bots.map((bot) => ({
            id: bot.id,
            teamId: bot.teamId,
            xPos: bot.xPos,
            yPos: bot.yPos,
            rotation: bot.rotation,
            lives: bot.lives,
            bullets: bot.bullets || []
        }))
    };
}

export function createTrajectoryGameBoardViewModel(stepFrame) {
    return {
        mode: 'trajectory',
        bots: stepFrame.players.map((player) => ({
            id: player.actorId,
            teamId: player.actorTeamId,
            xPos: player.measurements.positionX,
            yPos: player.measurements.positionY,
            rotation: 0,
            lives: player.measurements.hp,
            bullets: []
        }))
    };
}
