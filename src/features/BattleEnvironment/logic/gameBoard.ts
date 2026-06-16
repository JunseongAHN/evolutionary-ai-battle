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
    const projectiles = stepFrame.environment?.projectiles || [];

    return {
        mode: 'trajectory',
        bots: stepFrame.players.map((player) => ({
            id: player.actorId,
            teamId: player.actorTeamId,
            xPos: player.state.positionX,
            yPos: player.state.positionY,
            rotation: Math.atan2(player.state.headingY, player.state.headingX) * 180 / Math.PI,
            lives: player.state.hp,
            bullets: projectiles
                .filter((projectile) => projectile.shooterId === player.actorId)
                .map((projectile) => ({
                    id: projectile.id,
                    shooterId: projectile.shooterId,
                    shooterTeamId: projectile.shooterTeamId,
                    xPos: projectile.positionX,
                    yPos: projectile.positionY,
                    rotation: Math.atan2(projectile.headingY, projectile.headingX) * 180 / Math.PI
                }))
        }))
    };
}
