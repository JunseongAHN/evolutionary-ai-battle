export function isSameTeam(a, b) {
    return a.teamId === b.teamId;
}

export function isEnemy(a, b) {
    return !isSameTeam(a, b);
}

export function getAliveBots(bots) {
    return bots.filter((bot) => bot.lives > 0);
}

export function getAliveTeamIds(bots) {
    return Array.from(new Set(getAliveBots(bots).map((bot) => bot.teamId)));
}

export function getTeam(bots, playerId) {
    const player = bots.find((bot) => bot.actorId === playerId || bot.playerId === playerId);
    if (!player) {
        return null;
    }

    return {
        id: player.teamId,
        playerIds: bots
            .filter((bot) => bot.teamId === player.teamId)
            .map((bot) => bot.actorId || bot.playerId)
            .filter(Boolean)
    };
}

export function getTeammates(bots, playerId) {
    const player = bots.find((bot) => bot.actorId === playerId || bot.playerId === playerId);
    if (!player) {
        return [];
    }

    return bots.filter((bot) => bot !== player && bot.teamId === player.teamId);
}

export function getEnemies(bots, playerId) {
    const player = bots.find((bot) => bot.actorId === playerId || bot.playerId === playerId);
    if (!player) {
        return [];
    }

    return bots.filter((bot) => bot.teamId !== player.teamId);
}

export function getAlivePlayers(bots) {
    return getAliveBots(bots);
}

export function getAliveTeams(bots) {
    return getAliveTeamIds(bots).map((teamId) => ({
        id: teamId,
        playerIds: getAliveBots(bots)
            .filter((bot) => bot.teamId === teamId)
            .map((bot) => bot.actorId || bot.playerId)
            .filter(Boolean)
    }));
}
