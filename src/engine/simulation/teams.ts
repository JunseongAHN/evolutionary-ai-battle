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
