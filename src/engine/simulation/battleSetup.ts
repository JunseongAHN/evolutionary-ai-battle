import config from '../../../config/default.json';

export type BattleMode = 'solo' | 'duo';
export type PlayersPerTeam = 1 | 2;
export type TeamId = string;
export type PlayerId = string;

export interface BattleConfig {
    mode?: BattleMode;
    teamCount?: number;
    playersPerTeam?: PlayersPerTeam;
    maxSteps?: number;
}

export interface ResolvedBattleConfig {
    mode: BattleMode;
    teamCount: number;
    playersPerTeam: PlayersPerTeam;
    maxSteps?: number;
}

export interface Team {
    id: TeamId;
    playerIds: PlayerId[];
}

export interface BattlePlayer {
    id: PlayerId;
    numericId: number;
    teamId: TeamId;
    slotIndex: number;
    hp: number;
    alive: boolean;
}

export const DEFAULT_BATTLE_CONFIG: ResolvedBattleConfig = {
    mode: 'duo',
    teamCount: 2,
    playersPerTeam: 2
};

export function resolveBattleConfig(battleConfig: BattleConfig = {}): ResolvedBattleConfig {
    const mode = battleConfig.mode || DEFAULT_BATTLE_CONFIG.mode;
    if (mode !== 'solo' && mode !== 'duo') {
        throw new Error(`Unsupported battle mode: ${mode}`);
    }

    const playersPerTeam = battleConfig.playersPerTeam || (mode === 'solo' ? 1 : 2);
    if ((mode === 'solo' && playersPerTeam !== 1) || (mode === 'duo' && playersPerTeam !== 2)) {
        throw new Error(`Battle mode ${mode} requires playersPerTeam=${mode === 'solo' ? 1 : 2}`);
    }

    const teamCount = battleConfig.teamCount ?? DEFAULT_BATTLE_CONFIG.teamCount;
    if (!Number.isInteger(teamCount) || teamCount < 2) {
        throw new Error('Battle config teamCount must be an integer >= 2');
    }

    return {
        mode,
        teamCount,
        playersPerTeam,
        maxSteps: battleConfig.maxSteps
    };
}

export function getPlayerId(teamId: TeamId, slotIndex: number): PlayerId {
    return `${teamId}-${slotIndex}`;
}

export function createBattleTeams(battleConfig: BattleConfig = {}): { config: ResolvedBattleConfig; teams: Team[]; players: BattlePlayer[] } {
    const resolvedConfig = resolveBattleConfig(battleConfig);
    const teams: Team[] = [];
    const players: BattlePlayer[] = [];

    for (let teamIndex = 0; teamIndex < resolvedConfig.teamCount; teamIndex += 1) {
        const teamId = `team-${teamIndex}`;
        const playerIds: PlayerId[] = [];

        for (let slotIndex = 0; slotIndex < resolvedConfig.playersPerTeam; slotIndex += 1) {
            const playerId = getPlayerId(teamId, slotIndex);
            playerIds.push(playerId);
            players.push({
                id: playerId,
                numericId: players.length + 1,
                teamId,
                slotIndex,
                hp: config.startingLives,
                alive: true
            });
        }

        teams.push({ id: teamId, playerIds });
    }

    return {
        config: resolvedConfig,
        teams,
        players
    };
}
