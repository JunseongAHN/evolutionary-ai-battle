import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createBattleTeams, DEFAULT_BATTLE_CONFIG } from '../battleSetup';
import { getAlivePlayers, getAliveTeams, getEnemies, getTeam, getTeammates } from '../teams';

function toBots(players: ReturnType<typeof createBattleTeams>['players']) {
    return players.map((player) => ({
        actorId: player.id,
        teamId: player.teamId,
        lives: player.hp
    }));
}

test('solo mode with N=4 creates 4 teams and 4 players', () => {
    const setup = createBattleTeams({ mode: 'solo', teamCount: 4 });

    assert.equal(setup.config.playersPerTeam, 1);
    assert.equal(setup.teams.length, 4);
    assert.equal(setup.players.length, 4);
    setup.teams.forEach((team) => assert.equal(team.playerIds.length, 1));

    const bots = toBots(setup.players);
    setup.players.forEach((player) => {
        assert.deepEqual(getTeammates(bots, player.id), []);
    });
});

test('duo mode with N=2 creates 2 teams and 4 players', () => {
    const setup = createBattleTeams({ mode: 'duo', teamCount: 2 });

    assert.equal(setup.config.playersPerTeam, 2);
    assert.equal(setup.teams.length, 2);
    assert.equal(setup.players.length, 4);
    setup.teams.forEach((team) => assert.equal(team.playerIds.length, 2));

    const bots = toBots(setup.players);
    setup.players.forEach((player) => {
        assert.equal(getTeammates(bots, player.id).length, 1);
        assert.equal(getEnemies(bots, player.id).length, 2);
        assert.equal(getTeam(bots, player.id)?.id, player.teamId);
    });
});

test('default battle config preserves 2v2 behavior', () => {
    const setup = createBattleTeams(DEFAULT_BATTLE_CONFIG);

    assert.equal(setup.config.mode, 'duo');
    assert.equal(setup.config.teamCount, 2);
    assert.equal(setup.config.playersPerTeam, 2);
    assert.deepEqual(setup.teams.map((team) => team.playerIds), [
        ['team-0-0', 'team-0-1'],
        ['team-1-0', 'team-1-1']
    ]);
});

test('alive player and team helpers use generated teams', () => {
    const setup = createBattleTeams({ mode: 'solo', teamCount: 3 });
    const bots = toBots(setup.players);
    bots[1].lives = 0;

    assert.deepEqual(getAlivePlayers(bots).map((bot) => bot.actorId), ['team-0-0', 'team-2-0']);
    assert.deepEqual(getAliveTeams(bots).map((team) => team.id), ['team-0', 'team-2']);
});

