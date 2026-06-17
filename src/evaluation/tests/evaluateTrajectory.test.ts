import assert from 'node:assert/strict';
import { test } from 'node:test';
import { synthetic2v2Trajectory } from '../../engine/traces/fixtures/synthetic2v2Trajectory';
import { evaluateTrajectory } from '../evaluateTrajectory';

function cloneTrajectory(trajectory) {
    return JSON.parse(JSON.stringify(trajectory));
}

function createSoloTrajectory() {
    const trajectory = cloneTrajectory(synthetic2v2Trajectory);
    const ids = ['team-0-0', 'team-1-0', 'team-2-0', 'team-3-0'];
    const oldIds = ['team-a-0', 'team-a-1', 'team-b-0', 'team-b-1'];
    const idMap = Object.fromEntries(oldIds.map((oldId, index) => [oldId, ids[index]]));
    const teamMap = {
        'team-a': 'team-0',
        'team-b': 'team-2'
    };

    trajectory.scenarioId = 'solo_4x1';
    trajectory.battleConfig = { mode: 'solo', teamCount: 4, playersPerTeam: 1 };
    trajectory.teams = ids.map((id, index) => ({
        teamId: `team-${index}`,
        playerIds: [id]
    }));
    trajectory.players = ids.map((id, index) => ({
        id,
        teamId: `team-${index}`
    }));

    trajectory.initialState.players.forEach((player, index) => {
        player.id = ids[index];
        player.teamId = `team-${index}`;
    });

    trajectory.steps.forEach((step) => {
        step.players.forEach((player, index) => {
            player.actorId = idMap[player.actorId] || ids[index];
            player.actorTeamId = `team-${index}`;
            player.measurements.nearestAllyDistance = 0;
        });
        step.events = (step.events || []).map((event) => ({
            ...event,
            actorId: idMap[event.actorId] || event.actorId,
            targetId: idMap[event.targetId] || event.targetId,
            actorTeamId: teamMap[event.actorTeamId] || event.actorTeamId,
            targetTeamId: teamMap[event.targetTeamId] || event.targetTeamId
        }));
    });

    return trajectory;
}

test('evaluateTrajectory combines player and CPC metrics', () => {
    const evaluation = evaluateTrajectory(synthetic2v2Trajectory);

    assert.equal(evaluation.trajectoryId, synthetic2v2Trajectory.trajectoryId);
    assert.equal(evaluation.schemaVersion, synthetic2v2Trajectory.schemaVersion);
    assert.equal(Object.keys(evaluation.players).length, 4);
    assert.equal(Object.keys(evaluation.teams).length, 2);
});

test('evaluateTrajectory includes evaluationScore', () => {
    const evaluation = evaluateTrajectory(synthetic2v2Trajectory);

    const player = evaluation.players['team-a-0'];
    assert.equal(typeof player.evaluationScore, 'number');
});

test('evaluateTrajectory includes team aggregation', () => {
    const evaluation = evaluateTrajectory(synthetic2v2Trajectory);

    const team = evaluation.teams['team-a'];
    assert.deepEqual(team.playerIds.sort(), ['team-a-0', 'team-a-1']);
    assert.equal(team.damageDealt, 1);
    assert.equal(team.damageTaken, 2);
    assert.equal(team.survivalSteps, 4);
});

test('evaluateTrajectory marks cooperation not applicable in solo mode', () => {
    const evaluation = evaluateTrajectory(createSoloTrajectory());

    assert.equal(Object.keys(evaluation.players).length, 4);
    assert.equal(Object.keys(evaluation.teams).length, 4);
    Object.values(evaluation.players).forEach((player) => {
        assert.equal(player.cpc.applicable, false);
    });
});

test('evaluateTrajectory keeps cooperation applicable in duo mode', () => {
    const evaluation = evaluateTrajectory(synthetic2v2Trajectory);

    Object.values(evaluation.players).forEach((player) => {
        assert.equal(player.cpc.applicable, true);
    });
});

test('evaluateTrajectory output is JSON stringify compatible', () => {
    const evaluation = evaluateTrajectory(synthetic2v2Trajectory);
    assert.doesNotThrow(() => JSON.stringify(evaluation));
});
