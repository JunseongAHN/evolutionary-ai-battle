import { Trajectory, TRAJECTORY_SCHEMA_VERSION } from '../trace';

export const synthetic2v2Trajectory: Trajectory = {
    trajectoryId: 'synthetic-2v2-001',
    schemaVersion: TRAJECTORY_SCHEMA_VERSION,
    scenarioId: '2v2-skirmish',
    seed: 7,
    createdAt: '2026-06-11T00:00:00.000Z',
    teams: [
        {
            teamId: 'team-a',
            playerIds: ['team-a-0', 'team-a-1']
        },
        {
            teamId: 'team-b',
            playerIds: ['team-b-0', 'team-b-1']
        }
    ],
    players: [
        { id: 'team-a-0', teamId: 'team-a' },
        { id: 'team-a-1', teamId: 'team-a' },
        { id: 'team-b-0', teamId: 'team-b' },
        { id: 'team-b-1', teamId: 'team-b' }
    ],
    initialState: {
        environment: {
            width: 100,
            height: 100,
            bounds: {
                minX: -50,
                minY: -50,
                maxX: 50,
                maxY: 50
            },
            activeProjectileCount: 0
        },
        players: [
            {
                id: 'team-a-0',
                teamId: 'team-a',
                positionX: -20,
                positionY: 0,
                headingX: 1,
                headingY: 0,
                hp: 100,
                alive: true
            },
            {
                id: 'team-a-1',
                teamId: 'team-a',
                positionX: -18,
                positionY: 8,
                headingX: 1,
                headingY: 0,
                hp: 100,
                alive: true
            },
            {
                id: 'team-b-0',
                teamId: 'team-b',
                positionX: 20,
                positionY: 0,
                headingX: -1,
                headingY: 0,
                hp: 100,
                alive: true
            },
            {
                id: 'team-b-1',
                teamId: 'team-b',
                positionX: 18,
                positionY: -8,
                headingX: -1,
                headingY: 0,
                hp: 100,
                alive: true
            }
        ]
    },
    steps: [
        {
            step: 0,
            timeMs: 0,
            environment: {
                width: 100,
                height: 100,
                activeProjectileCount: 0
            },
            players: [
                {
                    actorId: 'team-a-0',
                    actorTeamId: 'team-a',
                    state: {
                        positionX: -20,
                        positionY: 0,
                        headingX: 1,
                        headingY: 0,
                        velocityX: 0,
                        velocityY: 0,
                        hp: 100,
                        alive: true,
                        cooldown: 0
                    },
                    action: {
                        moveX: 0.2,
                        moveY: 0,
                        aimX: 10,
                        aimY: 0,
                        fire: 0
                    },
                    reason: {
                        source: 'policy',
                        label: 'advance',
                        evidence: { focus: 'center' }
                    },
                    measurements: {
                        nearestAllyDistance: 3,
                        nearestEnemyDistance: 14,
                        targetId: 'team-b-0',
                        targetDistance: 14,
                        damageDealt: 0,
                        damageTaken: 0
                    }
                },
                {
                    actorId: 'team-a-1',
                    actorTeamId: 'team-a',
                    state: {
                        positionX: -18,
                        positionY: 8,
                        headingX: 1,
                        headingY: 0,
                        velocityX: 0,
                        velocityY: 0,
                        hp: 100,
                        alive: true,
                        cooldown: 0
                    },
                    action: {
                        moveX: 0.2,
                        moveY: 0.1,
                        aimX: 12,
                        aimY: 8,
                        fire: 0
                    },
                    reason: {
                        source: 'policy',
                        label: 'support',
                        evidence: { focus: 'team-a-0' }
                    },
                    measurements: {
                        nearestAllyDistance: 2,
                        nearestEnemyDistance: 18,
                        targetId: 'team-b-1',
                        targetDistance: 18,
                        damageDealt: 0,
                        damageTaken: 0
                    }
                },
                {
                    actorId: 'team-b-0',
                    actorTeamId: 'team-b',
                    state: {
                        positionX: 20,
                        positionY: 0,
                        headingX: -1,
                        headingY: 0,
                        velocityX: 0,
                        velocityY: 0,
                        hp: 100,
                        alive: true,
                        cooldown: 0
                    },
                    action: {
                        moveX: -0.2,
                        moveY: 0,
                        aimX: -10,
                        aimY: 0,
                        fire: 0
                    },
                    reason: {
                        source: 'policy',
                        label: 'hold',
                        evidence: { focus: 'center' }
                    },
                    measurements: {
                        nearestAllyDistance: 3,
                        nearestEnemyDistance: 14,
                        targetId: 'team-a-0',
                        targetDistance: 14,
                        damageDealt: 0,
                        damageTaken: 0
                    }
                },
                {
                    actorId: 'team-b-1',
                    actorTeamId: 'team-b',
                    state: {
                        positionX: 18,
                        positionY: -8,
                        headingX: -1,
                        headingY: 0,
                        velocityX: 0,
                        velocityY: 0,
                        hp: 100,
                        alive: true,
                        cooldown: 0
                    },
                    action: {
                        moveX: -0.1,
                        moveY: -0.1,
                        aimX: -12,
                        aimY: -8,
                        fire: 0
                    },
                    reason: {
                        source: 'policy',
                        label: 'support',
                        evidence: { focus: 'team-b-0' }
                    },
                    measurements: {
                        nearestAllyDistance: 2,
                        nearestEnemyDistance: 18,
                        targetId: 'team-a-1',
                        targetDistance: 18,
                        damageDealt: 0,
                        damageTaken: 0
                    }
                }
            ]
        },
        {
            step: 1,
            timeMs: 250,
            environment: {
                width: 100,
                height: 100,
                activeProjectileCount: 0
            },
            players: [
                {
                    actorId: 'team-a-0',
                    actorTeamId: 'team-a',
                    state: {
                        positionX: -10,
                        positionY: 2,
                        headingX: 1,
                        headingY: 0,
                        velocityX: 0.1,
                        velocityY: 0.1,
                        hp: 100,
                        alive: true,
                        cooldown: 0
                    },
                    action: {
                        moveX: 0.1,
                        moveY: 0.1,
                        aimX: 8,
                        aimY: 2,
                        fire: 1
                    },
                    reason: {
                        source: 'policy',
                        label: 'engage',
                        evidence: { targetId: 'team-b-0' }
                    },
                    measurements: {
                        nearestAllyDistance: 3,
                        nearestEnemyDistance: 10,
                        targetId: 'team-b-0',
                        targetDistance: 10,
                        damageDealt: 1,
                        damageTaken: 0
                    }
                },
                {
                    actorId: 'team-a-1',
                    actorTeamId: 'team-a',
                    state: {
                        positionX: -12,
                        positionY: 6,
                        headingX: 1,
                        headingY: 0,
                        velocityX: 0.1,
                        velocityY: 0,
                        hp: 98,
                        alive: true,
                        cooldown: 1
                    },
                    action: {
                        moveX: 0,
                        moveY: 0.1,
                        aimX: 10,
                        aimY: 6,
                        fire: 0
                    },
                    reason: {
                        source: 'policy',
                        label: 'protect',
                        evidence: { focus: 'team-a-0' }
                    },
                    measurements: {
                        nearestAllyDistance: 2,
                        nearestEnemyDistance: 12,
                        targetId: 'team-b-1',
                        targetDistance: 12,
                        damageDealt: 0,
                        damageTaken: 2
                    }
                },
                {
                    actorId: 'team-b-0',
                    actorTeamId: 'team-b',
                    state: {
                        positionX: 10,
                        positionY: 2,
                        headingX: -1,
                        headingY: 0,
                        velocityX: -0.1,
                        velocityY: 0.1,
                        hp: 99,
                        alive: true,
                        cooldown: 0
                    },
                    action: {
                        moveX: -0.1,
                        moveY: 0.1,
                        aimX: -8,
                        aimY: 2,
                        fire: 1
                    },
                    reason: {
                        source: 'policy',
                        label: 'counter',
                        evidence: { targetId: 'team-a-0' }
                    },
                    measurements: {
                        nearestAllyDistance: 3,
                        nearestEnemyDistance: 10,
                        targetId: 'team-a-0',
                        targetDistance: 10,
                        damageDealt: 1,
                        damageTaken: 0
                    }
                },
                {
                    actorId: 'team-b-1',
                    actorTeamId: 'team-b',
                    state: {
                        positionX: 12,
                        positionY: -6,
                        headingX: -1,
                        headingY: 0,
                        velocityX: -0.1,
                        velocityY: 0,
                        hp: 100,
                        alive: true,
                        cooldown: 1
                    },
                    action: {
                        moveX: 0,
                        moveY: -0.1,
                        aimX: -10,
                        aimY: -6,
                        fire: 0
                    },
                    reason: {
                        source: 'policy',
                        label: 'cover',
                        evidence: { focus: 'team-b-0' }
                    },
                    measurements: {
                        nearestAllyDistance: 2,
                        nearestEnemyDistance: 12,
                        targetId: 'team-a-1',
                        targetDistance: 12,
                        damageDealt: 0,
                        damageTaken: 1
                    }
                }
            ]
        }
    ],
    result: {
        winnerTeamId: null,
        endStep: 1,
        endReason: 'ongoing'
    }
};
