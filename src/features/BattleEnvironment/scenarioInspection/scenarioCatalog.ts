import { ScenarioDefinition } from './scenarioGt';

export const scenarioCatalog: ScenarioDefinition[] = [
    {
        scenarioId: 'direct_enemy_contact',
        split: 'train',
        grid: { width: 20, height: 20 },
        thresholds: {
            lowHp: 35,
            enemyThreatRange: 5,
            supportRange: 4,
            isolationRange: 6,
            responseWindowSteps: 5
        },
        initialState: {
            players: [
                { id: 'team-a-0', teamId: 'team-a', positionX: 5, positionY: 5, headingX: 1, headingY: 0, hp: 100, alive: true },
                { id: 'team-a-1', teamId: 'team-a', positionX: 4, positionY: 7, headingX: 1, headingY: 0, hp: 100, alive: true },
                { id: 'team-b-0', teamId: 'team-b', positionX: 7, positionY: 5, headingX: -1, headingY: 0, hp: 100, alive: true },
                { id: 'team-b-1', teamId: 'team-b', positionX: 16, positionY: 16, headingX: -1, headingY: 0, hp: 100, alive: true }
            ]
        },
        gt: {
            intent: 'attack_nearest_enemy',
            moveTargetId: 'team-b-0',
            aimTargetId: 'team-b-0',
            avoidTargetId: null,
            fireIntent: true,
            expectedReasonLabels: ['attack', 'engage', 'pressure']
        },
        metricDirection: {
            damageDealt: 'increase',
            survivalSteps: 'maintain'
        }
    },
    {
        scenarioId: 'teammate_under_pressure',
        split: 'train',
        grid: { width: 20, height: 20 },
        thresholds: {
            lowHp: 35,
            enemyThreatRange: 5,
            supportRange: 4,
            isolationRange: 6,
            responseWindowSteps: 5
        },
        initialState: {
            players: [
                { id: 'team-a-0', teamId: 'team-a', positionX: 3, positionY: 3, headingX: 1, headingY: 0, hp: 100, alive: true },
                { id: 'team-a-1', teamId: 'team-a', positionX: 6, positionY: 3, headingX: 1, headingY: 0, hp: 28, alive: true },
                { id: 'team-b-0', teamId: 'team-b', positionX: 8, positionY: 3, headingX: -1, headingY: 0, hp: 100, alive: true },
                { id: 'team-b-1', teamId: 'team-b', positionX: 16, positionY: 14, headingX: -1, headingY: 0, hp: 100, alive: true }
            ]
        },
        gt: {
            intent: 'support_teammate_under_pressure',
            moveTargetId: 'team-a-1',
            aimTargetId: 'team-b-0',
            avoidTargetId: null,
            fireIntent: true,
            expectedReasonLabels: ['support', 'assist', 'protect']
        },
        metricDirection: {
            teammateResponseRate: 'increase',
            isolationRate: 'decrease'
        }
    },
    {
        scenarioId: 'isolated_teammate',
        split: 'train',
        grid: { width: 20, height: 20 },
        thresholds: {
            lowHp: 35,
            enemyThreatRange: 5,
            supportRange: 4,
            isolationRange: 6,
            responseWindowSteps: 5
        },
        initialState: {
            players: [
                { id: 'team-a-0', teamId: 'team-a', positionX: 2, positionY: 2, headingX: 1, headingY: 0, hp: 100, alive: true },
                { id: 'team-a-1', teamId: 'team-a', positionX: 16, positionY: 16, headingX: 1, headingY: 0, hp: 100, alive: true },
                { id: 'team-b-0', teamId: 'team-b', positionX: 18, positionY: 2, headingX: -1, headingY: 0, hp: 100, alive: true },
                { id: 'team-b-1', teamId: 'team-b', positionX: 14, positionY: 18, headingX: -1, headingY: 0, hp: 100, alive: true }
            ]
        },
        gt: {
            intent: 'reduce_isolation',
            moveTargetId: 'team-a-1',
            aimTargetId: null,
            avoidTargetId: null,
            fireIntent: false,
            expectedReasonLabels: ['regroup', 'reduce_isolation', 'support']
        },
        metricDirection: {
            isolationRate: 'decrease',
            survivalSteps: 'maintain'
        }
    },
    {
        scenarioId: 'self_low_hp',
        split: 'train',
        grid: { width: 20, height: 20 },
        thresholds: {
            lowHp: 35,
            enemyThreatRange: 5,
            supportRange: 4,
            isolationRange: 6,
            responseWindowSteps: 5
        },
        initialState: {
            players: [
                { id: 'team-a-0', teamId: 'team-a', positionX: 3, positionY: 3, headingX: 1, headingY: 0, hp: 22, alive: true },
                { id: 'team-a-1', teamId: 'team-a', positionX: 6, positionY: 6, headingX: 1, headingY: 0, hp: 100, alive: true },
                { id: 'team-b-0', teamId: 'team-b', positionX: 5, positionY: 3, headingX: -1, headingY: 0, hp: 100, alive: true },
                { id: 'team-b-1', teamId: 'team-b', positionX: 15, positionY: 15, headingX: -1, headingY: 0, hp: 100, alive: true }
            ]
        },
        gt: {
            intent: 'retreat_toward_ally',
            moveTargetId: 'team-a-1',
            aimTargetId: 'team-b-0',
            avoidTargetId: 'team-b-0',
            fireIntent: true,
            expectedReasonLabels: ['retreat', 'survive', 'regroup']
        },
        metricDirection: {
            damageTaken: 'not_increase_much',
            survivalSteps: 'maintain',
            isolationRate: 'not_increase_much'
        }
    }
];

export const scenarioById = Object.fromEntries(
    scenarioCatalog.map((scenario) => [scenario.scenarioId, scenario])
);

export const scenarioOptions = scenarioCatalog.map((scenario) => ({
    value: scenario.scenarioId,
    label: scenario.scenarioId
}));
