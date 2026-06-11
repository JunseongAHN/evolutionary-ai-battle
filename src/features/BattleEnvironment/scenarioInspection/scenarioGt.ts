export interface ScenarioThresholds {
    lowHp?: number;
    enemyThreatRange?: number;
    supportRange?: number;
    isolationRange?: number;
    responseWindowSteps?: number;
}

export interface ScenarioPlayerState {
    id: string;
    teamId: string;
    positionX: number;
    positionY: number;
    headingX?: number;
    headingY?: number;
    hp: number;
    alive: boolean;
}

export interface ScenarioGroundTruth {
    intent: string;
    moveTargetId: string | null;
    aimTargetId: string | null;
    avoidTargetId: string | null;
    fireIntent: boolean;
    expectedReasonLabels: string[];
}

export interface ScenarioDefinition {
    scenarioId: string;
    split: string;
    grid: {
        width: number;
        height: number;
    };
    thresholds?: ScenarioThresholds;
    initialState: {
        players: ScenarioPlayerState[];
    };
    gt: ScenarioGroundTruth;
    metricDirection?: Record<string, string>;
}

function isObject(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null;
}

function isString(value: unknown): value is string {
    return typeof value === 'string' && value.length > 0;
}

function isNumber(value: unknown): value is number {
    return typeof value === 'number' && Number.isFinite(value);
}

function isBoolean(value: unknown): value is boolean {
    return typeof value === 'boolean';
}

function validateScenarioPlayer(player: unknown, path: string, errors: string[]): void {
    if (!isObject(player)) {
        errors.push(`${path} must be an object`);
        return;
    }

    if (!isString(player.id)) errors.push(`${path}.id is required`);
    if (!isString(player.teamId)) errors.push(`${path}.teamId is required`);
    if (!isNumber(player.positionX)) errors.push(`${path}.positionX is required`);
    if (!isNumber(player.positionY)) errors.push(`${path}.positionY is required`);
    if (!isNumber(player.hp)) errors.push(`${path}.hp is required`);
    if (!isBoolean(player.alive)) errors.push(`${path}.alive is required`);
}

function validateGroundTruth(gt: unknown, errors: string[]): void {
    if (!isObject(gt)) {
        errors.push('gt is required');
        return;
    }

    if (!isString(gt.intent)) errors.push('gt.intent is required');
    if (gt.moveTargetId !== null && gt.moveTargetId !== undefined && !isString(gt.moveTargetId)) {
        errors.push('gt.moveTargetId must be a string or null');
    }
    if (gt.aimTargetId !== null && gt.aimTargetId !== undefined && !isString(gt.aimTargetId)) {
        errors.push('gt.aimTargetId must be a string or null');
    }
    if (gt.avoidTargetId !== null && gt.avoidTargetId !== undefined && !isString(gt.avoidTargetId)) {
        errors.push('gt.avoidTargetId must be a string or null');
    }
    if (!isBoolean(gt.fireIntent)) errors.push('gt.fireIntent is required');
    if (!Array.isArray(gt.expectedReasonLabels)) errors.push('gt.expectedReasonLabels is required');
}

export function validateScenarioDefinition(raw: unknown): string[] {
    const errors: string[] = [];

    if (!isObject(raw)) {
        return ['scenario must be an object'];
    }

    if (!isString(raw.scenarioId)) errors.push('scenarioId is required');
    if (!isString(raw.split)) errors.push('split is required');
    if (!isObject(raw.grid)) {
        errors.push('grid is required');
    } else {
        if (!isNumber(raw.grid.width)) errors.push('grid.width is required');
        if (!isNumber(raw.grid.height)) errors.push('grid.height is required');
    }

    if (!isObject(raw.initialState)) {
        errors.push('initialState is required');
    } else if (!Array.isArray(raw.initialState.players)) {
        errors.push('initialState.players is required');
    } else {
        raw.initialState.players.forEach((player: unknown, index: number) => {
            validateScenarioPlayer(player, `initialState.players[${index}]`, errors);
        });
    }

    validateGroundTruth(raw.gt, errors);

    return errors;
}

export function loadScenarioDefinition(raw: unknown): ScenarioDefinition {
    const errors = validateScenarioDefinition(raw);
    if (errors.length) {
        throw new Error(`Invalid scenario definition: ${errors.join('; ')}`);
    }

    return JSON.parse(JSON.stringify(raw)) as ScenarioDefinition;
}

export function getScenarioPlayer(scenario: ScenarioDefinition | null | undefined, playerId: string) {
    if (!scenario || !Array.isArray(scenario.initialState?.players)) {
        return null;
    }

    return scenario.initialState.players.find((player) => player.id === playerId) || null;
}
