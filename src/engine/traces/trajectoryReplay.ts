import { PlayerStepRecord, ReplayState, Trajectory, TrajectoryStep } from './trace';

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

function validatePlayerState(state: unknown, path: string, errors: string[]): void {
    if (!isObject(state)) {
        errors.push(`${path}.state is required`);
        return;
    }

    if (!isNumber(state.positionX)) errors.push(`${path}.state.positionX is required`);
    if (!isNumber(state.positionY)) errors.push(`${path}.state.positionY is required`);
    if (!isNumber(state.hp)) errors.push(`${path}.state.hp is required`);
    if (!isBoolean(state.alive)) errors.push(`${path}.state.alive is required`);
    if ('headingX' in state && state.headingX !== undefined && !isNumber(state.headingX)) errors.push(`${path}.state.headingX must be a number`);
    if ('headingY' in state && state.headingY !== undefined && !isNumber(state.headingY)) errors.push(`${path}.state.headingY must be a number`);
    if ('velocityX' in state && state.velocityX !== undefined && !isNumber(state.velocityX)) errors.push(`${path}.state.velocityX must be a number`);
    if ('velocityY' in state && state.velocityY !== undefined && !isNumber(state.velocityY)) errors.push(`${path}.state.velocityY must be a number`);
}

function validatePlayerAction(action: unknown, path: string, errors: string[]): void {
    if (!isObject(action)) {
        errors.push(`${path}.action is required`);
        return;
    }

    if (!isNumber(action.moveX)) errors.push(`${path}.action.moveX is required`);
    if (!isNumber(action.moveY)) errors.push(`${path}.action.moveY is required`);
    if (!isNumber(action.aimX)) errors.push(`${path}.action.aimX is required`);
    if (!isNumber(action.aimY)) errors.push(`${path}.action.aimY is required`);
    if (!isNumber(action.fire)) errors.push(`${path}.action.fire is required`);
}

function validateDecisionReason(reason: unknown, path: string, errors: string[]): void {
    if (!isObject(reason)) {
        errors.push(`${path}.reason is required`);
        return;
    }

    if (!isString(reason.source)) errors.push(`${path}.reason.source is required`);
    if (!isString(reason.label)) errors.push(`${path}.reason.label is required`);
    if (!isObject(reason.evidence)) errors.push(`${path}.reason.evidence is required`);
}

function validateMeasurements(measurements: unknown, path: string, errors: string[]): void {
    if (!isObject(measurements)) {
        errors.push(`${path}.measurements is required`);
        return;
    }
}

function validateTrajectorySteps(raw: { steps: unknown[] }, errors: string[]): void {
    raw.steps.forEach((step, stepIndex) => {
        const stepPath = `steps[${stepIndex}]`;
        if (!isObject(step)) {
            errors.push(`${stepPath} must be an object`);
            return;
        }
        if (typeof step.step !== 'number') errors.push(`${stepPath}.step is required`);
        if (!Array.isArray(step.players)) {
            errors.push(`${stepPath}.players must be an array`);
            return;
        }
        step.players.forEach((player, playerIndex) => {
            const path = `${stepPath}.players[${playerIndex}]`;
            if (!isObject(player)) {
                errors.push(`${path} must be an object`);
                return;
            }
            if (!isString(player.actorId)) errors.push(`${path}.actorId is required`);
            if (!isString(player.actorTeamId)) errors.push(`${path}.actorTeamId is required`);
            validatePlayerState(player.state, path, errors);
            validatePlayerAction(player.action, path, errors);
            validateDecisionReason(player.reason, path, errors);
            validateMeasurements(player.measurements, path, errors);
        });
    });
}

export function validateReplayableTrajectory(raw: unknown): string[] {
    if (!isObject(raw)) return ['trajectory must be an object'];

    const errors: string[] = [];
    if (!isString(raw.schemaVersion)) errors.push('schemaVersion is required');
    if (!isObject(raw.initialState)) errors.push('initialState is required');
    if (!Array.isArray(raw.steps)) {
        errors.push('steps must be an array');
        return errors;
    }

    validateTrajectorySteps(raw as { steps: unknown[] }, errors);
    return errors;
}

export function validateTrajectory(raw: unknown): string[] {
    return validateReplayableTrajectory(raw);
}

export function loadTrajectoryFromObject(raw: unknown): Trajectory {
    const errors = validateReplayableTrajectory(raw);
    if (errors.length) throw new Error(`Invalid trajectory: ${errors.join('; ')}`);
    return JSON.parse(JSON.stringify(raw)) as Trajectory;
}

export function getStepFrame(trajectory: Trajectory, stepIndex: number): TrajectoryStep | null {
    return trajectory.steps[stepIndex] || null;
}

export function getPlayerFrame(stepFrame: TrajectoryStep, playerId: string): PlayerStepRecord | null {
    return stepFrame.players.find((player) => player.actorId === playerId) || null;
}

export function createReplayStateFromStep(stepFrame: TrajectoryStep): ReplayState {
    const missingState = stepFrame.players.find((player) => !player.state);
    if (missingState) throw new Error(`Cannot create replay state: player ${missingState.actorId} is missing state`);

    return {
        step: stepFrame.step,
        timeMs: stepFrame.timeMs,
        environment: JSON.parse(JSON.stringify(stepFrame.environment || { width: 0, height: 0 })),
        players: stepFrame.players.map((player) => ({
            id: player.actorId,
            teamId: player.actorTeamId,
            positionX: player.state!.positionX,
            positionY: player.state!.positionY,
            headingX: player.state!.headingX,
            headingY: player.state!.headingY,
            hp: player.state!.hp,
            alive: player.state!.alive,
            lastAction: JSON.parse(JSON.stringify(player.action)),
            reason: JSON.parse(JSON.stringify(player.reason)),
            measurements: JSON.parse(JSON.stringify(player.measurements))
        }))
    };
}
