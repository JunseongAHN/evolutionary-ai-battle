import { PlayerStepRecord, ReplayState, Trajectory, TrajectoryStep } from './trace';

function isObject(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null;
}

export function validateTrajectory(raw: unknown): string[] {
    if (!isObject(raw)) return ['trajectory must be an object'];

    const errors: string[] = [];
    if (typeof raw.schemaVersion !== 'string' || !raw.schemaVersion) errors.push('schemaVersion is required');
    if (!isObject(raw.initialState)) errors.push('initialState is required');
    if (!Array.isArray(raw.steps)) {
        errors.push('steps must be an array');
        return errors;
    }

    raw.steps.forEach((step, stepIndex) => {
        if (!isObject(step)) {
            errors.push(`steps[${stepIndex}] must be an object`);
            return;
        }
        if (!Array.isArray(step.players)) {
            errors.push(`steps[${stepIndex}].players must be an array`);
            return;
        }
        step.players.forEach((player, playerIndex) => {
            const path = `steps[${stepIndex}].players[${playerIndex}]`;
            if (!isObject(player)) {
                errors.push(`${path} must be an object`);
                return;
            }
            if (typeof player.actorId !== 'string' || !player.actorId) errors.push(`${path}.actorId is required`);
            if (!isObject(player.state)) errors.push(`${path}.state is required`);
            if (!isObject(player.action)) errors.push(`${path}.action is required`);
        });
    });
    return errors;
}

export function loadTrajectoryFromObject(raw: unknown): Trajectory {
    const errors = validateTrajectory(raw);
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
