import {
    PlayerStepRecord,
    Trajectory,
    TrajectoryStep
} from './trace';

function isObject(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null;
}

export function loadTrajectoryFromObject(raw: unknown): Trajectory {
    if (!isObject(raw)) {
        throw new Error('Invalid trajectory: expected an object');
    }
    if (typeof raw.schemaVersion !== 'string') {
        throw new Error('Invalid trajectory: missing schemaVersion');
    }
    if (!Array.isArray(raw.steps)) {
        throw new Error('Invalid trajectory: steps must be an array');
    }

    raw.steps.forEach((step, index) => {
        if (!isObject(step)) {
            throw new Error(`Invalid trajectory: step ${index} must be an object`);
        }
        if (!Array.isArray(step.players)) {
            throw new Error(`Invalid trajectory: step ${index} players must be an array`);
        }
        step.players.forEach((player, playerIndex) => {
            if (!isObject(player)) {
                throw new Error(`Invalid trajectory: step ${index} player ${playerIndex} must be an object`);
            }
            if (typeof player.actorId !== 'string') {
                throw new Error(`Invalid trajectory: step ${index} player ${playerIndex} is missing actorId`);
            }
            if (!isObject(player.measurements)) {
                throw new Error(`Invalid trajectory: step ${index} player ${playerIndex} is missing measurements`);
            }
        });
    });

    return raw as unknown as Trajectory;
}

export function getStepFrame(trajectory: Trajectory, stepIndex: number): TrajectoryStep | null {
    return trajectory.steps[stepIndex] || null;
}

export function getPlayerFrame(stepFrame: TrajectoryStep, playerId: string): PlayerStepRecord | null {
    return stepFrame.players.find((player) => player.actorId === playerId) || null;
}

export function createReplayStateFromStep(stepFrame: TrajectoryStep) {
    return {
        step: stepFrame.step,
        timeMs: stepFrame.timeMs,
        players: stepFrame.players.map((player) => ({
            id: player.actorId,
            teamId: player.actorTeamId,
            xPos: player.measurements.positionX,
            yPos: player.measurements.positionY,
            hp: player.measurements.hp,
            action: player.action,
            reason: player.reason,
            bullets: [],
            rotation: 0
        }))
    };
}

export function saveTrajectoryToObject(trajectory: Trajectory) {
    return JSON.parse(JSON.stringify(trajectory));
}
