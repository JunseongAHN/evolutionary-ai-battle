import { Trajectory } from './trace';
import { loadTrajectoryFromObject, validateReplayableTrajectory } from './trajectoryReplay';

export function trajectoryToJsonString(trajectory: Trajectory): string {
    return JSON.stringify(trajectory, null, 2);
}

export function parseTrajectoryJsonString(jsonString: string): Trajectory {
    let parsed: unknown;

    try {
        parsed = JSON.parse(jsonString);
    } catch (error) {
        const message = error instanceof Error ? error.message : 'Unknown parse error';
        throw new Error(`Failed to parse trajectory JSON: ${message}`);
    }

    const errors = validateReplayableTrajectory(parsed);
    if (errors.length) {
        throw new Error(`Invalid trajectory JSON: ${errors.join('; ')}`);
    }

    return loadTrajectoryFromObject(parsed);
}
