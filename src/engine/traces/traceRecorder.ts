import {
    createEmptyTrajectory,
    createDefaultDecisionReason,
    PlayerStepRecord,
    Trajectory,
    TrajectoryMetadata,
    TrajectoryResult,
    TrajectoryStep
} from './trace';

export default class TraceRecorder {
    private trajectory: Trajectory | null;

    constructor() {
        this.trajectory = null;
    }

    startTrajectory(metadata: TrajectoryMetadata) {
        this.trajectory = createEmptyTrajectory(metadata);
    }

    recordStep(step: number, timeMs: number, playerRecords: PlayerStepRecord[]) {
        if (!this.trajectory) return;

        const stepRecord: TrajectoryStep = {
            step,
            timeMs,
            players: playerRecords.map((playerRecord) => ({
                ...playerRecord,
                reason: playerRecord.reason || createDefaultDecisionReason()
            }))
        };

        this.trajectory.steps.push(stepRecord);
    }

    finishTrajectory(result: TrajectoryResult) {
        if (!this.trajectory) return;
        this.trajectory.result = result;
    }

    getTrajectory() {
        return this.trajectory ? {
            ...this.trajectory,
            steps: this.trajectory.steps.map((step) => ({
                ...step,
                players: step.players.map((player) => ({
                    ...player,
                    reason: player.reason || createDefaultDecisionReason()
                }))
            }))
        } : null;
    }

    reset() {
        this.trajectory = null;
    }
}
