import {
    createEmptyTrajectory,
    InitialState,
    Trajectory,
    TrajectoryEvent,
    TrajectoryMetadata,
    TrajectoryResult,
    TrajectoryStep
} from './trace';

function cloneJson<T>(value: T): T {
    return JSON.parse(JSON.stringify(value)) as T;
}

export default class TraceRecorder {
    private trajectory: Trajectory | null = null;
    private pendingEvents = new Map<number, TrajectoryEvent[]>();

    startTrajectory(metadata: TrajectoryMetadata) {
        this.pendingEvents.clear();
        this.trajectory = createEmptyTrajectory(cloneJson(metadata));
    }

    recordInitialState(initialState: InitialState) {
        if (!this.trajectory) return;
        this.trajectory.initialState = cloneJson(initialState);
    }

    recordStep(stepRecord: TrajectoryStep) {
        if (!this.trajectory) return;
        const pendingEvents = this.pendingEvents.get(stepRecord.step) || [];
        this.trajectory.steps.push(cloneJson({
            ...stepRecord,
            events: [...(stepRecord.events || []), ...pendingEvents]
        }));
        this.pendingEvents.delete(stepRecord.step);
    }

    recordEvent(step: number, event: TrajectoryEvent) {
        if (!this.trajectory) return;
        const matchingStep = this.trajectory.steps.find((candidate) => candidate.step === step);
        if (matchingStep) {
            matchingStep.events = matchingStep.events || [];
            matchingStep.events.push(cloneJson(event));
            return;
        }
        const events = this.pendingEvents.get(step) || [];
        events.push(cloneJson(event));
        this.pendingEvents.set(step, events);
    }

    finishTrajectory(result: TrajectoryResult) {
        if (!this.trajectory) return;
        this.trajectory.result = cloneJson(result);
    }

    getTrajectory(): Trajectory | null {
        return this.trajectory ? cloneJson(this.trajectory) : null;
    }

    reset() {
        this.trajectory = null;
        this.pendingEvents.clear();
    }
}
