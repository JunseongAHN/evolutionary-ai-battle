import Trace from './trace';

// Collects simulation events without interpreting or scoring them.
export default class TraceRecorder {
    events: unknown[];

    constructor() {
        this.events = [];
    }

    record(event: unknown) {
        this.events.push(event);
    }

    createTrace() {
        return new Trace(this.events.slice());
    }
}
