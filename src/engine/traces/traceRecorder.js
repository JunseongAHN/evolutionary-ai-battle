import Trace from './trace';

// Collects simulation events without interpreting or scoring them.
export default class TraceRecorder {
    constructor() {
        this.events = [];
    }

    record(event) {
        this.events.push(event);
    }

    createTrace() {
        return new Trace(this.events.slice());
    }
}
