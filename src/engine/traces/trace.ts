// Represents a recorded sequence of simulation events for later evaluation.
export default class Trace {
    events: unknown[];

    constructor(events: unknown[] = []) {
        this.events = events;
    }
}
