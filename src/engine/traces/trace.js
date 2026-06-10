// Represents a recorded sequence of simulation events for later evaluation.
export default class Trace {
    constructor(events = []) {
        this.events = events;
    }
}
