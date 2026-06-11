import { ScenarioDefinition, ScenarioGroundTruth } from './scenarioGt';

const FIRE_THRESHOLD = 0.5;

export interface ActionAlignmentDetails {
    notes: string[];
    moveTargetDistance?: number;
    aimTargetDistance?: number;
    avoidTargetDistance?: number;
    fireIntent?: boolean;
    fireAttempted?: boolean;
    canFire?: boolean;
    didFire?: boolean;
    avoidAlignment?: number | null;
    reasonLabel?: string | null;
    expectedReasonLabels?: string[];
}

export interface ActionAlignmentResult {
    intent: string;
    moveAlignment: number | null;
    aimAlignment: number | null;
    fireMatch: boolean;
    reasonMatch: boolean;
    passed: boolean;
    details: ActionAlignmentDetails;
}

function vectorLength(x: number, y: number): number {
    return Math.hypot(x, y);
}

function normalizeVector(x: number, y: number): { x: number; y: number } {
    const length = vectorLength(x, y);
    if (!length) {
        return { x: 0, y: 0 };
    }

    return { x: x / length, y: y / length };
}

function dotProduct(a: { x: number; y: number }, b: { x: number; y: number }): number {
    return (a.x * b.x) + (a.y * b.y);
}

function findStepPlayer(stepFrame: { players: Array<{ actorId: string; state?: { positionX: number; positionY: number } }> }, playerId: string) {
    return stepFrame.players.find((player) => player.actorId === playerId) || null;
}

function getDirectionToTarget(actorPosition: { positionX: number; positionY: number }, targetPosition: { positionX: number; positionY: number }) {
    return normalizeVector(
        targetPosition.positionX - actorPosition.positionX,
        targetPosition.positionY - actorPosition.positionY
    );
}

function getAlignment(actual: { x: number; y: number }, target: { x: number; y: number }): number {
    if (!vectorLength(actual.x, actual.y) || !vectorLength(target.x, target.y)) {
        return 0;
    }

    return dotProduct(normalizeVector(actual.x, actual.y), normalizeVector(target.x, target.y));
}

function findTargetPosition(stepFrame: { players: Array<{ actorId: string; state?: { positionX: number; positionY: number } }> }, targetId: string | null | undefined) {
    if (!targetId) {
        return null;
    }

    const targetPlayer = findStepPlayer(stepFrame, targetId);
    if (!targetPlayer?.state) {
        return null;
    }

    return targetPlayer.state;
}

function includesReasonLabel(actualLabel: unknown, expectedReasonLabels: string[]): boolean {
    if (typeof actualLabel !== 'string' || !expectedReasonLabels.length) {
        return false;
    }

    const lowerActual = actualLabel.toLowerCase();
    return expectedReasonLabels.some((label) => lowerActual.includes(String(label).toLowerCase()));
}

export function computeActionAlignment(actorRecord: any, scenario: ScenarioDefinition, stepFrame: any): ActionAlignmentResult {
    const gt: ScenarioGroundTruth = scenario.gt;
    const actorState = actorRecord?.state;
    const actorAction = actorRecord?.action || {};
    const actorMeasurements = actorRecord?.measurements || {};
    const actorReason = actorRecord?.reason || {};

    const details: ActionAlignmentDetails = {
        notes: []
    };

    if (!actorState || !stepFrame?.players) {
        details.notes.push('actor record or step frame missing');
        return {
            intent: gt.intent,
            moveAlignment: null,
            aimAlignment: null,
            fireMatch: false,
            reasonMatch: false,
            passed: false,
            details
        };
    }

    const actorPosition = {
        positionX: actorState.positionX,
        positionY: actorState.positionY
    };
    const moveVector = {
        x: typeof actorAction.moveX === 'number' ? actorAction.moveX : 0,
        y: typeof actorAction.moveY === 'number' ? actorAction.moveY : 0
    };
    const aimVector = {
        x: typeof actorAction.aimX === 'number' ? actorAction.aimX : 0,
        y: typeof actorAction.aimY === 'number' ? actorAction.aimY : 0
    };

    let moveAlignment: number | null = null;
    let aimAlignment: number | null = null;
    let avoidAlignment: number | null = null;

    const moveTargetState = findTargetPosition(stepFrame, gt.moveTargetId);
    if (moveTargetState) {
        moveAlignment = getAlignment(moveVector, getDirectionToTarget(actorPosition, moveTargetState));
        details.moveTargetDistance = Math.hypot(
            moveTargetState.positionX - actorPosition.positionX,
            moveTargetState.positionY - actorPosition.positionY
        );
    } else if (gt.moveTargetId) {
        details.notes.push(`move target ${gt.moveTargetId} is missing from this step`);
    }

    const aimTargetState = findTargetPosition(stepFrame, gt.aimTargetId);
    if (aimTargetState) {
        aimAlignment = getAlignment(aimVector, getDirectionToTarget(actorPosition, aimTargetState));
        details.aimTargetDistance = Math.hypot(
            aimTargetState.positionX - actorPosition.positionX,
            aimTargetState.positionY - actorPosition.positionY
        );
    } else if (gt.aimTargetId) {
        details.notes.push(`aim target ${gt.aimTargetId} is missing from this step`);
    }

    const avoidTargetState = findTargetPosition(stepFrame, gt.avoidTargetId);
    if (avoidTargetState) {
        avoidAlignment = getAlignment(moveVector, getDirectionToTarget(actorPosition, avoidTargetState));
        details.avoidTargetDistance = Math.hypot(
            avoidTargetState.positionX - actorPosition.positionX,
            avoidTargetState.positionY - actorPosition.positionY
        );
    } else if (gt.avoidTargetId) {
        details.notes.push(`avoid target ${gt.avoidTargetId} is missing from this step`);
    }

    const fireIntent = gt.fireIntent;
    const fireAttempted = typeof actorAction.fire === 'number' ? actorAction.fire > FIRE_THRESHOLD : false;
    const canFire = actorMeasurements.canFire === true;
    const didFire = actorMeasurements.didFire === true;

    let fireMatch = false;
    if (fireIntent) {
        fireMatch = fireAttempted;
        if (fireAttempted && !canFire) {
            details.notes.push('fire was intended but weapon was unavailable');
        }
    } else {
        fireMatch = typeof actorAction.fire === 'number'
            ? actorAction.fire <= FIRE_THRESHOLD || didFire === false
            : didFire === false;
    }

    const reasonLabels = Array.isArray(gt.expectedReasonLabels) ? gt.expectedReasonLabels : [];
    const reasonMatch = reasonLabels.length > 0 && includesReasonLabel(actorReason.label, reasonLabels);

    const movePass = moveAlignment === null || moveAlignment > 0.5;
    const aimPass = aimAlignment === null || aimAlignment > 0.5;
    const avoidPass = avoidAlignment === null || avoidAlignment <= 0;
    const reasonPass = reasonLabels.length > 0 ? reasonMatch : false;
    const passed = movePass && aimPass && avoidPass && fireMatch && reasonPass;

    return {
        intent: gt.intent,
        moveAlignment,
        aimAlignment,
        fireMatch,
        reasonMatch,
        passed,
        details: {
            ...details,
            fireIntent,
            fireAttempted,
            canFire,
            didFire,
            avoidAlignment,
            reasonLabel: actorReason.label || null,
            expectedReasonLabels: reasonLabels
        }
    };
}
