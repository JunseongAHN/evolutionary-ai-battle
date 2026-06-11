import { ScenarioDefinition } from './scenarioGt';

type StepLike = {
    players?: Array<{
        actorId: string;
        actorTeamId?: string;
        state?: {
            positionX: number;
            positionY: number;
            hp?: number;
            alive?: boolean;
        };
        measurements?: {
            nearestAllyDistance?: number;
            damageDealt?: number;
            damageTaken?: number;
        };
    }>;
};

function isNumber(value: unknown): value is number {
    return typeof value === 'number' && Number.isFinite(value);
}

function getStepPlayer(step: StepLike | null | undefined, actorId: string) {
    if (!step?.players) return null;
    return step.players.find((player) => player.actorId === actorId) || null;
}

function getNearestAllyDistance(step: StepLike | null | undefined, actorId: string): number | null {
    const actor = getStepPlayer(step, actorId);
    if (!actor) return null;

    if (isNumber(actor.measurements?.nearestAllyDistance)) {
        return actor.measurements.nearestAllyDistance;
    }

    if (!actor.state || !step?.players) {
        return null;
    }

    const sameTeam = step.players.filter((player) => player.actorId !== actorId && player.actorTeamId === actor.actorTeamId && player.state?.alive !== false);
    if (!sameTeam.length) {
        return null;
    }

    return sameTeam.reduce((closest, candidate) => {
        const distance = Math.hypot(
            (candidate.state?.positionX || 0) - actor.state!.positionX,
            (candidate.state?.positionY || 0) - actor.state!.positionY
        );
        return closest === null || distance < closest ? distance : closest;
    }, null as number | null);
}

function getTargetDistance(step: StepLike | null | undefined, actorId: string, targetId: string | null | undefined): number | null {
    if (!targetId || !step?.players) {
        return null;
    }

    const actor = getStepPlayer(step, actorId);
    const target = getStepPlayer(step, targetId);
    if (!actor?.state || !target?.state) {
        return null;
    }

    return Math.hypot(
        target.state.positionX - actor.state.positionX,
        target.state.positionY - actor.state.positionY
    );
}

function isSupportLikeIntent(intent: string): boolean {
    const lowered = intent.toLowerCase();
    return lowered.includes('support') || lowered.includes('retreat') || lowered.includes('reduce_isolation');
}

export function computeFiveStepMetricDirection(
    trajectory: { steps?: StepLike[] } | null | undefined,
    scenario: ScenarioDefinition,
    startStep: number,
    actorId: string
) {
    const steps = Array.isArray(trajectory?.steps) ? trajectory!.steps : [];
    const windowSteps = Math.min(5, Math.max(steps.length - startStep - 1, 0));
    const windowEnd = Math.min(startStep + windowSteps, Math.max(steps.length - 1, startStep));
    const window = steps.slice(startStep, windowEnd + 1);
    const notes: string[] = [];

    if (!window.length) {
        return {
            windowSteps: 5,
            isolationTrend: 'unknown' as const,
            damageDealtDelta: 0,
            damageTakenDelta: 0,
            teammateResponseTriggered: false,
            notes: ['no trajectory steps available in the inspection window']
        };
    }

    const startDistance = getNearestAllyDistance(window[0], actorId);
    const endDistance = getNearestAllyDistance(window[window.length - 1], actorId);

    let isolationTrend: 'improved' | 'worsened' | 'unchanged' | 'unknown' = 'unknown';
    if (isNumber(startDistance) && isNumber(endDistance)) {
        const delta = endDistance - startDistance;
        if (delta < -0.5) {
            isolationTrend = 'improved';
        } else if (delta > 0.5) {
            isolationTrend = 'worsened';
        } else {
            isolationTrend = 'unchanged';
        }
    } else {
        notes.push('nearest ally distance unavailable for part of the window');
    }

    let damageDealtDelta = 0;
    let damageTakenDelta = 0;
    let teammateResponseTriggered = false;

    window.forEach((step, index) => {
        const player = getStepPlayer(step, actorId);
        if (!player) {
            return;
        }

        damageDealtDelta += player.measurements?.damageDealt || 0;
        damageTakenDelta += player.measurements?.damageTaken || 0;

        if (isSupportLikeIntent(scenario.gt.intent)) {
            const targetDistance = getTargetDistance(step, actorId, scenario.gt.moveTargetId);
            const supportRange = scenario.thresholds?.supportRange ?? 4;
            if (isNumber(targetDistance) && targetDistance <= supportRange) {
                teammateResponseTriggered = true;
            }
            if (index > 0) {
                const previousTargetDistance = getTargetDistance(window[index - 1], actorId, scenario.gt.moveTargetId);
                if (isNumber(targetDistance) && isNumber(previousTargetDistance) && targetDistance < previousTargetDistance) {
                    teammateResponseTriggered = true;
                }
            }
        }
    });

    if (isSupportLikeIntent(scenario.gt.intent) && !teammateResponseTriggered) {
        notes.push('support-like intent did not visibly close distance to the teammate target');
    }

    return {
        windowSteps: 5,
        isolationTrend,
        damageDealtDelta,
        damageTakenDelta,
        teammateResponseTriggered,
        notes
    };
}
