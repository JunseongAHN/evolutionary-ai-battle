// @ts-nocheck
import React from 'react';
import { computeActionAlignment } from '../scenarioInspection/actionAlignment';
import { computeFiveStepMetricDirection } from '../scenarioInspection/metricDirection';

function formatAlignment(value) {
    if (value === null || value === undefined) return 'n/a';
    return value.toFixed(2);
}

function formatTarget(value) {
    return value || 'none';
}

export default function ScenarioInspectionPanel({
    trajectory,
    scenario,
    replayStepIndex,
    selectedActorId,
    replayStepFrame
}) {
    if (!scenario) {
        return (
            <section className="scenario-inspection-panel">
                <h3>Scenario Inspection</h3>
                <p>No scenario selected.</p>
            </section>
        );
    }

    if (!replayStepFrame) {
        return (
            <section className="scenario-inspection-panel">
                <h3>Scenario Inspection</h3>
                <p>No replay step available.</p>
            </section>
        );
    }

    const actorRecord = replayStepFrame.players.find((player) => player.actorId === selectedActorId) || null;
    if (!actorRecord) {
        return (
            <section className="scenario-inspection-panel">
                <h3>Scenario Inspection</h3>
                <p>Selected player {selectedActorId} is not present at replay step {replayStepIndex}.</p>
            </section>
        );
    }

    const alignment = computeActionAlignment(actorRecord, scenario, replayStepFrame);
    const metricDirection = computeFiveStepMetricDirection(trajectory, scenario, replayStepIndex, selectedActorId);

    return (
        <section className="scenario-inspection-panel">
            <h3>Scenario Inspection</h3>
            <p>Scenario ID: {scenario.scenarioId}</p>
            <p>GT Intent: {alignment.intent}</p>
            <p>Move Target: {formatTarget(scenario.gt.moveTargetId)}</p>
            <p>Aim Target: {formatTarget(scenario.gt.aimTargetId)}</p>
            <p>Avoid Target: {formatTarget(scenario.gt.avoidTargetId)}</p>
            <p>Fire Intent: {String(scenario.gt.fireIntent)}</p>
            <p>Expected Reasons: {scenario.gt.expectedReasonLabels.join(', ') || 'none'}</p>
            <hr />
            <p>Actual Action: moveX={actorRecord.action.moveX}, moveY={actorRecord.action.moveY}, aimX={actorRecord.action.aimX}, aimY={actorRecord.action.aimY}, fire={actorRecord.action.fire}</p>
            <p>Actual Reason: {actorRecord.reason?.label || 'none'}</p>
            <p>Move Alignment: {formatAlignment(alignment.moveAlignment)}</p>
            <p>Aim Alignment: {formatAlignment(alignment.aimAlignment)}</p>
            <p>Fire Match: {String(alignment.fireMatch)}</p>
            <p>Reason Match: {String(alignment.reasonMatch)}</p>
            <p>Overall Pass: {String(alignment.passed)}</p>
            <hr />
            <p>5-Step Isolation Trend: {metricDirection.isolationTrend}</p>
            <p>Damage Dealt Delta: {metricDirection.damageDealtDelta}</p>
            <p>Damage Taken Delta: {metricDirection.damageTakenDelta}</p>
            <p>Teammate Response Triggered: {String(metricDirection.teammateResponseTriggered)}</p>
            {metricDirection.notes?.length > 0 && (
                <ul>
                    {metricDirection.notes.map((note) => (
                        <li key={note}>{note}</li>
                    ))}
                </ul>
            )}
            {alignment.details?.notes?.length > 0 && (
                <ul>
                    {alignment.details.notes.map((note) => (
                        <li key={note}>{note}</li>
                    ))}
                </ul>
            )}
        </section>
    );
}
