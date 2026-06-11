// @ts-nocheck
import React from 'react';

function formatNumber(value) {
    return typeof value === 'number' ? value.toFixed(2).replace(/\.00$/, '') : '0';
}

function formatRate(value) {
    return typeof value === 'number' ? `${Math.round(value * 100)}%` : '0%';
}

function getPlayerRows(evaluation) {
    if (!evaluation?.players) {
        return [];
    }

    return Object.values(evaluation.players).sort((a, b) => a.playerId.localeCompare(b.playerId));
}

function getTeamRows(evaluation) {
    if (!evaluation?.teams) {
        return [];
    }

    return Object.values(evaluation.teams).sort((a, b) => a.teamId.localeCompare(b.teamId));
}

function PlayerTable({ title, evaluation }) {
    const rows = getPlayerRows(evaluation);

    return (
        <section className="score-board">
            <h3>{title}</h3>
            {!rows.length && <p>No evaluation data yet.</p>}
            {rows.length > 0 && (
                <div className="score-board-table-wrap">
                    <table>
                        <thead>
                            <tr>
                                <th>Player</th>
                                <th>Team</th>
                                <th>Dmg Dealt</th>
                                <th>Dmg Taken</th>
                                <th>Survival</th>
                                <th>Pressure Events</th>
                                <th>Pressure Responses</th>
                                <th>Response Rate</th>
                                <th>Isolation Rate</th>
                                <th>Eval Score</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows.map((row) => (
                                <tr key={row.playerId}>
                                    <td>{row.playerId}</td>
                                    <td>{row.teamId}</td>
                                    <td>{formatNumber(row.player.damageDealt)}</td>
                                    <td>{formatNumber(row.player.damageTaken)}</td>
                                    <td>{formatNumber(row.player.survivalSteps)}</td>
                                    <td>{formatNumber(row.cpc.teammateUnderPressureEvents)}</td>
                                    <td>{formatNumber(row.cpc.teammateUnderPressureResponses)}</td>
                                    <td>{formatRate(row.cpc.teammateResponseRate)}</td>
                                    <td>{formatRate(row.cpc.isolationRate)}</td>
                                    <td>{formatNumber(row.evaluationScore)}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </section>
    );
}

function TeamTable({ title, evaluation }) {
    const rows = getTeamRows(evaluation);

    return (
        <section className="score-board">
            <h3>{title}</h3>
            {!rows.length && <p>No team summary yet.</p>}
            {rows.length > 0 && (
                <div className="score-board-table-wrap">
                    <table>
                        <thead>
                            <tr>
                                <th>Team</th>
                                <th>Players</th>
                                <th>Dmg Dealt</th>
                                <th>Dmg Taken</th>
                                <th>Survival</th>
                                <th>Avg Response</th>
                                <th>Avg Isolation</th>
                                <th>Avg Eval</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows.map((row) => (
                                <tr key={row.teamId}>
                                    <td>{row.teamId}</td>
                                    <td>{row.playerIds.join(', ')}</td>
                                    <td>{formatNumber(row.damageDealt)}</td>
                                    <td>{formatNumber(row.damageTaken)}</td>
                                    <td>{formatNumber(row.survivalSteps)}</td>
                                    <td>{formatRate(row.avgTeammateResponseRate)}</td>
                                    <td>{formatRate(row.avgIsolationRate)}</td>
                                    <td>{formatNumber(row.avgEvaluationScore)}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </section>
    );
}

export default function ScoreBoard({ currentEvaluation, accumulatedEvaluation, onResetScores }) {
    return (
        <section className="score-board-section">
            <div className="score-board-header">
                <h2>Evaluation Score Board</h2>
                <button type="button" onClick={onResetScores}>
                    Reset Scores
                </button>
            </div>
            <PlayerTable title="Current Run" evaluation={currentEvaluation} />
            <PlayerTable title="Accumulated Players" evaluation={accumulatedEvaluation} />
            <TeamTable title="Accumulated Teams" evaluation={accumulatedEvaluation} />
        </section>
    );
}
