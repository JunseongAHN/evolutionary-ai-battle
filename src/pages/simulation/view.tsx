// @ts-nocheck
import React, { useMemo, useRef, useState } from 'react';
import ScoreBoard from '../../features/BattleEnvironment/components/ScoreBoard';
import ScenarioInspectionPanel from '../../features/BattleEnvironment/components/ScenarioInspectionPanel';
import { scenarioById, scenarioOptions } from '../../features/BattleEnvironment/scenarioInspection/scenarioCatalog';
import { getStepFrame } from '../../engine/traces/trajectoryReplay';
import { actorIdForBot, useSimulation } from './useSimulation';
import { runLinearIntentScenarioDemo } from '../../engine/policies/linearIntent/linearIntentScenarioDemo';
import { LINEAR_INTENT_MODEL_URL } from '../../engine/policies/linearIntent/linearIntentTypes';
import { loadLinearIntentModelFromUrl } from '../../engine/policies/linearIntent/linearIntentModel';
import { BOT_POLICY_TYPES, formatBotPolicy } from './botPolicyConfig';

function getTeamLetter(teamIndex) {
    return String.fromCharCode(65 + teamIndex);
}

function parseActorId(actorId) {
    const match = String(actorId || '').match(/^team-(\d+)-(\d+)$/);
    if (!match) {
        return null;
    }
    return {
        teamIndex: Number(match[1]),
        slotIndex: Number(match[2])
    };
}

function teamLabelFromActorId(actorId) {
    const parsed = parseActorId(actorId);
    if (!parsed) {
        return 'Team';
    }
    return `team${getTeamLetter(parsed.teamIndex)}`;
}

function botLabelFromActorId(actorId) {
    const parsed = parseActorId(actorId);
    if (!parsed) {
        return String(actorId || 'Bot');
    }
    const teamLetter = getTeamLetter(parsed.teamIndex);
    return `bot${teamLetter}-${parsed.slotIndex + 1}`;
}

function botLabel(botId, actorIdByBotId = {}) {
    return botLabelFromActorId(actorIdByBotId[botId]) || `Bot ${botId}`;
}

function getPolicyTeams(simulation) {
    const rows = simulation.botIds.map((botId) => {
        const actorId = simulation.actorIdByBotId[botId];
        const parsed = parseActorId(actorId) || { teamIndex: botId - 1, slotIndex: 0 };
        return {
            botId,
            actorId,
            teamIndex: parsed.teamIndex,
            teamName: teamLabelFromActorId(actorId),
            botName: botLabelFromActorId(actorId),
            policy: simulation.botPolicyConfig[botId] || BOT_POLICY_TYPES.genome
        };
    });

    return rows.reduce((teams, row) => {
        const current = teams.find((team) => team.teamIndex === row.teamIndex);
        if (current) {
            current.players.push(row);
            return teams;
        }
        return [
            ...teams,
            {
                teamIndex: row.teamIndex,
                teamName: row.teamName,
                players: [row]
            }
        ];
    }, []);
}

function TeamPolicyCards({ simulation }) {
    const teams = getPolicyTeams(simulation);

    const setTeamPolicy = (team, policy) => {
        team.players.forEach((player) => {
            simulation.setBotPolicyForBot(player.botId, policy);
        });
    };

    return (
        <div className="team-policy-grid">
            {teams.map((team) => (
                <article className="selected-bot-panel team-policy-card" key={team.teamIndex}>
                    <header className="team-policy-card-header">
                        <h3>{team.teamName}</h3>
                        <span>{team.players.length} bot{team.players.length === 1 ? '' : 's'}</span>
                    </header>
                    <div className="replay-controls">
                        <button type="button" onClick={() => setTeamPolicy(team, BOT_POLICY_TYPES.genome)}>
                            {team.teamName} Genome
                        </button>
                        <button type="button" onClick={() => setTeamPolicy(team, BOT_POLICY_TYPES.linearIntent)}>
                            {team.teamName} Linear
                        </button>
                        <button type="button" onClick={() => setTeamPolicy(team, BOT_POLICY_TYPES.none)}>
                            {team.teamName} None
                        </button>
                    </div>
                    <div className="score-board-table-wrap">
                        <table>
                            <thead>
                                <tr>
                                    <th>Bot</th>
                                    <th>Player ID</th>
                                    <th>Policy</th>
                                </tr>
                            </thead>
                            <tbody>
                                {team.players.map((player) => (
                                    <tr key={player.botId}>
                                        <td>{player.botName}</td>
                                        <td>{player.actorId}</td>
                                        <td>
                                            <select
                                                value={player.policy}
                                                onChange={(event) => simulation.setBotPolicyForBot(player.botId, event.target.value)}
                                            >
                                                <option value={BOT_POLICY_TYPES.genome}>Genome</option>
                                                <option value={BOT_POLICY_TYPES.linearIntent}>Linear Intent</option>
                                                <option value={BOT_POLICY_TYPES.random}>Random</option>
                                                <option value={BOT_POLICY_TYPES.userControlled}>User Controlled</option>
                                                <option value={BOT_POLICY_TYPES.none}>None</option>
                                            </select>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </article>
            ))}
        </div>
    );
}

function StageABaselinePanel({ simulation }) {
    const rows = simulation.baselineRows || [];
    const summaries = simulation.baselineSummaries || [];

    return (
        <section className="score-board">
            <h2>Stage A Solo Combat Baseline</h2>
            <div className="replay-controls">
                <label>
                    <span>Seed</span>
                    <input
                        min="1"
                        step="1"
                        type="number"
                        value={simulation.baselineSeed}
                        onChange={(event) => simulation.setBaselineSeed(Number(event.target.value))}
                        disabled={simulation.isBattleRunning}
                    />
                </label>
                <button
                    type="button"
                    onClick={simulation.runRandomSoloBaseline}
                    disabled={simulation.isBattleRunning}
                >
                    Run Random Solo
                </button>
                <button
                    type="button"
                    onClick={simulation.runUserControlledSoloBaseline}
                    disabled={simulation.isBattleRunning}
                >
                    Run User-Controlled Solo
                </button>
                <button
                    type="button"
                    onClick={simulation.downloadBaselineResults}
                    disabled={!rows.length}
                >
                    Export Baselines JSON
                </button>
            </div>
            <p>User control: WASD or arrow keys move, Q/E rotate, Space fires.</p>
            <div className="score-board-table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>Run</th>
                            <th>Seed</th>
                            <th>Policy</th>
                            <th>Player</th>
                            <th>Dmg Dealt</th>
                            <th>Dmg Taken</th>
                            <th>Survival</th>
                            <th>Alive End</th>
                            <th>Kills</th>
                            <th>Deaths</th>
                            <th>Waste Fire</th>
                            <th>Coop</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows.map((row) => (
                            <tr key={`${row.runId}-${row.playerId}`}>
                                <td>{row.runLabel || row.runId}</td>
                                <td>{row.seed}</td>
                                <td>{row.policyType}</td>
                                <td>{row.playerId}</td>
                                <td>{row.damageDealt}</td>
                                <td>{row.damageTaken}</td>
                                <td>{row.survivalSteps}</td>
                                <td>{row.aliveAtEnd ? 'yes' : 'no'}</td>
                                <td>{row.kills ?? 'N/A'}</td>
                                <td>{row.deaths ?? 'N/A'}</td>
                                <td>{row.wastefulFireCount ?? 'N/A'}</td>
                                <td>{row.cooperation?.applicable ? 'applicable' : 'N/A'}</td>
                            </tr>
                        ))}
                        {!rows.length && (
                            <tr>
                                <td colSpan="12">No Stage A baseline runs yet.</td>
                            </tr>
                        )}
                    </tbody>
                </table>
            </div>
            {!!summaries.length && (
                <div className="score-board-table-wrap">
                    <table>
                        <thead>
                            <tr>
                                <th>Policy</th>
                                <th>Runs</th>
                                <th>Avg Dmg Dealt</th>
                                <th>Avg Dmg Taken</th>
                                <th>Avg Survival</th>
                                <th>Survival Rate</th>
                                <th>Seeds</th>
                            </tr>
                        </thead>
                        <tbody>
                            {summaries.map((summary) => (
                                <tr key={summary.policyType}>
                                    <td>{summary.policyType}</td>
                                    <td>{summary.runCount}</td>
                                    <td>{summary.avgDamageDealt}</td>
                                    <td>{summary.avgDamageTaken}</td>
                                    <td>{summary.avgSurvivalSteps}</td>
                                    <td>{Math.round(summary.survivalRate * 100)}%</td>
                                    <td>{summary.seeds.join(', ')}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </section>
    );
}

function BattleEnvironment({ botIds, actorIdByBotId = {}, prefix, title, showBrains = true, children = null }) {
    return (
        <section className="battle-environment">
            <h2>{title}</h2>
            <canvas className="battleground-canvas" id={`${prefix}-battleground`} width="1000" height="500" />
            {children}
            {showBrains && (
                <div className="brain-grid">
                    {botIds.map((botId) => (
                        <div className="brain-board" key={botId}>
                            <h4>{botLabel(botId, actorIdByBotId)} Brain</h4>
                            <canvas id={`${prefix}-bot${botId}brain`} width="400" height="400" />
                        </div>
                    ))}
                </div>
            )}
        </section>
    );
}

function LiveBattleSummary({ simulation }) {
    const bots = simulation.latestLiveFrame?.bots || [];
    const result = simulation.latestBattleResult;
    const playerRows = Object.values(simulation.currentEvaluation?.players || {});
    const teamRows = Object.values(simulation.currentEvaluation?.teams || {});

    return (
        <section className="score-board live-battle-summary">
            <h3>{simulation.isBattleRunning ? 'Live HP' : 'Battle Summary'}</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(120px, 1fr))', gap: '0.75rem' }}>
                {simulation.botIds.map((botId) => {
                    const bot = bots.find((candidate) => candidate.id === botId);
                    return (
                        <article className="selected-bot-panel" key={botId}>
                            <strong>{botLabel(botId, simulation.actorIdByBotId)}</strong>
                            <p>HP: {bot?.lives ?? '-'}</p>
                            <p>Status: {bot ? (bot.lives > 0 ? 'Alive' : 'Eliminated') : 'Waiting'}</p>
                        </article>
                    );
                })}
            </div>
            {!simulation.isBattleRunning && result && (
                <>
                    <p>
                        Winner: {result.winnerTeamId || 'Draw'} | Reason: {result.endReason} | Duration: {result.totalTime.toFixed(1)}s
                    </p>
                    <div className="score-board-table-wrap">
                        <table>
                            <thead>
                                <tr>
                                    <th>Team</th>
                                    <th>Damage Dealt</th>
                                    <th>Damage Taken</th>
                                    <th>Avg Eval</th>
                                </tr>
                            </thead>
                            <tbody>
                                {teamRows.map((team) => (
                                    <tr key={team.teamId}>
                                        <td>{team.teamId}</td>
                                        <td>{team.damageDealt}</td>
                                        <td>{team.damageTaken}</td>
                                        <td>{team.avgEvaluationScore}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                    <div className="score-board-table-wrap">
                        <table>
                            <thead>
                                <tr>
                                    <th>Player</th>
                                    <th>Damage Dealt</th>
                                    <th>Damage Taken</th>
                                    <th>Eval Score</th>
                                </tr>
                            </thead>
                            <tbody>
                                {playerRows.map((player) => (
                                    <tr key={player.playerId}>
                                        <td>{player.playerId}</td>
                                        <td>{player.player.damageDealt}</td>
                                        <td>{player.player.damageTaken}</td>
                                        <td>{player.evaluationScore}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </>
            )}
            {!simulation.isBattleRunning && !result && <p>Run a battle to see its summary.</p>}
        </section>
    );
}

function BotDetails({ simulation, replay = false }) {
    const stats = simulation.selectedStats;
    const replayPlayer = simulation.selectedReplayPlayer;
    const liveBot = simulation.latestLiveFrame?.bots?.find((bot) => bot.id === simulation.selectedBotId);
    const liveDecision = liveBot?.decision;
    const selectedPolicy = simulation.botPolicyConfig?.[simulation.selectedBotId] || BOT_POLICY_TYPES.genome;

    return (
        <section className="species-stats">
            <p>Generation: {simulation.generation}</p>
            <p>Max Fitness: {simulation.maxFitness}</p>
            <p>Mode: {replay ? 'Replay' : 'Live'}</p>
            <p>Selected Policy: {formatBotPolicy(selectedPolicy)}</p>
            <h3>{replay ? 'Replay Competitor Details' : 'Live Competitor Details'}</h3>
            <div className="bot-button-row">
                {simulation.botIds.map((botId) => (
                    <button
                        className={simulation.selectedBotId === botId ? 'active' : ''}
                        key={botId}
                        onClick={() => simulation.setSelectedBotId(botId)}
                        type="button"
                    >
                        {botLabel(botId, simulation.actorIdByBotId)}
                    </button>
                ))}
            </div>
            <div className="selected-bot-panel">
                <h4>{botLabel(simulation.selectedBotId, simulation.actorIdByBotId)}</h4>
                <p>Previous Fitness: {stats.lastFitness ?? 'NEW'}</p>
                <p>Current Fitness: {stats.fitness ?? 'NEW'}</p>
                {!replay && selectedPolicy === BOT_POLICY_TYPES.linearIntent && liveDecision && (
                    <>
                        <p>Policy: {liveBot.policyMode}</p>
                        <p>Intent: {liveDecision.intent}</p>
                        <p>Reason: {liveDecision.reason?.label}</p>
                        <p>Action: moveX={liveDecision.action?.moveX}, moveY={liveDecision.action?.moveY}, aimX={liveDecision.action?.aimX}, aimY={liveDecision.action?.aimY}, fire={liveDecision.action?.fire}</p>
                        <p>Scores: {formatList(liveDecision.scores || [])}</p>
                        <p>Probabilities: {formatList(liveDecision.probabilities || [])}</p>
                    </>
                )}
                {replay && replayPlayer && (
                    <>
                        <p>Replay Step: {simulation.replayStepIndex}</p>
                        <p>
                            Action: moveX={replayPlayer.action.moveX}, moveY={replayPlayer.action.moveY},
                            aimX={replayPlayer.action.aimX}, aimY={replayPlayer.action.aimY},
                            fire={replayPlayer.action.fire}
                        </p>
                        <p>Reason: {replayPlayer.reason.label}</p>
                        <p>HP: {replayPlayer.state.hp}</p>
                        <p>Weapon Cooldown: {replayPlayer.state.weaponCooldownSteps}</p>
                        <p>Position: {replayPlayer.state.positionX}, {replayPlayer.state.positionY}</p>
                        <p>Can Fire: {String(replayPlayer.measurements.canFire)}</p>
                        <p>Did Fire: {String(replayPlayer.measurements.didFire)}</p>
                        <p>Nearest Ally: {replayPlayer.measurements.nearestAllyDistance}</p>
                        <p>Nearest Enemy: {replayPlayer.measurements.nearestEnemyDistance}</p>
                    </>
                )}
            </div>
        </section>
    );
}

function formatList(values) {
    return values.map((value) => Number(value).toFixed(3)).join(', ');
}

function LinearIntentModelDemo() {
    const [model, setModel] = useState(null);
    const [loadStatus, setLoadStatus] = useState('Model not loaded');
    const [loadError, setLoadError] = useState('');
    const [results, setResults] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [isRunning, setIsRunning] = useState(false);

    const handleLoadModel = async () => {
        setIsLoading(true);
        setLoadError('');
        try {
            const loadedModel = await loadLinearIntentModelFromUrl(LINEAR_INTENT_MODEL_URL);
            setModel(loadedModel);
            setLoadStatus(`Loaded ${loadedModel.schemaVersion}`);
        } catch (error) {
            setModel(null);
            setLoadStatus('Model load failed');
            setLoadError(error instanceof Error ? error.message : 'Failed to load linear intent model');
        } finally {
            setIsLoading(false);
        }
    };

    const handleRunDemo = () => {
        if (!model) {
            return;
        }

        setIsRunning(true);
        try {
            setResults(runLinearIntentScenarioDemo(model));
            setLoadError('');
        } catch (error) {
            setLoadError(error instanceof Error ? error.message : 'Failed to run linear intent demo');
        } finally {
            setIsRunning(false);
        }
    };

    return (
        <section className="linear-intent-demo">
            <h2>Linear Intent Model Demo</h2>
            <p>Status: {loadStatus}</p>
            <p>Asset URL: <code>{LINEAR_INTENT_MODEL_URL}</code></p>
            {model && (
                <div className="linear-intent-demo-meta">
                    <p>Schema: {model.schemaVersion}</p>
                    <p>Model Type: {model.modelType}</p>
                    <p>Train Accuracy: {model.training?.trainAccuracy ?? 'n/a'}</p>
                    <p>Eval Accuracy: {model.training?.evalAccuracy ?? 'n/a'}</p>
                </div>
            )}
            {loadError && <p className="replay-error">{loadError}</p>}
            <div className="replay-controls">
                <button onClick={handleLoadModel} disabled={isLoading}>
                    {isLoading ? 'Loading Model...' : 'Load Linear Intent Model'}
                </button>
                <button onClick={handleRunDemo} disabled={!model || isRunning}>
                    {isRunning ? 'Running Demo...' : 'Run Scenario Demo'}
                </button>
            </div>
            <div className="linear-intent-demo-results">
                {results.map((result) => (
                    <article className="linear-intent-demo-card" key={result.scenarioId}>
                        <h3>{result.scenarioId}</h3>
                        <p>Expected: {result.expectedIntent}</p>
                        <p>Predicted: {result.predictedIntent}</p>
                        <p>Result: {result.passed ? 'PASS' : 'FAIL'}</p>
                        <p>Action: moveX={result.action.moveX.toFixed(3)}, moveY={result.action.moveY.toFixed(3)}, aimX={result.action.aimX.toFixed(3)}, aimY={result.action.aimY.toFixed(3)}, fire={result.action.fire}</p>
                        <p>Scores: {formatList(result.scores)}</p>
                        <p>Probabilities: {formatList(result.probabilities)}</p>
                    </article>
                ))}
            </div>
        </section>
    );
}

export function SimulationPage() {
    const simulation = useSimulation();
    const trajectoryFileInputRef = useRef(null);
    const [selectedScenarioId, setSelectedScenarioId] = useState('');
    const selectedScenario = selectedScenarioId ? scenarioById[selectedScenarioId] : null;
    const replayStepFrame = useMemo(
        () => simulation.replayTrajectory ? getStepFrame(simulation.replayTrajectory, simulation.replayStepIndex) : null,
        [simulation.replayStepIndex, simulation.replayTrajectory]
    );

    return (
        <>
            <h1>Co-Player Evolution Harness</h1>
            <section id="battle-select">
                <h2>Select Species</h2>
                {simulation.loading && <p>Loading species...</p>}
                {!simulation.loading && !simulation.species.length && (
                    <p>No trained species are available. Run <code>npm run train</code>.</p>
                )}
                <div className="species-list">
                    {simulation.species.map((species) => (
                        <button className="species-card" key={species.id} onClick={() => simulation.selectSpecies(species.id)}>
                            <span className="container">
                                <strong>{species.id}</strong>
                                <span>Total Generations: {species.latestGeneration}</span>
                                <span>Last Update: {species.lastUpdate}</span>
                            </span>
                        </button>
                    ))}
                </div>
            </section>
            <section className="linear-intent-controls">
                <h2>Battle Policy</h2>
                <div className="replay-controls">
                    <label>
                        <span>Mode</span>
                        <select
                            value={simulation.battleConfig.mode}
                            onChange={(event) => simulation.setBattleConfig({
                                mode: event.target.value,
                                teamCount: simulation.battleConfig.teamCount
                            })}
                            disabled={simulation.isBattleRunning}
                        >
                            <option value="duo">Duo</option>
                            <option value="solo">Solo</option>
                        </select>
                    </label>
                    {simulation.battleConfig.mode === 'duo' && (
                        <label>
                            <span>Teams</span>
                            <input
                                min="2"
                                step="1"
                                type="number"
                                value={simulation.battleConfig.teamCount}
                                onChange={(event) => simulation.setBattleConfig({
                                    mode: simulation.battleConfig.mode,
                                    teamCount: Number(event.target.value)
                                })}
                                disabled={simulation.isBattleRunning}
                            />
                        </label>
                    )}
                </div>
                <p>Current Assignment Summary:</p>
                <TeamPolicyCards simulation={simulation} />
                <div className="replay-controls">
                    <button onClick={simulation.setAllGenomePolicies} type="button">
                        All Genome
                    </button>
                    <button onClick={simulation.setAllLinearPolicies} type="button">
                        All Linear
                    </button>
                    <button onClick={simulation.setAllNonePolicies} type="button">
                        All None
                    </button>
                </div>
                <div className="replay-controls">
                    <button onClick={simulation.loadLinearIntentModel} disabled={simulation.linearModelLoading}>
                        {simulation.linearModelLoading ? 'Loading Linear Model...' : 'Load Linear Intent Model'}
                    </button>
                    <button
                        onClick={simulation.runBattleOnce}
                        disabled={simulation.isBattleRunning || (simulation.runBattleRequiresLinearModel && !simulation.linearIntentModel)}
                    >
                        {simulation.isBattleRunning ? 'Battle Running...' : 'Run Battle'}
                    </button>
                </div>
                <p>Model Status: {simulation.linearModelLoadStatus}</p>
                {simulation.linearModelError && <p className="replay-error">{simulation.linearModelError}</p>}
                {simulation.linearIntentModel && (
                    <div>
                        <p>Schema: {simulation.linearIntentModel.schemaVersion}</p>
                        <p>Train Accuracy: {simulation.linearIntentModel.training?.trainAccuracy ?? 'n/a'}</p>
                        <p>Eval Accuracy: {simulation.linearIntentModel.training?.evalAccuracy ?? 'n/a'}</p>
                    </div>
                )}
            </section>
            <main id="battle">
                <div className="battle-layout">
                    <BattleEnvironment
                        actorIdByBotId={simulation.actorIdByBotId}
                        botIds={simulation.botIds}
                        prefix="live"
                        title="Live Battle"
                        showBrains={false}
                    >
                        <LiveBattleSummary simulation={simulation} />
                    </BattleEnvironment>
                    <BotDetails simulation={simulation} />
                </div>
                <section className="replay-controls">
                    <label className="replay-scenario-select">
                        <span>Select Scenario</span>
                        <select value={selectedScenarioId} onChange={(event) => setSelectedScenarioId(event.target.value)}>
                            <option value="">No scenario selected</option>
                            {scenarioOptions.map((scenario) => (
                                <option key={scenario.value} value={scenario.value}>
                                    {scenario.label}
                                </option>
                            ))}
                        </select>
                    </label>
                    <button onClick={simulation.downloadLatestTrajectory} disabled={!simulation.latestTrajectory}>
                        Download Trajectory
                    </button>

                    <input
                        ref={trajectoryFileInputRef}
                        accept="application/json,.json"
                        hidden
                        type="file"
                        onChange={(event) => {
                            const file = event.target.files?.[0];
                            if (file) {
                                simulation.loadTrajectoryFile(file);
                            }
                            event.target.value = '';
                        }}
                    />
                </section>
                <section className="replay-section">
                    <div className="battle-layout">
                        <BattleEnvironment
                            actorIdByBotId={simulation.actorIdByBotId}
                            botIds={simulation.botIds}
                            prefix="replay"
                            title="Replay Viewer"
                        />
                        <BotDetails simulation={simulation} replay />
                    </div>
                    <section className="replay-controls">
                        {simulation.replayError && <p className="replay-error">{simulation.replayError}</p>}
                        <button onClick={simulation.loadLatestTrajectoryForReplay} disabled={!simulation.latestTrajectory}>
                            Replay Latest Battle
                        </button>
                        <button
                            onClick={() => trajectoryFileInputRef.current?.click()}
                            type="button"
                        >
                            Import Trajectory JSON
                        </button>
                        <button onClick={simulation.goToPreviousReplayStep} disabled={!simulation.replayTrajectory?.steps.length}>
                            Prev
                        </button>
                        <button onClick={simulation.goToNextReplayStep} disabled={!simulation.replayTrajectory?.steps.length}>
                            Next
                        </button>
                        <button onClick={simulation.resetReplay} disabled={!simulation.replayTrajectory?.steps.length}>
                            Reset
                        </button>
                        <button onClick={simulation.toggleReplayPlayback} disabled={!simulation.replayTrajectory?.steps.length}>
                            {simulation.replayAutoPlay ? 'Pause' : 'Play'}
                        </button>
                        <label className="replay-slider">
                            <span>Step {simulation.replayStepIndex} / {simulation.replayMaxStep}</span>
                            <input
                                type="range"
                                min="0"
                                max={simulation.replayMaxStep}
                                step="1"
                                value={simulation.replayStepIndex}
                                disabled={!simulation.replayTrajectory?.steps.length}
                                onChange={(event) => simulation.setReplayStepIndex(Number(event.target.value))}
                            />
                        </label>
                    </section>
                    <ScenarioInspectionPanel
                        trajectory={simulation.replayTrajectory}
                        scenario={selectedScenario}
                        replayStepIndex={simulation.replayStepIndex}
                        selectedActorId={simulation.actorIdByBotId[simulation.selectedBotId] || actorIdForBot(simulation.selectedBotId)}
                        replayStepFrame={replayStepFrame}
                    />
                </section>
                <ScoreBoard
                    accumulatedEvaluation={simulation.accumulatedEvaluation}
                    currentEvaluation={simulation.currentEvaluation}
                    onResetScores={simulation.resetScores}
                />
                <StageABaselinePanel simulation={simulation} />
            </main>
            <LinearIntentModelDemo />
        </>
    );
}
