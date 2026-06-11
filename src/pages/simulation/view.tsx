// @ts-nocheck
import React, { useRef } from 'react';
import ScoreBoard from '../../features/BattleEnvironment/components/ScoreBoard';
import { useSimulation } from './useSimulation';

function botLabel(botId) {
    return botId <= 2 ? `Team A - Bot ${botId}` : `Team B - Bot ${botId}`;
}

function BattleEnvironment({ botIds, prefix, title }) {
    return (
        <section className="battle-environment">
            <h2>{title}</h2>
            <canvas className="battleground-canvas" id={`${prefix}-battleground`} width="1000" height="500" />
            <div className="brain-grid">
                {botIds.map((botId) => (
                    <div className="brain-board" key={botId}>
                        <h4>{botLabel(botId)} Brain</h4>
                        <canvas id={`${prefix}-bot${botId}brain`} width="400" height="400" />
                    </div>
                ))}
            </div>
        </section>
    );
}

function BotDetails({ simulation, replay = false }) {
    const stats = simulation.selectedStats;
    const replayPlayer = simulation.selectedReplayPlayer;

    return (
        <section className="species-stats">
            <p>Generation: {simulation.generation}</p>
            <p>Max Fitness: {simulation.maxFitness}</p>
            <p>Mode: {replay ? 'Replay' : 'Live'}</p>
            <h3>{replay ? 'Replay Competitor Details' : 'Live Competitor Details'}</h3>
            <div className="bot-button-row">
                {simulation.botIds.map((botId) => (
                    <button
                        className={simulation.selectedBotId === botId ? 'active' : ''}
                        key={botId}
                        onClick={() => simulation.setSelectedBotId(botId)}
                        type="button"
                    >
                        Bot {botId}
                    </button>
                ))}
            </div>
            <div className="selected-bot-panel">
                <h4>{botLabel(simulation.selectedBotId)}</h4>
                <p>Previous Fitness: {stats.lastFitness ?? 'NEW'}</p>
                <p>Current Fitness: {stats.fitness ?? 'NEW'}</p>
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
                        <p>Position: {replayPlayer.state.positionX}, {replayPlayer.state.positionY}</p>
                        <p>Nearest Ally: {replayPlayer.measurements.nearestAllyDistance}</p>
                        <p>Nearest Enemy: {replayPlayer.measurements.nearestEnemyDistance}</p>
                    </>
                )}
            </div>
        </section>
    );
}

export function SimulationPage() {
    const simulation = useSimulation();
    const trajectoryFileInputRef = useRef(null);

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
            <main id="battle">
                <div className="battle-layout">
                    <BattleEnvironment botIds={simulation.botIds} prefix="live" title="Live Battle" />
                    <BotDetails simulation={simulation} />
                </div>
                <section className="replay-controls">
                    <button onClick={simulation.runBattleOnce} disabled={simulation.isBattleRunning}>
                        {simulation.isBattleRunning ? 'Battle Running...' : 'Run Battle'}
                    </button>
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
                        <BattleEnvironment botIds={simulation.botIds} prefix="replay" title="Replay Viewer" />
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
                </section>
                <ScoreBoard
                    accumulatedEvaluation={simulation.accumulatedEvaluation}
                    currentEvaluation={simulation.currentEvaluation}
                    onResetScores={simulation.resetScores}
                />
            </main>
        </>
    );
}
