import { battleEnvironmentTemplate } from '../../features/BattleEnvironment/view';

const simulationPageTemplate = `
    <h1>Co-Player Evolution Harness</h1>
    <div id="loading" v-if="!speciesData && loading">
        <h1>Loading...</h1>
    </div>
    <div id="battle-select" v-if="!speciesData && !loading">
        <h2>Select Species</h2>
        <div v-if="!species.length">
            <p>No trained species are available yet.</p>
            <p>Run <code>npm run train</code>, then select a generated species.</p>
        </div>
        <div v-else v-for="s in species" class="species-card" @click="selectSpecies(s.id)">
            <div class="container">
                <h2>{{s.id}}</h2>
                <p>Total Generations: {{s.latestGeneration}}</p>
                <p>Last Update: {{s.lastUpdate}}</p>
            </div>
        </div>
    </div>
    <main id="battle" v-if="speciesData">
        ${battleEnvironmentTemplate}
        <section class="replay-controls">
            <button type="button" @click="loadLatestTrajectoryForReplay" :disabled="!latestTrajectory">
                Load Last Trajectory
            </button>
            <button type="button" @click="toggleReplayPlayback" :disabled="!replayTrajectory || !replayTrajectory.steps.length">
                {{ replayAutoPlay ? 'Pause Replay' : 'Play Replay' }}
            </button>
            <button type="button" @click="showLiveBattle" :disabled="displayMode === 'live'">
                Show Live Battle
            </button>
            <label class="replay-slider">
                <span>Step {{ replayStepIndex }} / {{ replayMaxStep }}</span>
                <input type="range" min="0" :max="replayMaxStep" step="1" v-model.number="replayStepIndex"
                    :disabled="!replayTrajectory || !replayTrajectory.steps.length">
            </label>
        </section>
        <section id="species-stats">
            <p>Display: {{ displayMode }}</p>
            <p>Generation: {{ generation }}</p>
            <p>Max Fitness: {{ maxFitness }}</p>
            <h3>Competitor Details</h3>
            <div class="bot-button-row">
                <button v-for="botId in [1, 2, 3, 4]" type="button" @click="selectBot(botId)"
                    :class="{ active: selectedBotId === botId }">Bot {{ botId }}</button>
            </div>
            <div class="selected-bot-panel">
                <h4>{{ botLabel(selectedBotId) }}</h4>
                <p>Previous Fitness: {{ selectedBotInfo.lastFitness }}</p>
                <p>Current Fitness: {{ selectedBotInfo.fitness }}</p>
                <div v-if="displayMode === 'trajectory' && selectedReplayPlayer">
                    <p>Replay Step: {{ replayStepIndex }}</p>
                    <p>Action: dx={{ selectedReplayPlayer.action.dx }}, dy={{ selectedReplayPlayer.action.dy }},
                        dh={{ selectedReplayPlayer.action.dh }}, shoot={{ selectedReplayPlayer.action.ds }}</p>
                    <p>Reason: {{ selectedReplayPlayer.reason.label }}</p>
                    <p>HP: {{ selectedReplayPlayer.measurements.hp }}</p>
                    <p>Position: {{ selectedReplayPlayer.measurements.positionX }}, {{ selectedReplayPlayer.measurements.positionY }}</p>
                    <p>Nearest Ally: {{ selectedReplayPlayer.measurements.nearestAllyDistance }}</p>
                    <p>Nearest Enemy: {{ selectedReplayPlayer.measurements.nearestEnemyDistance }}</p>
                </div>
            </div>
        </section>
    </main>
`;

export function renderSimulationPage(rootId = 'evolutionary-ai-battle') {
    const root = document.getElementById(rootId);
    if (!root) {
        throw new Error(`Simulation page root #${rootId} was not found`);
    }
    root.innerHTML = simulationPageTemplate;
}
