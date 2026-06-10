// @ts-nocheck
/**
 * This is the main file for the browser based trainer. It imports the battleground and bots and 
 * play bots against each other one round at a time. 
 **/
import Vue from 'vue/dist/vue.js'
import Bot from '../engine/agents/bot'
import Battleground from '../engine/simulation/battleground'
import Trainer from '../engine/training/trainer'
import Genome from '../engine/evolution/genome'
import { createReplayStateFromStep, getPlayerFrame, getStepFrame, loadTrajectoryFromObject } from '../engine/traces/trajectoryReplay'
import BattleEnvironmentController from '../features/BattleEnvironment/controller'
import { renderSimulationPage } from '../pages/simulation/view'

const trainer = new Trainer();
const geneticBattleViewer = new BattleEnvironmentController();
const trajectoryBattleViewer = new BattleEnvironmentController();
renderSimulationPage();

function getCanvasIds() {
    return {
        battle: 'battleground',
        bot: (botId) => {
            const normalizedBotId = String(botId);
            if (normalizedBotId === 'team-a-0') return 'bot1brain';
            if (normalizedBotId === 'team-a-1') return 'bot2brain';
            if (normalizedBotId === 'team-b-0') return 'bot3brain';
            if (normalizedBotId === 'team-b-1') return 'bot4brain';
            return `bot${normalizedBotId}brain`;
        }
    };
}

var app = new Vue({
    el: '#evolutionary-ai-battle',
    data() {
        return {
            loading: true,
            species: [],
            speciesData: null,
            latestTrajectory: null,
            replayTrajectory: null,
            replayFrame: null,
            isReplaying: false,
            replayTimer: null,
            replayStepIndex: 0,
            replayAutoPlay: false,
            displayMode: 'live',
            latestLiveFrame: null,
            selectedBotId: 1,
            bot1Stats: {},
            bot2Stats: {},
            bot3Stats: {},
            bot4Stats: {}
        }
    },
    methods: {
        botLabel(botId) {
            const labels = {
                1: 'Team A - Bot 1',
                2: 'Team A - Bot 3',
                3: 'Team B - Bot 2',
                4: 'Team B - Bot 4'
            };
            return labels[botId] || `Bot ${botId}`;
        },
        selectBot(botId) {
            this.selectedBotId = botId;
        },
        async selectSpecies(speciesId) {
            this.loading = true;
            const response = await fetch(`/species/${speciesId}/latest`);
            const speciesData = await response.json();
            this.speciesData = speciesData;
            this.loading = false;
            Vue.nextTick(() => {
                battle.call(this, speciesData);
            });
        },
        async refreshSpecies() {
            try {
                const response = await fetch('/species');
                const species = await response.json();
                this.species = formatSpecies(sortSpecies(species));
            } catch (err) {
                console.error('Failed to refresh species list', err);
            }
        },
        stopReplayPlayback() {
            if (this.replayTimer) {
                window.clearInterval(this.replayTimer);
                this.replayTimer = null;
            }
            this.replayAutoPlay = false;
            this.isReplaying = false;
        },
        renderReplayStep(stepIndex) {
            if (!this.replayTrajectory) {
                return;
            }

            const maxStepIndex = Math.max(this.replayTrajectory.steps.length - 1, 0);
            const safeStepIndex = Math.min(Math.max(stepIndex, 0), maxStepIndex);
            const stepFrame = getStepFrame(this.replayTrajectory, safeStepIndex);
            if (!stepFrame) {
                return;
            }

            this.replayStepIndex = safeStepIndex;
            this.replayFrame = createReplayStateFromStep(stepFrame);
            this.displayMode = 'trajectory';
            trajectoryBattleViewer.renderTrajectoryBattle(getCanvasIds(), stepFrame);
        },
        startReplayPlayback() {
            if (!this.replayTrajectory || !this.replayTrajectory.steps.length) {
                return;
            }

            this.stopReplayPlayback();
            this.isReplaying = true;
            this.replayAutoPlay = true;
            this.renderReplayStep(this.replayStepIndex || 0);
            this.replayTimer = window.setInterval(() => {
                if (!this.replayTrajectory) {
                    this.stopReplayPlayback();
                    return;
                }

                const nextStep = this.replayStepIndex + 1;
                if (nextStep >= this.replayTrajectory.steps.length) {
                    this.stopReplayPlayback();
                    return;
                }

                this.renderReplayStep(nextStep);
            }, 120);
        },
        toggleReplayPlayback() {
            if (this.replayAutoPlay) {
                this.stopReplayPlayback();
                return;
            }
            this.startReplayPlayback();
        },
        loadLatestTrajectoryForReplay() {
            if (!this.latestTrajectory) {
                return;
            }

            this.stopReplayPlayback();
            this.replayTrajectory = loadTrajectoryFromObject(this.latestTrajectory);
            this.replayFrame = null;
            this.renderReplayStep(0);
        },
        showLiveBattle() {
            this.stopReplayPlayback();
            this.displayMode = 'live';
            if (this.latestLiveFrame) {
                geneticBattleViewer.renderLiveBattle(getCanvasIds(), this.latestLiveFrame.bots);
            }
        },
        replayLastTrajectory() {
            if (!this.latestTrajectory) {
                return;
            }

            this.loadLatestTrajectoryForReplay();
            this.startReplayPlayback();
        }
    },
    computed: {
        generation() {
            return this.speciesData.totalGenerations;
        },
        maxFitness() {
            return this.speciesData.maxFitness;
        },
        bot1Info() {
            return {
                lastFitness: this.bot1Stats.lastFitness || "NEW",
                fitness: this.bot1Stats.fitness
            }
        },
        bot2Info() {
            return {
                lastFitness: this.bot2Stats.lastFitness || "NEW",
                fitness: this.bot2Stats.fitness
            }
        },
        bot3Info() {
            return {
                lastFitness: this.bot3Stats.lastFitness || "NEW",
                fitness: this.bot3Stats.fitness
            }
        },
        bot4Info() {
            return {
                lastFitness: this.bot4Stats.lastFitness || "NEW",
                fitness: this.bot4Stats.fitness
            }
        },
        selectedBotInfo() {
            const statsByBot = {
                1: this.bot1Info,
                2: this.bot3Info,
                3: this.bot2Info,
                4: this.bot4Info
            };
            return statsByBot[this.selectedBotId] || this.bot1Info;
        },
        selectedReplayPlayer() {
            if (!this.replayFrame || !this.replayTrajectory) {
                return null;
            }
            const stepFrame = getStepFrame(this.replayTrajectory, this.replayStepIndex);
            if (!stepFrame) {
                return null;
            }
            const botId = `team-${this.selectedBotId <= 2 ? 'a' : 'b'}-${this.selectedBotId <= 2 ? this.selectedBotId - 1 : this.selectedBotId - 3}`;
            return getPlayerFrame(stepFrame, botId);
        },
        replayMaxStep() {
            return this.replayTrajectory ? Math.max(this.replayTrajectory.steps.length - 1, 0) : 0;
        }
    },
    watch: {
        replayStepIndex(newValue) {
            if (this.replayTrajectory) {
                this.renderReplayStep(newValue);
            }
        }
    },
    async mounted() {
        await this.refreshSpecies();
        this.loading = false;
        this.refreshTimer = window.setInterval(() => {
            this.refreshSpecies();
        }, 5000);
    },
    beforeDestroy() {
        if (this.refreshTimer) {
            window.clearInterval(this.refreshTimer);
        }
        this.stopReplayPlayback();
    }
});

function sortSpecies(speciesData) {
    return speciesData.sort((a, b) => {
        const aLastUpdate = new Date(a.lastUpdate);
        const bLastUpdate = new Date(b.lastUpdate);
        return aLastUpdate.getTime() - bLastUpdate.getTime();
    });
}

function formatSpecies(speciesData) {
    return speciesData.map((species) => {
        console.log("LastUpdate: ", species.lastUpdate);
        console.log("LastUpdate Formatted: ", new Date(species.lastUpdate).toLocaleString("en-US"));
        return {
            id: species.id,
            lastUpdate: new Date(species.lastUpdate).toLocaleString(),
            latestGeneration: species.latestGeneration
        }
    });
}

function battle(existingSpecies) {
    if (existingSpecies != null) {
        trainer.loadSpeciesFromJSON(existingSpecies);
    } else {
        trainer.createInitialSpecies();
    }
    this.replayTrajectory = null;
    this.replayFrame = null;
    this.replayStepIndex = 0;
    this.replayAutoPlay = false;
    this.isReplaying = false;
    this.displayMode = 'live';

    /* Bot 1 is the one we're training */
    const bot1 = new Bot(1, 'team-a');
    const bot1Genome = trainer.getTopGenome();
    bot1.loadGenome(bot1Genome);
    this.bot1Stats = bot1Genome.getStats();
    const bot1Teammate = new Bot(2, 'team-a');
    bot1Teammate.loadGenome(Genome.loadFromJSON(bot1Genome.serialize()));
    this.bot3Stats = bot1Genome.getStats();

    /**
     * Bot 2 picks a random algorithm initially, and after more rounds are completed
     * it starts using genomes for its movement. 
     **/
    const bot2 = new Bot(3, 'team-b');
    const bot2Genome = trainer.getTopGenome();
    bot2.loadGenome(bot2Genome);
    bot2.selectAIMethod(trainer.totalGenerations);
    this.bot2Stats = bot2Genome.getStats();
    const bot2Teammate = new Bot(4, 'team-b');
    bot2Teammate.loadGenome(Genome.loadFromJSON(bot2Genome.serialize()));
    bot2Teammate.selectAIMethod(trainer.totalGenerations);
    this.bot4Stats = bot2Genome.getStats();

    const battleground = new Battleground()
    battleground.addBots(bot1, bot1Teammate, bot2, bot2Teammate);
    battleground.start((results) => {
        this.latestTrajectory = results.trajectory;

        /* Calculate the bots fitness using the trainer method */
        const botFitness =  Trainer.calculateBotFitnessFromResults(results, trainer.totalGenerations);

        /**
         * Add the fitness for this round to the bot. Then increase played rounds as each bot only 
         * plays config.roundsPerGenome rounds in each generation.
         */
        bot1.genome.addFitness(botFitness);
        bot1.genome.totalRounds++;

        const roundsRemaining = trainer.getTotalRoundsRemaining() 
        if (roundsRemaining <= 0) {
            trainer.newGeneration();
        }

        setTimeout(() => battle.call(this));
    }, (frame) => {
        this.latestLiveFrame = frame;
        if (this.displayMode === 'live') {
            geneticBattleViewer.renderLiveBattle(getCanvasIds(), frame.bots);
        }
    });
}
