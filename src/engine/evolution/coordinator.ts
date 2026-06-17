// @ts-nocheck
/**
 * Coordinator
 * 
 * The coordinator is for running multiple battles in parallel in NodeJS. It uses the cluster module to spawn new 
 * threads to be able to do this. 
 * 
 * The main thread runs the trainer which keeps track of all the rounds, the AI's versing each other, and the results of
 * their battles. This main trainer thread then creates workers by forking the process. Each worker then runs a battle 
 * between two bots. 
 * 
 * The number of workers that are spawned is either the total battles happening in a round, or config.max_workers
 * 
 */
const uuid = require("uuid");
const cluster = require('cluster');
const async = require("async");
const fs = require("fs");

const isPrimaryProcess = cluster.isPrimary || cluster.isMaster;

import Bot from '../agents/bot'
import Battleground from '../simulation/battleground'
import { DEFAULT_BATTLE_CONFIG, createBattleTeams } from '../simulation/battleSetup'
import Trainer from '../training/trainer'
import Genome from './genome'
import log from '../../shared/logger'

/**
 * If you'd like to resume training from an existing species file, uncomment this line
 */
// import existingSpecies from "../../../species/SPECIESID/SPECIESID-generation-GENERATION-species.json";

/** 
 * On first run of this file cluster.isMaster is true. There is only one master process. 
 * When a worker calls fork inside that fork cluster.isMaster will be false so the battle process begins
 */
if (isPrimaryProcess) {
    trainerProcess();
} else {
    battleProcess();
}

/**
 * Initializes the trainer and species that will be evolved today. Then begins the battle.  
 */
function trainerProcess() {
    const runId = uuid.v1().toString().slice(0, 8);
    log.info("Starting Training Run ID: " + runId);
    fs.mkdirSync(`species/${runId}/`, {recursive: true});
    const trainer = new Trainer();
    if (typeof existingSpecies !== "undefined") {
        trainer.loadSpeciesFromJSON(existingSpecies);
    } else {
        trainer.createInitialSpecies();
    }
    startGenerationBattles(runId, trainer);
}

/**
 * Start all the battles in a single generation. Running them in parallel.  
 * @param {string} runId 
 * @param {Trainer} trainer 
 */
function startGenerationBattles(runId, trainer) {
    let roundsRemaining = trainer.getTotalRoundsRemaining();
    log.info(`Starting Generation ${trainer.totalGenerations}`)
    async.timesSeries(roundsRemaining, (i, next) => {
        log.debug(`Starting round ${i}...`);
        trainer.getRandomAvailableGenome((genome1) => {
            genome1.totalRounds++;
            const genome2 = trainer.getRandomGenome();
            startRound(trainer.totalGenerations, genome1, genome2, (results) => {
                roundsRemaining--;
                log.debug(`${roundsRemaining} rounds remaining`);
                if (roundsRemaining <= 0) {
                    const speciesFilePath = `species/${runId}/${runId}-generation-${trainer.totalGenerations}-species.json`
                    log.info(`Generation Complete`)
                    log.debug(`Saving all species to file ${speciesFilePath}`);
                    const serializedSpecies = trainer.serializeSpecies();
                    fs.writeFile(`${speciesFilePath}`, serializedSpecies, (err, result) => {
                        log.debug("Creating new generation")
                        trainer.newGeneration();
                        setTimeout(startGenerationBattles.bind(this, runId, trainer));
                    });
                }
            });
            next();
        });
    }, (err, result) => {
        log.debug("Finished starting rounds");
    });

}

function startRound(totalGenerations, genome1, genome2, callback) {
    let settled = false;

    const finalize = (results) => {
        if (settled) {
            return;
        }
        settled = true;
        return callback(results);
    };

    let worker;

    try {
        worker = cluster.fork();
    } catch (err) {
        log.info(`Unable to start training worker: ${err.message}`);
        return finalize(null);
    }

    worker.on('message', (msg) => {
        if (msg.type === 'results') {
            handleResults(msg.data);
        } else if (msg.type === 'error') {
            log.info(`Worker reported an error: ${msg.data.message}`);
            finalize(null);
        }
    });

    worker.on('error', (err) => {
        log.info(`Training worker failed: ${err.message}`);
        finalize(null);
    });

    worker.on('exit', (code, signal) => {
        if (!settled) {
            log.debug(`Worker exited before returning results (code=${code}, signal=${signal})`);
            finalize(null);
        }
    });

    worker.send({
        totalGenerations,
        genomes: [
            genome1.serialize(),
            genome2.serialize()
        ]
    });

    function handleResults(results) {
        if (!results) {
            genome1.addFitness(0);
            return finalize(null);
        }

        let botFitness = Trainer.calculateBotFitnessFromResults(results, totalGenerations);
        genome1.addFitness(botFitness);
        return finalize(results);
    }
}

function battleProcess() {
    process.on('uncaughtException', (err) => {
        process.send({
            type: 'error',
            data: {
                message: err.message,
                stack: err.stack
            }
        });
    });

    process.on('message', (msg) => {
        try {
            const setup = createBattleTeams(DEFAULT_BATTLE_CONFIG);
            const descriptors = setup.players;
            const bot1 = new Bot(descriptors[0].numericId, descriptors[0].teamId);
            bot1.actorId = descriptors[0].id;
            const genome1 = Genome.loadFromJSON(msg.genomes[0]);
            bot1.loadGenome(genome1);
            const bot1Teammate = new Bot(descriptors[1].numericId, descriptors[1].teamId);
            bot1Teammate.actorId = descriptors[1].id;
            bot1Teammate.loadGenome(Genome.loadFromJSON(msg.genomes[0]));

            // Bot 2 just does random stuff
            const bot2 = new Bot(descriptors[2].numericId, descriptors[2].teamId);
            bot2.actorId = descriptors[2].id;
            const genome2 = Genome.loadFromJSON(msg.genomes[1]);
            bot2.loadGenome(genome2);
            bot2.selectAIMethod(msg.totalGenerations);
            const bot2Teammate = new Bot(descriptors[3].numericId, descriptors[3].teamId);
            bot2Teammate.actorId = descriptors[3].id;
            bot2Teammate.loadGenome(Genome.loadFromJSON(msg.genomes[1]));
            bot2Teammate.selectAIMethod(msg.totalGenerations);

            const battleground = new Battleground(DEFAULT_BATTLE_CONFIG)
            battleground.addBots(bot1, bot1Teammate, bot2, bot2Teammate);
            battleground.start((results) => {
                process.send({
                    type: 'results',
                    data: results
                });
            });
        } catch (err) {
            process.send({
                type: 'error',
                data: {
                    message: err.message,
                    stack: err.stack
                }
            });
        }
    });
}

