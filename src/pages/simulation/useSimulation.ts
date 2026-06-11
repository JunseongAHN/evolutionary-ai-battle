// @ts-nocheck
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Bot from '../../engine/agents/bot';
import Genome from '../../engine/evolution/genome';
import Battleground from '../../engine/simulation/battleground';
import Trainer from '../../engine/training/trainer';
import {
    createReplayStateFromStep,
    getPlayerFrame,
    getStepFrame,
    loadTrajectoryFromObject
} from '../../engine/traces/trajectoryReplay';
import BattleEnvironmentController from '../../features/BattleEnvironment/controller';

const BOT_IDS = [1, 2, 3, 4];

function getCanvasIds(prefix) {
    return {
        battle: `${prefix}-battleground`,
        bot: (botId) => {
            const canvasByActorId = {
                'team-a-0': `${prefix}-bot1brain`,
                'team-a-1': `${prefix}-bot2brain`,
                'team-b-0': `${prefix}-bot3brain`,
                'team-b-1': `${prefix}-bot4brain`
            };
            return canvasByActorId[String(botId)] || `${prefix}-bot${botId}brain`;
        }
    };
}

function actorIdForBot(botId) {
    return botId <= 2 ? `team-a-${botId - 1}` : `team-b-${botId - 3}`;
}

function sortAndFormatSpecies(speciesData) {
    return speciesData
        .sort((a, b) => new Date(a.lastUpdate).getTime() - new Date(b.lastUpdate).getTime())
        .map((species) => ({
            id: species.id,
            lastUpdate: new Date(species.lastUpdate).toLocaleString(),
            latestGeneration: species.latestGeneration
        }));
}

export function useSimulation() {
    const trainerRef = useRef(new Trainer());
    const liveEnvironmentRef = useRef(new BattleEnvironmentController());
    const replayEnvironmentRef = useRef(new BattleEnvironmentController());
    const mountedRef = useRef(true);
    const replayTimerRef = useRef(null);

    const [loading, setLoading] = useState(true);
    const [species, setSpecies] = useState([]);
    const [speciesData, setSpeciesData] = useState(null);
    const [latestTrajectory, setLatestTrajectory] = useState(null);
    const [replayTrajectory, setReplayTrajectory] = useState(null);
    const [replayStepIndex, setReplayStepIndex] = useState(0);
    const [replayAutoPlay, setReplayAutoPlay] = useState(false);
    const [replayError, setReplayError] = useState(null);
    const [latestLiveFrame, setLatestLiveFrame] = useState(null);
    const [selectedBotId, setSelectedBotId] = useState(1);
    const [botStats, setBotStats] = useState({});

    const stopReplayPlayback = useCallback(() => {
        if (replayTimerRef.current) {
            window.clearInterval(replayTimerRef.current);
            replayTimerRef.current = null;
        }
        setReplayAutoPlay(false);
    }, []);

    const refreshSpecies = useCallback(async () => {
        try {
            const response = await fetch('/species');
            setSpecies(sortAndFormatSpecies(await response.json()));
        } catch (error) {
            console.error('Failed to refresh species list', error);
        }
    }, []);

    const runBattle = useCallback((existingSpecies = null) => {
        if (!mountedRef.current) return;

        const trainer = trainerRef.current;
        if (existingSpecies) {
            trainer.loadSpeciesFromJSON(existingSpecies);
        } else {
            trainer.createInitialSpecies();
        }

        const bot1 = new Bot(1, 'team-a');
        const bot1Genome = trainer.getTopGenome();
        bot1.loadGenome(bot1Genome);

        const bot2 = new Bot(2, 'team-a');
        bot2.loadGenome(Genome.loadFromJSON(bot1Genome.serialize()));

        const bot3 = new Bot(3, 'team-b');
        const bot3Genome = trainer.getTopGenome();
        bot3.loadGenome(bot3Genome);
        bot3.selectAIMethod(trainer.totalGenerations);

        const bot4 = new Bot(4, 'team-b');
        bot4.loadGenome(Genome.loadFromJSON(bot3Genome.serialize()));
        bot4.selectAIMethod(trainer.totalGenerations);

        setBotStats({
            1: bot1Genome.getStats(),
            2: bot1Genome.getStats(),
            3: bot3Genome.getStats(),
            4: bot3Genome.getStats()
        });

        const battleground = new Battleground();
        battleground.addBots(bot1, bot2, bot3, bot4);
        battleground.start((results) => {
            if (!mountedRef.current) return;

            setLatestTrajectory(results.trajectory);
            const fitness = Trainer.calculateBotFitnessFromResults(results, trainer.totalGenerations);
            bot1.genome.addFitness(fitness);
            bot1.genome.totalRounds++;

            if (trainer.getTotalRoundsRemaining() <= 0) {
                trainer.newGeneration();
            }
            window.setTimeout(() => runBattle(), 0);
        }, (frame) => {
            if (mountedRef.current) setLatestLiveFrame(frame);
        });
    }, []);

    const selectSpecies = useCallback(async (speciesId) => {
        setLoading(true);
        const response = await fetch(`/species/${speciesId}/latest`);
        const selectedSpecies = await response.json();
        setSpeciesData(selectedSpecies);
        setLoading(false);
        runBattle(selectedSpecies);
    }, [runBattle]);

    const loadLatestTrajectoryForReplay = useCallback(() => {
        if (!latestTrajectory) return;
        stopReplayPlayback();
        try {
            const trajectory = loadTrajectoryFromObject(latestTrajectory);
            setReplayTrajectory(trajectory);
            setReplayStepIndex(0);
            setReplayError(null);
        } catch (error) {
            setReplayTrajectory(null);
            setReplayError(error instanceof Error ? error.message : 'Failed to load trajectory');
        }
    }, [latestTrajectory, stopReplayPlayback]);

    const toggleReplayPlayback = useCallback(() => {
        if (!replayTrajectory?.steps.length) return;
        if (replayAutoPlay) {
            stopReplayPlayback();
            return;
        }

        if (replayStepIndex >= replayTrajectory.steps.length - 1) {
            setReplayStepIndex(0);
        }
        setReplayError(null);
        setReplayAutoPlay(true);
    }, [replayAutoPlay, replayStepIndex, replayTrajectory, stopReplayPlayback]);

    const selectReplayStep = useCallback((stepIndex) => {
        stopReplayPlayback();
        setReplayStepIndex(stepIndex);
    }, [stopReplayPlayback]);

    useEffect(() => {
        mountedRef.current = true;
        liveEnvironmentRef.current.start(getCanvasIds('live'));
        replayEnvironmentRef.current.start(getCanvasIds('replay'));
        refreshSpecies().finally(() => setLoading(false));
        const refreshTimer = window.setInterval(refreshSpecies, 5000);

        return () => {
            mountedRef.current = false;
            window.clearInterval(refreshTimer);
            stopReplayPlayback();
            liveEnvironmentRef.current.stop();
            replayEnvironmentRef.current.stop();
        };
    }, [refreshSpecies, stopReplayPlayback]);

    useEffect(() => {
        if (latestLiveFrame) {
            liveEnvironmentRef.current.setLiveFrame(latestLiveFrame.bots);
        }
    }, [latestLiveFrame]);

    const replayStepFrame = useMemo(
        () => replayTrajectory ? getStepFrame(replayTrajectory, replayStepIndex) : null,
        [replayStepIndex, replayTrajectory]
    );

    useEffect(() => {
        if (replayStepFrame) {
            replayEnvironmentRef.current.setTrajectoryFrame(replayStepFrame);
        }
    }, [replayStepFrame]);

    useEffect(() => {
        if (!replayAutoPlay || !replayTrajectory?.steps.length) return;

        const timer = window.setInterval(() => {
            setReplayStepIndex((stepIndex) => {
                if (stepIndex + 1 >= replayTrajectory.steps.length) {
                    window.clearInterval(timer);
                    replayTimerRef.current = null;
                    setReplayAutoPlay(false);
                    return stepIndex;
                }
                return stepIndex + 1;
            });
        }, 120);
        replayTimerRef.current = timer;

        return () => {
            window.clearInterval(timer);
            if (replayTimerRef.current === timer) {
                replayTimerRef.current = null;
            }
        };
    }, [replayAutoPlay, replayTrajectory]);

    const selectedStats = botStats[selectedBotId] || {};
    const selectedReplayPlayer = replayStepFrame
        ? getPlayerFrame(replayStepFrame, actorIdForBot(selectedBotId))
        : null;

    return {
        botIds: BOT_IDS,
        botStats,
        generation: speciesData?.totalGenerations,
        latestTrajectory,
        loading,
        loadLatestTrajectoryForReplay,
        maxFitness: speciesData?.maxFitness,
        replayAutoPlay,
        replayError,
        replayMaxStep: replayTrajectory ? Math.max(replayTrajectory.steps.length - 1, 0) : 0,
        replayStepIndex,
        replayTrajectory,
        selectSpecies,
        selectedBotId,
        selectedReplayPlayer,
        selectedStats,
        setReplayStepIndex: selectReplayStep,
        setSelectedBotId,
        species,
        speciesData,
        toggleReplayPlayback
    };
}
