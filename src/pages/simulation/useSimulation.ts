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
import { downloadTrajectoryJson, readTrajectoryFile } from '../../features/BattleEnvironment/trajectoryFileIO';

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
    const selectedSpeciesRef = useRef(null);
    const isBattleRunningRef = useRef(false);

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
    const [isBattleRunning, setIsBattleRunning] = useState(false);
    const [autoRunBattles, setAutoRunBattles] = useState(false);
    const [mode, setMode] = useState('live');

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

    const enterReplayMode = useCallback((trajectory) => {
        const loadedTrajectory = loadTrajectoryFromObject(trajectory);
        stopReplayPlayback();
        setReplayTrajectory(loadedTrajectory);
        setReplayStepIndex(0);
        setReplayError(null);
        setMode('replay');
        return loadedTrajectory;
    }, [stopReplayPlayback]);

    const runBattle = useCallback((existingSpecies = null) => {
        if (!mountedRef.current) return;
        if (isBattleRunningRef.current) return;

        const trainer = trainerRef.current;
        const speciesToUse = existingSpecies || selectedSpeciesRef.current;
        if (speciesToUse) {
            trainer.loadSpeciesFromJSON(speciesToUse);
        } else {
            trainer.createInitialSpecies();
        }

        isBattleRunningRef.current = true;
        setIsBattleRunning(true);
        setMode('live');

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
            isBattleRunningRef.current = false;
            setIsBattleRunning(false);
            const fitness = Trainer.calculateBotFitnessFromResults(results, trainer.totalGenerations);
            bot1.genome.addFitness(fitness);
            bot1.genome.totalRounds++;

            if (trainer.getTotalRoundsRemaining() <= 0) {
                trainer.newGeneration();
            }

            if (autoRunBattles) {
                window.setTimeout(() => runBattle(speciesToUse), 0);
            }
        }, (frame) => {
            if (mountedRef.current) setLatestLiveFrame(frame);
        });
    }, [autoRunBattles]);

    const selectSpecies = useCallback(async (speciesId) => {
        setLoading(true);
        const response = await fetch(`/species/${speciesId}/latest`);
        const selectedSpecies = await response.json();
        selectedSpeciesRef.current = selectedSpecies;
        setSpeciesData(selectedSpecies);
        setLoading(false);
        runBattle(selectedSpecies);
    }, [runBattle]);

    const loadLatestTrajectoryForReplay = useCallback(() => {
        if (!latestTrajectory) return;
        try {
            enterReplayMode(latestTrajectory);
        } catch (error) {
            stopReplayPlayback();
            setReplayTrajectory(null);
            setReplayStepIndex(0);
            setMode('live');
            setReplayError(error instanceof Error ? error.message : 'Failed to load trajectory');
        }
    }, [enterReplayMode, latestTrajectory]);

    const downloadLatestTrajectory = useCallback(() => {
        if (!latestTrajectory) return;
        downloadTrajectoryJson(latestTrajectory);
    }, [latestTrajectory]);

    const loadTrajectoryFile = useCallback(async (file) => {
        if (!file) return;

        try {
            const trajectory = await readTrajectoryFile(file);
            enterReplayMode(trajectory);
        } catch (error) {
            stopReplayPlayback();
            setReplayTrajectory(null);
            setReplayStepIndex(0);
            setMode('live');
            setReplayError(error instanceof Error ? error.message : 'Failed to load trajectory');
        }
    }, [enterReplayMode]);

    const clampReplayStepIndex = useCallback((stepIndex) => {
        if (!replayTrajectory?.steps.length) return 0;
        return Math.max(0, Math.min(stepIndex, replayTrajectory.steps.length - 1));
    }, [replayTrajectory]);

    const setReplayStep = useCallback((stepIndex) => {
        if (!replayTrajectory?.steps.length) return;
        stopReplayPlayback();
        setReplayStepIndex(clampReplayStepIndex(stepIndex));
    }, [clampReplayStepIndex, replayTrajectory, stopReplayPlayback]);

    const goToPreviousReplayStep = useCallback(() => {
        setReplayStep(replayStepIndex - 1);
    }, [replayStepIndex, setReplayStep]);

    const goToNextReplayStep = useCallback(() => {
        setReplayStep(replayStepIndex + 1);
    }, [replayStepIndex, setReplayStep]);

    const resetReplay = useCallback(() => {
        setReplayStep(0);
    }, [setReplayStep]);

    const toggleReplayPlayback = useCallback(() => {
        if (!replayTrajectory?.steps.length) return;
        if (replayAutoPlay) {
            stopReplayPlayback();
            return;
        }

        if (replayStepIndex >= replayTrajectory.steps.length - 1) {
            setReplayStepIndex(0);
        }
        setMode('replay');
        setReplayError(null);
        setReplayAutoPlay(true);
    }, [replayAutoPlay, replayStepIndex, replayTrajectory, stopReplayPlayback]);

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
        autoRunBattles,
        downloadLatestTrajectory,
        isBattleRunning,
        generation: speciesData?.totalGenerations,
        latestTrajectory,
        loading,
        loadTrajectoryFile,
        loadLatestTrajectoryForReplay,
        maxFitness: speciesData?.maxFitness,
        mode,
        replayAutoPlay,
        replayError,
        replayMaxStep: replayTrajectory ? Math.max(replayTrajectory.steps.length - 1, 0) : 0,
        replayStepIndex,
        replayTrajectory,
        resetReplay,
        goToNextReplayStep,
        goToPreviousReplayStep,
        selectSpecies,
        selectedBotId,
        selectedReplayPlayer,
        selectedStats,
        setReplayStepIndex: setReplayStep,
        setSelectedBotId,
        setAutoRunBattles,
        species,
        speciesData,
        runBattleOnce: runBattle,
        toggleReplayPlayback
    };
}
