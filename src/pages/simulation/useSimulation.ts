// @ts-nocheck
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Bot from '../../engine/agents/bot';
import Genome from '../../engine/evolution/genome';
import Battleground from '../../engine/simulation/battleground';
import { DEFAULT_BATTLE_CONFIG, createBattleTeams, resolveBattleConfig } from '../../engine/simulation/battleSetup';
import Trainer from '../../engine/training/trainer';
import { evaluateTrajectory } from '../../evaluation/evaluateTrajectory';
import { createBaselineRunResult, groupBaselineRowsBySeed, summarizeBaselineRows } from '../../evaluation/baselineComparison';
import {
    createReplayStateFromStep,
    getPlayerFrame,
    getStepFrame,
    loadTrajectoryFromObject
} from '../../engine/traces/trajectoryReplay';
import BattleEnvironmentController from '../../features/BattleEnvironment/controller';
import { downloadTrajectoryJson, readTrajectoryFile } from '../../features/BattleEnvironment/trajectoryFileIO';
import { loadLinearIntentModelFromUrl } from '../../engine/policies/linearIntent/linearIntentModel';
import {
    BOT_POLICY_TYPES,
    createDefaultBotPolicyConfig,
    formatBotPolicy,
    requiresLinearIntentModel,
    setBotPolicy
} from './botPolicyConfig';

const DEFAULT_SETUP = createBattleTeams(DEFAULT_BATTLE_CONFIG);
function getCanvasIds(prefix, battleConfig = DEFAULT_BATTLE_CONFIG) {
    const setup = createBattleTeams(battleConfig);
    const canvasByActorId = setup.players.reduce((mapping, player) => ({
        ...mapping,
        [player.id]: `${prefix}-bot${player.numericId}brain`
    }), {
        'team-a-0': `${prefix}-bot1brain`,
        'team-a-1': `${prefix}-bot2brain`,
        'team-b-0': `${prefix}-bot3brain`,
        'team-b-1': `${prefix}-bot4brain`
    });

    return {
        battle: `${prefix}-battleground`,
        bot: (botId) => canvasByActorId[String(botId)] || `${prefix}-bot${botId}brain`
    };
}

export function actorIdForBot(botId) {
    const defaultPlayer = DEFAULT_SETUP.players.find((player) => player.numericId === botId);
    return defaultPlayer ? defaultPlayer.id : `bot-${botId}`;
}

function createBotFromDescriptor(descriptor, genome, totalGenerations, policyConfig, linearIntentModel, useOpponentAi = false) {
    const bot = new Bot(descriptor.numericId, descriptor.teamId);
    bot.actorId = descriptor.id;
    bot.loadGenome(genome);
    if (useOpponentAi) {
        bot.selectAIMethod(totalGenerations);
    }

    const policy = policyConfig[bot.id] || BOT_POLICY_TYPES.genome;
    bot.setPolicyMode(policy);
    if (policy === BOT_POLICY_TYPES.linearIntent) {
        bot.setLinearIntentModel(linearIntentModel);
    }

    return bot;
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

function createEmptySessionTotals(playerId, teamId) {
    return {
        playerId,
        teamId,
        battlesPlayed: 0,
        totalDamageDealt: 0,
        totalDamageTaken: 0,
        totalSurvivalSteps: 0,
        cpcApplicable: false,
        cpcBattleCount: 0,
        totalTeammateUnderPressureEvents: 0,
        totalTeammateUnderPressureResponses: 0,
        totalResponseRateSum: 0,
        totalIsolationRateSum: 0,
        totalEvaluationScoreSum: 0
    };
}

function mergeSessionTotals(previousTotals, evaluation) {
    const nextTotals = { ...previousTotals };
    const playerIds = Object.keys(evaluation?.players || {});

    playerIds.forEach((playerId) => {
        const summary = evaluation?.players?.[playerId];
        if (!summary) {
            return;
        }

        const teamId = summary.teamId || 'unknown';
        const current = nextTotals[playerId] || createEmptySessionTotals(playerId, teamId);

        current.teamId = teamId;
        current.battlesPlayed += 1;
        current.totalDamageDealt += summary.player.damageDealt || 0;
        current.totalDamageTaken += summary.player.damageTaken || 0;
        current.totalSurvivalSteps += summary.player.survivalSteps || 0;
        if (summary.cpc?.applicable) {
            current.cpcApplicable = true;
            current.cpcBattleCount += 1;
            current.totalTeammateUnderPressureEvents += summary.cpc.teammateUnderPressureEvents || 0;
            current.totalTeammateUnderPressureResponses += summary.cpc.teammateUnderPressureResponses || 0;
            current.totalResponseRateSum += summary.cpc.teammateResponseRate || 0;
            current.totalIsolationRateSum += summary.cpc.isolationRate || 0;
        }
        current.totalEvaluationScoreSum += summary.evaluationScore || 0;

        nextTotals[playerId] = current;
    });

    return nextTotals;
}

function summarizeSessionTotals(sessionTotals) {
    const players = {};
    const teams = {};

    Object.values(sessionTotals || {}).forEach((totals) => {
        const battleCount = totals.battlesPlayed || 1;
        const player = {
            damageDealt: totals.totalDamageDealt,
            damageTaken: totals.totalDamageTaken,
            survivalSteps: totals.totalSurvivalSteps,
        };

        const cpcBattleCount = totals.cpcBattleCount || 0;
        const cpc = {
            applicable: Boolean(totals.cpcApplicable),
            teammateUnderPressureEvents: totals.totalTeammateUnderPressureEvents,
            teammateUnderPressureResponses: totals.totalTeammateUnderPressureResponses,
            teammateResponseRate: cpcBattleCount > 0 ? Number((totals.totalResponseRateSum / cpcBattleCount).toFixed(2)) : 0,
            isolatedSteps: 0,
            isolationRate: cpcBattleCount > 0 ? Number((totals.totalIsolationRateSum / cpcBattleCount).toFixed(2)) : 0,
            avgAllyDistance: null
        };

        players[totals.playerId] = {
            playerId: totals.playerId,
            teamId: totals.teamId,
            player,
            cpc,
            evaluationScore: battleCount > 0 ? Number((totals.totalEvaluationScoreSum / battleCount).toFixed(2)) : 0
        };

        if (!teams[totals.teamId]) {
            teams[totals.teamId] = {
                teamId: totals.teamId,
                playerIds: [],
                damageDealt: 0,
                damageTaken: 0,
                survivalSteps: 0,
                avgTeammateResponseRate: 0,
                avgIsolationRate: 0,
                avgEvaluationScore: 0,
                _playerCount: 0,
                _cpcPlayerCount: 0,
                _responseRateSum: 0,
                _isolationRateSum: 0,
                _evaluationScoreSum: 0
            };
        }

        const team = teams[totals.teamId];
        team.playerIds.push(totals.playerId);
        team.damageDealt += totals.totalDamageDealt;
        team.damageTaken += totals.totalDamageTaken;
        team.survivalSteps += totals.totalSurvivalSteps;
        if (cpc.applicable) {
            team._responseRateSum += cpc.teammateResponseRate;
            team._isolationRateSum += cpc.isolationRate;
            team._cpcPlayerCount += 1;
        }
        team._evaluationScoreSum += players[totals.playerId].evaluationScore;
        team._playerCount += 1;
    });

    Object.values(teams).forEach((team) => {
        team.avgTeammateResponseRate = team._cpcPlayerCount > 0 ? Number((team._responseRateSum / team._cpcPlayerCount).toFixed(2)) : 0;
        team.avgIsolationRate = team._cpcPlayerCount > 0 ? Number((team._isolationRateSum / team._cpcPlayerCount).toFixed(2)) : 0;
        team.avgEvaluationScore = team._playerCount > 0 ? Number((team._evaluationScoreSum / team._playerCount).toFixed(2)) : 0;
        delete team._playerCount;
        delete team._cpcPlayerCount;
        delete team._responseRateSum;
        delete team._isolationRateSum;
        delete team._evaluationScoreSum;
    });

    return {
        trajectoryId: 'session',
        schemaVersion: 'session',
        players,
        teams
    };
}

export function useSimulation() {
    const trainerRef = useRef(new Trainer());
    const liveEnvironmentRef = useRef(new BattleEnvironmentController());
    const replayEnvironmentRef = useRef(new BattleEnvironmentController());
    const mountedRef = useRef(true);
    const replayTimerRef = useRef(null);
    const selectedSpeciesRef = useRef(null);
    const isBattleRunningRef = useRef(false);
    const baselineControlledBotRef = useRef(null);
    const pressedKeysRef = useRef(new Set());

    const [loading, setLoading] = useState(true);
    const [species, setSpecies] = useState([]);
    const [speciesData, setSpeciesData] = useState(null);
    const [latestTrajectory, setLatestTrajectory] = useState(null);
    const [latestBattleResult, setLatestBattleResult] = useState(null);
    const [currentEvaluation, setCurrentEvaluation] = useState(null);
    const [replayTrajectory, setReplayTrajectory] = useState(null);
    const [replayStepIndex, setReplayStepIndex] = useState(0);
    const [replayAutoPlay, setReplayAutoPlay] = useState(false);
    const [replayError, setReplayError] = useState(null);
    const [latestLiveFrame, setLatestLiveFrame] = useState(null);
    const [selectedBotId, setSelectedBotId] = useState(1);
    const [botStats, setBotStats] = useState({});
    const [sessionTotals, setSessionTotals] = useState({});
    const [isBattleRunning, setIsBattleRunning] = useState(false);
    const [autoRunBattles, setAutoRunBattles] = useState(false);
    const [mode, setMode] = useState('live');
    const [botPolicyConfig, setBotPolicyConfig] = useState(createDefaultBotPolicyConfig());
    const [battleConfig, setBattleConfig] = useState(DEFAULT_BATTLE_CONFIG);
    const [linearIntentModel, setLinearIntentModel] = useState(null);
    const [linearModelLoadStatus, setLinearModelLoadStatus] = useState('Linear model not loaded');
    const [linearModelError, setLinearModelError] = useState('');
    const [linearModelLoading, setLinearModelLoading] = useState(false);
    const [baselineSeed, setBaselineSeed] = useState(1);
    const [baselineResults, setBaselineResults] = useState([]);

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

    const loadLinearIntentModel = useCallback(async () => {
        setLinearModelLoading(true);
        setLinearModelError('');
        setLinearModelLoadStatus('Loading linear intent model...');
        try {
            const model = await loadLinearIntentModelFromUrl();
            setLinearIntentModel(model);
            setLinearModelLoadStatus(`Loaded ${model.schemaVersion}`);
            return model;
        } catch (error) {
            const message = error instanceof Error ? error.message : 'Failed to load linear intent model';
            setLinearIntentModel(null);
            setLinearModelLoadStatus('Linear model load failed');
            setLinearModelError(message);
            throw error;
        } finally {
            setLinearModelLoading(false);
        }
    }, []);

    const setBotPolicyForBot = useCallback((botId, policy) => {
        setBotPolicyConfig((current) => setBotPolicy(current, botId, policy));
    }, []);

    const setAllGenomePolicies = useCallback(() => {
        const botIds = createBattleTeams(battleConfig).players.map((player) => player.numericId);
        setBotPolicyConfig(botIds.reduce((config, botId) => ({ ...config, [botId]: BOT_POLICY_TYPES.genome }), {}));
    }, [battleConfig]);

    const setAllLinearPolicies = useCallback(() => {
        const botIds = createBattleTeams(battleConfig).players.map((player) => player.numericId);
        setBotPolicyConfig(botIds.reduce((config, botId) => ({ ...config, [botId]: BOT_POLICY_TYPES.linearIntent }), {}));
    }, [battleConfig]);

    const setAllNonePolicies = useCallback(() => {
        const botIds = createBattleTeams(battleConfig).players.map((player) => player.numericId);
        setBotPolicyConfig(botIds.reduce((config, botId) => ({ ...config, [botId]: BOT_POLICY_TYPES.none }), {}));
    }, [battleConfig]);

    useEffect(() => {
        if (requiresLinearIntentModel(botPolicyConfig) && !linearIntentModel && !linearModelLoading) {
            loadLinearIntentModel().catch(() => {
                // Error state already set by the loader.
            });
        }
    }, [botPolicyConfig, linearIntentModel, linearModelLoading, loadLinearIntentModel]);

    useEffect(() => {
        if (!requiresLinearIntentModel(botPolicyConfig)) {
            setLinearModelError('');
        }
    }, [botPolicyConfig]);

    const enterReplayMode = useCallback((trajectory) => {
        const loadedTrajectory = loadTrajectoryFromObject(trajectory);
        stopReplayPlayback();
        setReplayTrajectory(loadedTrajectory);
        setReplayStepIndex(0);
        setReplayError(null);
        setMode('replay');
        return loadedTrajectory;
    }, [stopReplayPlayback]);

    const runBattle = useCallback(function runBattleInternal(existingSpecies = null) {
        if (!mountedRef.current) return;
        if (isBattleRunningRef.current) return;

        const trainer = trainerRef.current;
        const speciesToUse = existingSpecies || selectedSpeciesRef.current;

        const usesLinearIntent = requiresLinearIntentModel(botPolicyConfig);
        if (usesLinearIntent && !linearIntentModel) {
            setLinearModelError('Load the linear intent model before running a linear intent battle.');
            return;
        }

        if (speciesToUse) {
            trainer.loadSpeciesFromJSON(speciesToUse);
        } else {
            trainer.createInitialSpecies();
        }

        isBattleRunningRef.current = true;
        setIsBattleRunning(true);
        setMode('live');
        setLatestBattleResult(null);
        setCurrentEvaluation(null);

        const setup = createBattleTeams(battleConfig);
        const bot1Genome = trainer.getTopGenome();
        if (!bot1Genome) {
            isBattleRunningRef.current = false;
            setIsBattleRunning(false);
            setLinearModelError('No genome was available for the selected species. Try another species or retrain.');
            return;
        }
        const bot3Genome = trainer.getTopGenome();
        if (!bot3Genome) {
            isBattleRunningRef.current = false;
            setIsBattleRunning(false);
            setLinearModelError('No opponent genome was available for the selected species. Try another species or retrain.');
            return;
        }
        const bots = setup.players.map((descriptor, index) => {
            const teamIndex = Math.floor(index / setup.config.playersPerTeam);
            const sourceGenome = teamIndex === 0 ? bot1Genome : bot3Genome;
            const genome = descriptor.slotIndex === 0 ? sourceGenome : Genome.loadFromJSON(sourceGenome.serialize());
            return createBotFromDescriptor(
                descriptor,
                genome,
                trainer.totalGenerations,
                botPolicyConfig,
                linearIntentModel,
                teamIndex > 0
            );
        });

        setBotStats(bots.reduce((stats, bot, index) => ({
            ...stats,
            [bot.id]: Math.floor(index / setup.config.playersPerTeam) === 0 ? bot1Genome.getStats() : bot3Genome.getStats()
        }), {}));

        const battleground = new Battleground(setup.config);
        battleground.addBots(...bots);
        battleground.start((results) => {
            if (!mountedRef.current) return;

            const trajectory = results.trajectory;
            const evaluation = evaluateTrajectory(trajectory);

            setLatestTrajectory(trajectory);
            setLatestBattleResult(results);
            setCurrentEvaluation(evaluation);
            setSessionTotals((previousTotals) => mergeSessionTotals(previousTotals, evaluation));
            isBattleRunningRef.current = false;
            setIsBattleRunning(false);
            const fitness = Trainer.calculateBotFitnessFromResults(results, trainer.totalGenerations);
            bots[0].genome.addFitness(fitness);
            bots[0].genome.totalRounds++;

            if (trainer.getTotalRoundsRemaining() <= 0) {
                trainer.newGeneration();
            }

            if (autoRunBattles) {
                window.setTimeout(() => runBattleInternal(speciesToUse), 0);
            }
        }, (frame) => {
            if (mountedRef.current) setLatestLiveFrame(frame);
        });
    }, [autoRunBattles, battleConfig, botPolicyConfig, linearIntentModel]);

    const selectSpecies = useCallback(async (speciesId) => {
        setLoading(true);
        const response = await fetch(`/species/${speciesId}/latest`);
        const selectedSpecies = await response.json();
        selectedSpeciesRef.current = selectedSpecies;
        setSpeciesData(selectedSpecies);
        setLoading(false);
        runBattle(selectedSpecies);
    }, [runBattle]);

    const runBattleOnce = useCallback(() => {
        runBattle();
    }, [runBattle]);

    const loadLatestTrajectoryForReplay = useCallback(() => {
        if (!latestTrajectory) return;
        try {
            enterReplayMode(latestTrajectory);
            setCurrentEvaluation(evaluateTrajectory(latestTrajectory));
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
            setCurrentEvaluation(evaluateTrajectory(trajectory));
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

    const resetScores = useCallback(() => {
        setSessionTotals({});
    }, []);

    const updateUserControlledAction = useCallback(() => {
        const bot = baselineControlledBotRef.current;
        if (!bot) return;

        const keys = pressedKeysRef.current;
        bot.setUserAction({
            dx: (keys.has('d') || keys.has('arrowright') ? 15 : 0) - (keys.has('a') || keys.has('arrowleft') ? 15 : 0),
            dy: (keys.has('s') || keys.has('arrowdown') ? 15 : 0) - (keys.has('w') || keys.has('arrowup') ? 15 : 0),
            dh: (keys.has('e') ? 15 : 0) - (keys.has('q') ? 15 : 0),
            ds: keys.has(' ')
        });
    }, []);

    const runSoloBaseline = useCallback((policyType) => {
        if (!mountedRef.current) return;
        if (isBattleRunningRef.current) return;

        const playerCount = 4;
        const maxSteps = 600;
        const seed = Number(baselineSeed) || 1;
        const baselineConfig = {
            mode: 'solo',
            seed,
            policyType,
            playerCount,
            maxSteps,
            runLabel: policyType === 'user-controlled' ? 'Stage A Human Solo Baseline' : 'Stage A Random Solo Baseline'
        };
        const setup = createBattleTeams({ mode: 'solo', teamCount: playerCount, maxSteps });
        const bots = setup.players.map((descriptor, index) => {
            const bot = new Bot(descriptor.numericId, descriptor.teamId);
            bot.actorId = descriptor.id;
            const isHuman = policyType === 'user-controlled' && index === 0;
            bot.setPolicyMode(isHuman ? BOT_POLICY_TYPES.userControlled : BOT_POLICY_TYPES.random);
            if (isHuman) {
                baselineControlledBotRef.current = bot;
                updateUserControlledAction();
            }
            return bot;
        });

        isBattleRunningRef.current = true;
        setIsBattleRunning(true);
        setMode('live');
        setLatestBattleResult(null);
        setCurrentEvaluation(null);

        const battleground = new Battleground({ mode: 'solo', teamCount: playerCount, maxSteps });
        battleground.addBots(...bots);
        battleground.start((results) => {
            if (!mountedRef.current) return;

            baselineControlledBotRef.current = null;
            const trajectory = results.trajectory;
            trajectory.baselineRun = baselineConfig;
            trajectory.seed = seed;
            const evaluation = evaluateTrajectory(trajectory);
            const baselineResult = createBaselineRunResult({
                config: baselineConfig,
                trajectory,
                evaluation
            });

            setLatestTrajectory(trajectory);
            setLatestBattleResult(results);
            setCurrentEvaluation(evaluation);
            setBaselineResults((currentResults) => [baselineResult, ...currentResults]);
            isBattleRunningRef.current = false;
            setIsBattleRunning(false);
        }, (frame) => {
            if (mountedRef.current) setLatestLiveFrame(frame);
        });
    }, [baselineSeed, updateUserControlledAction]);

    const downloadBaselineResults = useCallback(() => {
        const rows = baselineResults.flatMap((result) => result.rows);
        const payload = {
            createdAt: new Date().toISOString(),
            results: baselineResults,
            summaries: summarizeBaselineRows(rows),
            sameSeedGroups: groupBaselineRowsBySeed(rows)
        };
        const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json;charset=utf-8' });
        const objectUrl = window.URL.createObjectURL(blob);
        const anchor = document.createElement('a');
        anchor.href = objectUrl;
        anchor.download = `stage-a-solo-baselines-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
        anchor.rel = 'noopener';
        anchor.style.display = 'none';
        document.body.appendChild(anchor);
        anchor.click();
        document.body.removeChild(anchor);
        window.URL.revokeObjectURL(objectUrl);
    }, [baselineResults]);

    useEffect(() => {
        mountedRef.current = true;
        liveEnvironmentRef.current.start(getCanvasIds('live', battleConfig));
        replayEnvironmentRef.current.start(getCanvasIds('replay', battleConfig));
        refreshSpecies().finally(() => setLoading(false));
        const refreshTimer = window.setInterval(refreshSpecies, 5000);

        return () => {
            mountedRef.current = false;
            window.clearInterval(refreshTimer);
            stopReplayPlayback();
            liveEnvironmentRef.current.stop();
            replayEnvironmentRef.current.stop();
        };
    }, [battleConfig, refreshSpecies, stopReplayPlayback]);

    useEffect(() => {
        const onKeyDown = (event) => {
            const key = event.key.toLowerCase();
            if (['w', 'a', 's', 'd', 'q', 'e', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright', ' '].includes(key)) {
                pressedKeysRef.current.add(key);
                updateUserControlledAction();
                if (baselineControlledBotRef.current) {
                    event.preventDefault();
                }
            }
        };
        const onKeyUp = (event) => {
            const key = event.key.toLowerCase();
            pressedKeysRef.current.delete(key);
            updateUserControlledAction();
        };

        window.addEventListener('keydown', onKeyDown);
        window.addEventListener('keyup', onKeyUp);

        return () => {
            window.removeEventListener('keydown', onKeyDown);
            window.removeEventListener('keyup', onKeyUp);
        };
    }, [updateUserControlledAction]);

    useEffect(() => {
        const activeBotIds = createBattleTeams(battleConfig).players.map((player) => player.numericId);
        if (!activeBotIds.includes(selectedBotId)) {
            setSelectedBotId(activeBotIds[0] || 1);
        }
    }, [battleConfig, selectedBotId]);

    useEffect(() => {
        if (latestLiveFrame) {
            liveEnvironmentRef.current.setLiveFrame(latestLiveFrame.bots);
        }
    }, [latestLiveFrame]);

    const replayStepFrame = useMemo(
        () => replayTrajectory ? getStepFrame(replayTrajectory, replayStepIndex) : null,
        [replayStepIndex, replayTrajectory]
    );

    const battleSetup = useMemo(
        () => createBattleTeams(battleConfig),
        [battleConfig]
    );
    const botIds = useMemo(
        () => battleSetup.players.map((player) => player.numericId),
        [battleSetup]
    );
    const actorIdByBotId = useMemo(
        () => battleSetup.players.reduce((mapping, player) => ({
            ...mapping,
            [player.numericId]: player.id
        }), {}),
        [battleSetup]
    );

    const accumulatedEvaluation = useMemo(
        () => summarizeSessionTotals(sessionTotals),
        [sessionTotals]
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
        ? getPlayerFrame(replayStepFrame, actorIdByBotId[selectedBotId] || actorIdForBot(selectedBotId))
        : null;
    const runBattleRequiresLinearModel = requiresLinearIntentModel(botPolicyConfig);
    const baselineRows = baselineResults.flatMap((result) => result.rows);

    return {
        botIds,
        actorIdByBotId,
        botStats,
        autoRunBattles,
        accumulatedEvaluation,
        downloadLatestTrajectory,
        isBattleRunning,
        generation: speciesData?.totalGenerations,
        latestTrajectory,
        latestBattleResult,
        latestLiveFrame,
        currentEvaluation,
        loading,
        loadTrajectoryFile,
        loadLatestTrajectoryForReplay,
        loadLinearIntentModel,
        linearIntentModel,
        linearModelError,
        linearModelLoadStatus,
        linearModelLoading,
        maxFitness: speciesData?.maxFitness,
        mode,
        botPolicyConfig,
        battleConfig,
        baselineResults,
        baselineRows,
        baselineSeed,
        baselineSummaries: summarizeBaselineRows(baselineRows),
        baselineSameSeedGroups: groupBaselineRowsBySeed(baselineRows),
        runBattleRequiresLinearModel,
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
        setBotPolicyForBot,
        setBattleConfig: (nextConfig) => setBattleConfig(resolveBattleConfig(nextConfig)),
        setBaselineSeed,
        runRandomSoloBaseline: () => runSoloBaseline('random'),
        runUserControlledSoloBaseline: () => runSoloBaseline('user-controlled'),
        downloadBaselineResults,
        setAllGenomePolicies,
        setAllLinearPolicies,
        setAllNonePolicies,
        setAutoRunBattles,
        resetScores,
        species,
        speciesData,
        runBattleOnce,
        toggleReplayPlayback
    };
}
