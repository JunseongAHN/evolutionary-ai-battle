// @ts-nocheck
/**
 * The battleground class controls the updating and drawing of an invidual battle between bots.
 *
 * It runs an update loop that ticks every config.tickTime milliseconds, and runs a draw loop that runs
 * as fast as your computer can handle.
 */
import { distanceBetweenPoints } from '../../shared/math';
import config from '../../../config/default.json';
import log from '../../shared/logger';
import { getAliveBots, getAliveTeamIds, isEnemy } from './teams';
import TraceRecorder from '../traces/traceRecorder';
import { createDefaultDecisionReason } from '../traces/trace';

const TICK_TIME = config.tickTime;
const BOT_RADIUS = config.botSize / 2;
const BULLET_RADIUS = config.bulletSize / 2;

const MIN_X_POS = 0 + BOT_RADIUS;
const MIN_Y_POS = 0 + BOT_RADIUS;
const MAX_X_POS = config.mapWidth - BOT_RADIUS;
const MAX_Y_POS = config.mapHeight - BOT_RADIUS;
const MAX_SPEED = config.maxSpeed;
const BULLET_SPEED = config.bulletSpeed;

const NO_ACTION_TIMEOUT = config.noActionTimeout;
const NO_MOVE_TIMEOUT = config.noMoveTimeout;
const BATTLE_TIMEOUT = config.maxRoundTime;

class Battleground {
    constructor() {
        this.bots = [];
        this.botActions = [];
        this.bullets = [];
        this.traceRecorder = new TraceRecorder();
        this.trajectoryStep = 0;
        this.projectileSequence = 0;
        this.onEnd = null;
        this.onFrame = null;
        this.winner = null;
        this.endReason = 'battle_timeout';
        this.lastActionTime = null;
        this.lastBotMoveTime = null;
        this.lastShootTime = [];
    }

    /**
     * Add both bots to the battleground
     * @param {Bot} bot1
     * @param {Bot} bot2
     */
    addBots(...bots) {
        this.bots.push(...bots);
        this.lastShootTime = this.bots.map(() => Date.now());
    }

    /**
     * Initializes all the variables for the battle and starts the battleground.
     * Sets updateBots function to run every TICK_TIME, while the update and draw
     * functions run at 10ms to make the game look smooth.
     * @param {Function} onEnd - callback to call after the battle has ended
     */
    start(onEnd, onFrame = null) {
        this.onEnd = onEnd;
        this.onFrame = onFrame;
        this.startTime = Date.now();
        this.lastUpdate = Date.now();
        this.lastActionTime = Date.now();
        this.lastBotMoveTime = Date.now();
        this.trajectoryStep = 0;
        this.projectileSequence = 0;
        this.endReason = 'battle_timeout';
        this.traceRecorder.reset();
        this.traceRecorder.startTrajectory(this.createTrajectoryMetadata());
        this.traceRecorder.recordInitialState(this.createInitialState());
        this.emitFrame();
        this.updateBots();
        this.updateBotsInterval = setInterval(this.updateBots.bind(this), TICK_TIME);
        this.updateInterval = setInterval(this.update.bind(this), 10);
    }

    /**
     * End the battle, clearing all the update timers, calculating results and reporting those
     * results to the onEnd callback function.
     */
    end() {
        if (!this.onEnd) return;

        clearInterval(this.updateBotsInterval);
        clearInterval(this.updateInterval);
        this.endTime = Date.now();
        this.traceRecorder.finishTrajectory({
            winnerTeamId: this.winner || null,
            endStep: this.trajectoryStep,
            endReason: this.endReason
        });
        const totalTime = (this.endTime - this.startTime) / 1000;
        const results = {
            startTime: this.startTime,
            endTime: this.endTime,
            totalTime,
            winner: this.winner,
            winnerTeamId: this.winner,
            endReason: this.endReason,
            trajectory: this.traceRecorder.getTrajectory(),
            bots: this.bots.map((bot) => ({
                id: bot.id,
                teamId: bot.teamId,
                lives: bot.lives
            })),
            bot1: {
                lives: this.bots[0].lives,
                teamId: this.bots[0].teamId
            },
            bot2: {
                lives: this.bots[2].lives,
                teamId: this.bots[2].teamId
            }
        };
        this.onEnd(results);
        this.onEnd = null;
    }

    /**
     * Calls the bot update function with the current game state, then retrieves the actions the bot
     * wants to take and returns them to the calling function.
     * @param {Bot} bot
     * @param {Bot} otherBot
     */
    updateBot(bot, otherBot) {
        const gameState = {
            xPos: bot.xPos,
            yPos: bot.yPos,
            rotation: bot.rotation,
            bullets: bot.bullets,
            otherPlayer: {
                xPos: otherBot.xPos,
                yPos: otherBot.yPos,
                rotation: otherBot.rotation,
                bullets: otherBot.bullets
            }
        };
        const botActions = bot.update(gameState);
        return botActions;
    }

    /**
     * Main update loop for the two bots in the world. Gathers their actions which are then used
     * in the update loop. Also keeps track of the last time bot1 (the bot we are training)
     * performed an action so that if it stops doing anything for a while the battlefield ends.
     */
    updateBots() {
        this.bots.forEach((bot, index) => {
            const nearestEnemy = this.getNearestEnemy(bot);
            this.botActions[index] = bot.lives > 0 && nearestEnemy
                ? this.updateBot(bot, nearestEnemy)
                : { dx: 0, dy: 0, dh: 0, ds: false };
        });
        if (this.botDidActions(this.botActions[0])) {
            this.lastActionTime = Date.now();
        }
        this.checkForWinner();
        if ((Date.now() - this.lastActionTime) / 1000 > NO_ACTION_TIMEOUT) {
            this.endReason = 'no_action_timeout';
            this.end();
        }
        if ((Date.now() - this.lastBotMoveTime) / 1000 > NO_MOVE_TIMEOUT) {
            this.endReason = 'no_move_timeout';
            this.end();
        }
        if ((Date.now() - this.startTime) / 1000 > BATTLE_TIMEOUT) {
            this.endReason = 'battle_timeout';
            this.end();
        }
    }

    /**
     * Takes a set of actions returned from the bot.update function and determines if the bot
     * is actually taking any action. Returns true if the bot is doing anything, false if not.
     * @param {Object} botActions
     */
    botDidActions(botActions) {
        return botActions.dx != 0 || botActions.dy != 0 || botActions.dh != 0 || botActions.ds != 0;
    }

    /**
     * Compares the bots old position to it's new posittion. Returns true if the bot moved, false
     * if the bot did not move.
     * @param {Bot} bot
     * @param {int} newXPos
     * @param {int} newYPos
     */
    botMoved(bot, newXPos, newYPos) {
        return bot.xPos != newXPos || bot.yPos != newYPos;
    }

    /**
     * The main update loop of the battlefield. Takes the actions for each bot and makes the bots
     * move around and shoot based on them. Then calculates if any bullets collided, checks lives
     * lost, and ends the game if there is a final winner.
     */
    update() {
        const delta = (Date.now() - this.lastUpdate) / 1000;
        const moveSpeedMultiplier = 1000 / TICK_TIME; // Bots actually move at maxSpeed every 75ms not every 1000ms.

        this.lastUpdate = Date.now();
        this.stepDamage = {};
        this.stepActions = this.botActions.map((action) => action ? { ...action } : null);

        for (var i = 0; i < this.bots.length; i++) {
            const bot = this.bots[i];
            const botActions = this.botActions[i];
            if (bot.lives <= 0 || !botActions) {
                continue;
            }

            const xMovement = Math.max(Math.min(botActions.dx, MAX_SPEED), -MAX_SPEED) * delta * moveSpeedMultiplier;
            const yMovement = Math.max(Math.min(botActions.dy, MAX_SPEED), -MAX_SPEED) * delta * moveSpeedMultiplier;
            const rotation = Math.max(Math.min(botActions.dh, MAX_SPEED), -MAX_SPEED) * delta * moveSpeedMultiplier;

            const newXPos = Math.min(Math.max(bot.xPos + xMovement, MIN_X_POS), MAX_X_POS);
            const newYPos = Math.min(Math.max(bot.yPos + yMovement, MIN_Y_POS), MAX_Y_POS);

            if (bot.id === 1 && this.botMoved(bot, newXPos, newYPos)) {
                this.lastBotMoveTime = Date.now();
            }

            bot.xPos = newXPos;
            bot.yPos = newYPos;
            bot.rotation += rotation;
            if (bot.rotation > 360) {
                bot.rotation -= 360;
            }
            if (bot.rotation < 0) {
                bot.rotation += 360;
            }

            bot.bullets.forEach((bullet) => {
                const xDistance = BULLET_SPEED * Math.cos(bullet.rotation * Math.PI / 180) * delta * moveSpeedMultiplier;
                const yDistance = BULLET_SPEED * Math.sin(bullet.rotation * Math.PI / 180) * delta * moveSpeedMultiplier;
                bullet.xPos += xDistance;
                bullet.yPos += yDistance;
                if (bullet.xPos > MAX_X_POS || bullet.xPos < 0) {
                    bullet.dead = true;
                }
                if (bullet.yPos > MAX_Y_POS || bullet.yPos < 0) {
                    bullet.dead = true;
                }

                const hitEnemy = getAliveBots(this.bots).find((otherBot) => {
                    return isEnemy(bot, otherBot)
                        && distanceBetweenPoints(bullet.xPos, bullet.yPos, otherBot.xPos, otherBot.yPos) < (BULLET_RADIUS + BOT_RADIUS);
                });
                if (!hitEnemy) return;

                hitEnemy.lives -= 1;
                const actorId = this.getActorId(bot);
                const targetId = this.getActorId(hitEnemy);
                this.stepDamage[actorId] = {
                    ...(this.stepDamage[actorId] || {}),
                    dealt: (this.stepDamage[actorId]?.dealt || 0) + 1
                };
                this.stepDamage[targetId] = {
                    ...(this.stepDamage[targetId] || {}),
                    taken: (this.stepDamage[targetId]?.taken || 0) + 1
                };
                this.traceRecorder.recordEvent(this.trajectoryStep, {
                    type: hitEnemy.lives <= 0 ? 'elimination' : 'damage',
                    actorId,
                    actorTeamId: bot.teamId,
                    targetId,
                    targetTeamId: hitEnemy.teamId,
                    damage: 1
                });
                log.debug('Bot ' + hitEnemy.id + ' hit! Now has ' + hitEnemy.lives + ' lives left.');
                bullet.dead = true;
            });

            bot.bullets = bot.bullets.filter(function (bullet) { return !bullet.dead; });
            log.debug('Bot bullets: ', bot.bullets);

            if (botActions.ds && bot.bullets.length < 5 && (Date.now() - this.lastShootTime[i]) >= TICK_TIME) {
                this.lastShootTime[i] = Date.now();
                let bullet = this.spawnBullet(bot);
                log.debug('Spawning bullet: ', bullet);
                botActions.ds = false;
                bot.bullets.push(bullet);
            }
        }

        this.recordTrajectoryStep();
        this.checkForWinner();
        this.emitFrame();
    }

    getNearestEnemy(bot) {
        return getAliveBots(this.bots)
            .filter((otherBot) => isEnemy(bot, otherBot))
            .reduce((nearestEnemy, enemy) => {
                if (!nearestEnemy) return enemy;
                const nearestDistance = distanceBetweenPoints(bot.xPos, bot.yPos, nearestEnemy.xPos, nearestEnemy.yPos);
                const enemyDistance = distanceBetweenPoints(bot.xPos, bot.yPos, enemy.xPos, enemy.yPos);
                return enemyDistance < nearestDistance ? enemy : nearestEnemy;
            }, null);
    }

    checkForWinner() {
        const aliveTeamIds = getAliveTeamIds(this.bots);
        if (aliveTeamIds.length === 1) {
            this.winner = aliveTeamIds[0];
            this.endReason = 'team_eliminated';
            this.end();
        }
    }

    createTrajectoryMetadata() {
        const teamIds = Array.from(new Set(this.bots.map((bot) => bot.teamId)));
        const teams = teamIds.map((teamId) => ({
            teamId,
            playerIds: this.bots
                .filter((bot) => bot.teamId === teamId)
                .map((bot) => this.getActorId(bot))
        }));

        const players = this.bots.map((bot) => {
            return {
                id: this.getActorId(bot),
                teamId: bot.teamId,
                tacticId: null,
                policyId: bot.aiMethod ? 'heuristic' : 'neural_network',
                genomeId: null
            };
        });

        return {
            trajectoryId: `battle-${Date.now()}`,
            schemaVersion: '0.1.0',
            scenarioId: '2v2_default',
            seed: null,
            createdAt: new Date().toISOString(),
            teams,
            players
        };
    }

    getActorId(bot) {
        const teamIndex = this.bots.filter((candidate) => candidate.teamId === bot.teamId).indexOf(bot);
        return `${bot.teamId}-${teamIndex}`;
    }

    getHeading(bot) {
        const rotationRadians = bot.rotation * Math.PI / 180;
        return {
            headingX: Math.cos(rotationRadians),
            headingY: Math.sin(rotationRadians)
        };
    }

    getEnvironmentState(includeBounds = false) {
        const projectiles = this.bots.flatMap((bot) => bot.bullets.map((bullet) => {
            const rotationRadians = bullet.rotation * Math.PI / 180;
            return {
                id: bullet.id,
                shooterId: bullet.shooterId,
                shooterTeamId: bullet.shooterTeamId,
                positionX: bullet.xPos,
                positionY: bullet.yPos,
                headingX: Math.cos(rotationRadians),
                headingY: Math.sin(rotationRadians)
            };
        }));

        return {
            width: config.mapWidth,
            height: config.mapHeight,
            ...(includeBounds ? {
                bounds: {
                    minX: 0,
                    minY: 0,
                    maxX: config.mapWidth,
                    maxY: config.mapHeight
                }
            } : {}),
            activeProjectileCount: projectiles.length,
            projectiles,
            remainingTeams: getAliveTeamIds(this.bots)
        };
    }

    createInitialState() {
        return {
            environment: this.getEnvironmentState(true),
            players: this.bots.map((bot) => ({
                id: this.getActorId(bot),
                teamId: bot.teamId,
                positionX: bot.xPos,
                positionY: bot.yPos,
                ...this.getHeading(bot),
                hp: bot.lives,
                alive: bot.lives > 0
            }))
        };
    }

    getPlayerState(bot, action) {
        return {
            positionX: bot.xPos,
            positionY: bot.yPos,
            ...this.getHeading(bot),
            velocityX: action?.dx || 0,
            velocityY: action?.dy || 0,
            hp: bot.lives,
            alive: bot.lives > 0,
            // TODO: expose exact remaining cooldown from the simulation.
            cooldown: 0
        };
    }

    getPlayerAction(bot, action) {
        const heading = this.getHeading(bot);
        return {
            moveX: action?.dx || 0,
            moveY: action?.dy || 0,
            aimX: heading.headingX,
            aimY: heading.headingY,
            fire: action?.ds ? 1 : 0
        };
    }

    getPlayerMeasurements(bot) {
        const aliveBots = getAliveBots(this.bots);
        const sameTeamBots = aliveBots.filter((otherBot) => otherBot !== bot && !isEnemy(bot, otherBot));
        const enemyBots = aliveBots.filter((otherBot) => isEnemy(bot, otherBot));
        const nearestEnemy = this.getNearestEnemy(bot);
        const actorId = this.getActorId(bot);
        const nearestDistance = (candidates) => {
            if (!candidates.length) return 0;
            return candidates.reduce((nearest, candidate) => {
                const candidateDistance = distanceBetweenPoints(bot.xPos, bot.yPos, candidate.xPos, candidate.yPos);
                if (!nearest) return candidateDistance;
                return candidateDistance < nearest ? candidateDistance : nearest;
            }, 0);
        };

        return {
            nearestAllyDistance: nearestDistance(sameTeamBots),
            nearestEnemyDistance: nearestDistance(enemyBots),
            targetId: nearestEnemy ? this.getActorId(nearestEnemy) : null,
            targetDistance: nearestEnemy
                ? distanceBetweenPoints(bot.xPos, bot.yPos, nearestEnemy.xPos, nearestEnemy.yPos)
                : null,
            // TODO: expose the policy's selected target to calculate aim alignment.
            aimTargetAlignment: null,
            damageDealt: this.stepDamage?.[actorId]?.dealt || 0,
            damageTaken: this.stepDamage?.[actorId]?.taken || 0
        };
    }

    recordTrajectoryStep() {
        const playerRecords = this.bots
            .map((bot, index) => {
                if (bot.lives <= 0) {
                    return null;
                }

                const action = this.stepActions?.[index] || { dx: 0, dy: 0, dh: 0, ds: false };
                return {
                    actorId: this.getActorId(bot),
                    actorTeamId: bot.teamId,
                    state: this.getPlayerState(bot, action),
                    action: this.getPlayerAction(bot, action),
                    reason: createDefaultDecisionReason(),
                    measurements: this.getPlayerMeasurements(bot)
                };
            })
            .filter(Boolean);

        this.traceRecorder.recordStep({
            step: this.trajectoryStep,
            timeMs: Date.now() - this.startTime,
            environment: this.getEnvironmentState(),
            players: playerRecords,
            events: []
        });
        this.trajectoryStep += 1;
    }

    emitFrame() {
        if (typeof this.onFrame === 'function') {
            this.onFrame({
                bots: this.bots.map((bot) => ({
                    id: bot.id,
                    teamId: bot.teamId,
                    xPos: bot.xPos,
                    yPos: bot.yPos,
                    rotation: bot.rotation,
                    lives: bot.lives,
                    bullets: bot.bullets.map((bullet) => ({
                        id: bullet.id,
                        shooterId: bullet.shooterId,
                        shooterTeamId: bullet.shooterTeamId,
                        xPos: bullet.xPos,
                        yPos: bullet.yPos,
                        rotation: bullet.rotation
                    }))
                }))
            });
        }
    }

    /**
     * Returns a bullet object given a position and rotation
     */
    spawnBullet(bot) {
        const projectileId = `projectile-${this.projectileSequence}`;
        this.projectileSequence += 1;

        return {
            id: projectileId,
            shooterId: this.getActorId(bot),
            shooterTeamId: bot.teamId,
            xPos: bot.xPos,
            yPos: bot.yPos,
            rotation: bot.rotation
        };
    }
}

export default Battleground;
