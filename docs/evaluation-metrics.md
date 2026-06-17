# Evaluation Metrics

## Evaluation Philosophy

Current evaluation is metric-vector based.

We intentionally do not collapse all metrics into a single scalar reward yet. The goal is to diagnose bot behavior across combat, cooperation, survival, and movement dimensions.

Scalar reward composition will be added later as a separate policy-optimization layer after the metrics are stable and interpretable.

## Report Shape

`evaluateTrajectory` returns independent metric groups:

```ts
EvaluationReport = {
  players: PlayerEvaluationMetrics[],
  teams: TeamEvaluationMetrics[],
  metadata: {
    totalSteps,
    battleId,
    trajectoryId,
    schemaVersion,
    createdAt
  }
}
```

Each player row has:

```ts
PlayerEvaluationMetrics = {
  teamId,
  playerId,
  policyName,
  policyType,
  combat,
  survival,
  cooperation,
  movement
}
```

## Combat

```ts
damageDealt[player] =
  sum(step.players[player].measurements.damageDealt)

damageTaken[player] =
  sum(step.players[player].measurements.damageTaken)

kills[player] =
  count(event where event.type in ["death", "elimination"] and event.actorId == player.id)

deaths[player] =
  count(event where event.type in ["death", "elimination"] and event.targetId == player.id)

friendlyFireDamage[player] =
  sum(event.damage where event.actorId == player.id and event.actorTeamId == event.targetTeamId)

wastefulFireCount[player] =
  count(step where action.fire > 0 and measurements.didFire == false)
```

These metrics are reported independently. Damage is not automatically converted into reward.

## Survival

```ts
aliveSteps[player] =
  count(step where player.state.alive == true and player.state.hp > 0)

aliveAtEnd[player] =
  lastStep.player.state.alive == true and lastStep.player.state.hp > 0

died[player] =
  deaths[player] > 0 or aliveAtEnd[player] == false
```

Survival is shown as behavior evidence, not a weighted bonus.

## Cooperation

```ts
teammateUnderPressureEvents[player] =
  count(step where an alive teammate has low hp and a nearby enemy)

supportResponses[player] =
  count(pressure event where player moved closer, fired while nearby, or had support-like reason)

teammateResponseRate =
  supportResponses / max(1, teammateUnderPressureEvents)

isolatedSteps =
  count(alive step where nearest ally distance > isolationDistanceThreshold)

isolationRate =
  isolatedSteps / max(1, aliveStepsWithAllyDistance)

avgAllyDistance =
  sum(nearestAllyDistance over alive steps) / max(1, aliveStepsWithAllyDistance)

formationGoodSteps =
  count(alive step where nearest ally distance <= supportDistanceThreshold)

allyAbandonmentEvents =
  currently 0 unless a trajectory provides a dedicated signal
```

Cooperation metrics are intentionally separate. A bot can have high damage and high isolation, or low damage and strong support responses; both patterns should remain visible.

## Movement

```ts
stuckSteps =
  count(alive step where velocity magnitude is effectively zero)

movementPenaltyEvents =
  currently 0 unless a trajectory provides a dedicated signal
```

## Team Aggregates

Team metrics are independent aggregates:

```ts
totalDamageDealt =
  sum(player.combat.damageDealt for players in team)

totalDamageTaken =
  sum(player.combat.damageTaken for players in team)

totalKills =
  sum(player.combat.kills for players in team)

totalDeaths =
  sum(player.combat.deaths for players in team)

avgTeammateResponseRate =
  mean(player.cooperation.teammateResponseRate for players in team)

avgIsolationRate =
  mean(player.cooperation.isolationRate for players in team)

avgAllyDistance =
  mean(player.cooperation.avgAllyDistance for players in team)

totalAliveSteps =
  sum(player.survival.aliveSteps for players in team)

alivePlayersAtEnd =
  count(player where player.survival.aliveAtEnd)
```

Team aggregates are not weighted rewards.

## Deprecated Scalar Helper

`computeEvaluationScoreBreakdown` remains only as a deprecated compatibility helper for older callers. It is not part of the main evaluation report and should not be displayed as the primary dashboard result.
