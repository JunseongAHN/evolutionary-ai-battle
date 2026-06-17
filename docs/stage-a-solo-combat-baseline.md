# Stage A: Solo Combat Baseline

## Goal

Compare future agents against random and user-controlled human baselines in a toy Survev-like solo combat environment.

## Why Human Baseline Matters

Random baseline is too weak.
A meaningful Stage A agent should eventually be compared against user-controlled runs under the same seeds and action constraints.

## Current Scope

This stage only records and compares baselines.
It does not implement a new combat policy, RL, or training.

## Metrics

- damageDealt
- damageTaken
- survivalSteps
- aliveAtEnd
- kills/deaths if available
- wastefulFireCount if available

Cooperation metrics are not applicable in solo mode.

## Running Baselines

Use the Stage A Solo Combat Baseline panel in the browser simulation page.

- Run Random Solo starts a solo battle where every player uses the random policy.
- Run User-Controlled Solo starts a solo battle where the first player is user-controlled and the remaining players use the random policy.
- User controls are WASD or arrow keys for movement, Q/E for rotation, and Space to fire.
- Export Baselines JSON downloads the collected baseline rows, aggregate summaries, and same-seed groups.

## Baseline Comparison Shape

Each comparison row stores seed, runId, policyType, playerId, damageDealt, damageTaken, survivalSteps, aliveAtEnd, optional kills/deaths, optional wastefulFireCount, and cooperation applicability.

Aggregate summaries group by policyType and include runCount, average damage, average survival, survival rate, and participating seeds.

## Success Criteria for Future Policies

A future policy should:

- clearly beat random baseline,
- approach or exceed user-controlled baseline on selected metrics,
- respond consistently to visible enemies,
- avoid obviously bad low-HP aggression.

