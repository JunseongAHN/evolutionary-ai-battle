# Tactical Move Score Baseline

`TacticalMoveScorer` is a deterministic tactical movement baseline. It is not a ground-truth movement oracle: movement usually has several valid choices, so this scorer provides a reasonable, debuggable policy rather than a single correct answer.

The scorer enumerates every env `move_bin`, simulates the next candidate position, scores each candidate, and chooses the highest total score. The debug output keeps every term visible so poor movement can be traced to action convention, candidate simulation, obstacle or boundary checks, enemy spacing, or weight balance.

## Run Debug

```bash
python experiment/baselines/move_score/run_move_score_debug.py \
  --config configs/env/manual_enemy_far_right.yaml \
  --steps 10 \
  --print-score-breakdown

python experiment/baselines/move_score/run_move_score_debug.py \
  --config configs/env/manual_obstacle_front.yaml \
  --steps 10 \
  --save-png
```

## Score Terms

- Strong obstacle collision penalty.
- Strong map boundary penalty.
- Enemy spacing score toward an ideal range.
- Close enemy threat penalty.
- Optional strafe preference when the enemy is in range.

The env coordinate convention is reused from `training.cpc_actions`: `move_bin=0` is stay, +x is right, +y is down, and diagonal movement is normalized by the env decoder.

## Limitations

This baseline uses the nearest/current enemy only. It does not do multi-enemy tactics, ally support, complex pathfinding, neural imitation learning, PPO integration, or fire timing.

## Future Usage

1. Imitation learning label generator: observation -> TacticalMoveScorer -> move_bin label.
2. PPO auxiliary target: add temporary cross-entropy loss between learned move_head and scorer-selected move_bin.
3. Regression baseline: learned agents should not fail simple obstacle/boundary/range scenarios.
4. Hybrid controller: use aim oracle + move scorer + fire rule as a deterministic tactical baseline before replacing components with learned policies.
5. Baseline comparison: learned movement should outperform the scorer in complex scenarios, not merely copy it forever.
