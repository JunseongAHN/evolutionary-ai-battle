# Tactical Baseline

The tactical baseline is a deterministic, debuggable local-combat bot. It combines three small pieces:

- `TacticalAimOracleBot`: maps the nearest enemy cell in the local occupancy grid to `aim_bin`.
- `TacticalMoveScorer`: scores candidate `move_bin` choices for spacing, boundary safety, obstacle safety, and strafing.
- `FireRule`: fires only when a live enemy is in weapon range, aim error is below the threshold, cooldown is ready, optional line of sight is clear, and optional ammo is available.

This lives under `experiment/baselines/tactical_baseline/` and is not imported by the core env. The env remains normally step based:

```python
obs, reward, done, info = env.step(action)
```

The bot returns the existing env action keys (`move`, `aim`, `fire`) plus debug-friendly aliases (`move_bin`, `aim_bin`).

## Run Autoplay

```bash
python experiment/baselines/tactical_baseline/run_tactical_autoplay.py \
  --config configs/env/autoplay_enemy_right.yaml \
  --steps 300 \
  --fps 10 \
  --render \
  --print-debug
```

```bash
python experiment/baselines/tactical_baseline/run_tactical_autoplay.py \
  --config configs/env/autoplay_obstacle_between.yaml \
  --steps 300 \
  --fps 10 \
  --save-png \
  --output-dir experiment/runs/tactical_autoplay_obstacle \
  --print-debug
```

`--render` reuses the existing pygame viewer when available. `--save-png` writes local occupancy grid frames. Console debug includes the selected action and separate aim, move, and fire reasons.

## Fire Rule Fallbacks

Missing enemy, range, aim, or cooldown information blocks firing. Missing line-of-sight information is treated as clear because line of sight is optional if unavailable. Missing ammo information does not block firing because the current toy env has no ammo field.

## Limitations

This baseline is not a final policy. Aim is oracle-like geometry. Movement is candidate scoring, not ground truth. Fire is a simple rule. The goal is to create a human-evaluable autoplay breakpoint before imitation learning or RL.

## Future IL/RL Usage

1. Trajectory collection: `obs -> tactical baseline -> action -> env.step -> JSONL dataset`.
2. Behavior cloning: train neural policy heads to imitate `move_bin`, `aim_bin`, and `fire` labels.
3. PPO hybrid controller: keep aim oracle fixed while RL learns movement/fire.
4. PPO auxiliary loss: temporarily regularize learned heads toward baseline actions.
5. Regression baseline: learned policies should not fail simple scenarios where this tactical baseline succeeds.
