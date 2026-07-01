# Tactical Baseline

The tactical baseline is a deterministic teacher policy for local CPC combat. It keeps tactical selection and execution separate:

1. `RuleBasedTacticalModeSelector` chooses `engage`, `kite`, `hold_range`, or `reposition`.
2. `ModeConditionedBFSPlanner` builds an inflated blocked grid, finds reachable cells, scores them for that mode, and converts the selected path's first step to `move_bin`.
3. `TacticalAimOracleBot` supplies rule-based oracle aim.
4. `FireRule` gates firing on enemy state, range, aim, cooldown, and line of sight.

This code stays under `experiment/baselines/tactical_baseline/`; the core env does not import it. It does not add PPO, TorchRL, or training dependencies.

## Action And Debug Contract

The env action uses discrete movement with continuous aim: `{"move": ..., "aim_dx": ..., "aim_dy": ..., "fire": ...}`. The bot keeps `move_bin` as a movement debug alias.

Combined debug has independent `mode`, `move`, `aim`, `fire`, and `action` sections. Movement debug includes:

- `tactical_mode`, `enemy_cell`, `target_cell`, `next_cell`, and `move_bin`
- the inclusive BFS `path`
- `reachable_count`, blocked/reachable cells, and deterministic top candidates
- selected score terms for distance, ideal range, LOS, strafe, open space, obstacle clearance, boundary safety, path length, and staying

Grid coordinates are `[row_y, column_x]`: positive y moves down and positive x moves right. Obstacles and map boundaries are inflated by one cell for clearance. Diagonal BFS steps cannot pass between blocked corners.

## Tactical Modes

- `engage`: reduce distance toward weapon/ideal range while preserving LOS.
- `kite`: increase spacing and prefer open cells away from boundaries and corners.
- `hold_range`: stay near ideal range, preserve LOS, and prefer a perpendicular first step.
- `reposition`: restore LOS and seek open, obstacle-cleared space.

The selector is intentionally small and rule based. It is stable baseline infrastructure, not a final policy.

## Run Autoplay

```bash
python experiment/baselines/tactical_baseline/run_tactical_autoplay.py \
  --config configs/env/autoplay_enemy_in_range.yaml \
  --steps 300 \
  --fps 10 \
  --render \
  --print-debug \
  --show-tactical-debug
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

`--render` reuses the pygame viewer. `--print-debug` prints every step's `tactical_mode`, `target_cell`, `next_cell`, `move_bin`, continuous `aim_dir`, and `fire` to the console; `--print-every N` can reduce its frequency. `--show-tactical-debug` shows the same fields in the render side panel. Both debug outputs are off unless their option is supplied. `--save-png` writes local occupancy grids. The scenario enemies are stationary so movement behavior remains easy to inspect.

## Manual Checklist

- [ ] enemy far -> engage -> approach
- [ ] enemy close -> kite -> backoff/strafe
- [ ] enemy in range -> hold_range -> strafe/maintain distance
- [ ] LOS blocked -> reposition -> seek LOS/open space
- [ ] obstacle between -> does not push into obstacle repeatedly
- [ ] target_cell and path shown in debug
- [ ] move_bin matches first path step

## Limitations

The local planner reasons over a coarse occupancy grid and one visible enemy. Aim remains oracle-like, and tactical mode selection has no item, farming, zone, or team behavior. A target outside the local footprint is projected to the nearest grid edge. PNG output currently shows occupancy; route details are available in console/debug data.

## Future IL/RL Usage

The baseline can later label `tactical_mode`, `target_cell`, `path`, `next_cell`, and `move_bin` for imitation learning. PPO can eventually replace tactical mode selection when item, farming, zone, and team decisions make the rules unwieldy; the mode-conditioned planner can remain an interpretable executor and regression baseline.
