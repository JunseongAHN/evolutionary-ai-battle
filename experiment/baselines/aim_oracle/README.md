# Aim Oracle Baseline

This aim oracle is a deterministic baseline and debugging tool. It is not the final learned policy.

Movement is fixed to `STAY`. Fire is fixed to `0`. The only intended behavior in this PR is enemy-cell-to-aim-bin mapping from the local occupancy grid.

## Why It Exists

The oracle makes aim bugs easy to attribute by isolating a short chain:

1. enemy placement in the local occupancy grid
2. grid cell to local vector conversion
3. local vector to `aim_bin` conversion
4. action sent to the env

It lives under `experiment/baselines/aim_oracle/` so the core env, schemas, training adapters, and production agent paths do not import it.

## Run

```bash
python experiment/baselines/aim_oracle/run_aim_oracle_debug.py \
  --config configs/env/manual_enemy_right.yaml \
  --steps 5 \
  --save-png
```

```bash
python experiment/baselines/aim_oracle/run_aim_oracle_debug.py \
  --config configs/env/manual_enemy_left.yaml \
  --steps 5 \
  --save-png
```

PNG output defaults to `experiment/runs/aim_oracle_debug`.

## Coordinate Conventions

Local occupancy grids are `cells[row_y][column_x][channel]`.

The center cell is the controlled agent position. Columns increase to the right, and rows increase with env/world y. The CPC toy env uses screen-style coordinates, so positive y is down.

`grid_cell_to_local_vector` maps:

```text
dx = (cell_x - center_x) * cell_size
dy = (cell_y - center_y) * cell_size
```

Aim bin `0` points right along `+x`. Bins follow the existing CPC action convention. Because positive y is down, increasing bins rotate clockwise on screen. With 16 bins:

```text
right = 0
down  = 4
left  = 8
up    = 12
```

## Known Limitations

The bot chooses the nearest active enemy cell in the local grid. It does not score tactical movement, choose firing windows, reason about obstacles, or predict future enemy motion.

The returned env action uses the existing CPC raw action schema:

```python
{"move": 0, "aim": computed_aim_bin, "fire": 0}
```

Debug output includes `enemy_cell`, `local_vector`, `aim_bin`, and `reason`.

## Future Usage

1. Imitation learning label generator: `local_occupancy_grid -> aim_oracle -> aim_bin label`
2. PPO auxiliary target: add cross-entropy loss between learned `aim_head` and oracle `aim_bin`
3. Hybrid controller: use oracle aim while RL learns movement/fire first
4. Debugging oracle: compare learned agent `aim_bin` with oracle `aim_bin` to attribute aim mistakes
5. Baseline comparison: learned policy should not perform worse than oracle on simple aim scenarios
