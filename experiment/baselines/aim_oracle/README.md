# Aim Oracle Baseline

This aim oracle is a deterministic baseline and debugging tool. It is not the final learned policy.

Movement is fixed to `STAY`. Fire is fixed to `0`. Aim is a normalized continuous direction calculated from `enemy_pos - self_pos`.

## Why It Exists

The oracle makes aim bugs easy to attribute by isolating a short chain:

1. player and enemy world positions in the observation
2. enemy-relative direction calculation
3. continuous action decoding
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

The continuous aim direction is normalized before it is returned:

```text
right = ( 1,  0)
down  = ( 0,  1)
left  = (-1,  0)
up    = ( 0, -1)
```

## Known Limitations

When world positions are unavailable, the bot falls back to the nearest active enemy cell in the local grid. It does not predict future enemy motion.

The returned env action uses the existing CPC raw action schema:

```python
{"move": 0, "aim_dx": normalized_dx, "aim_dy": normalized_dy, "fire": 0}
```

Debug output includes `enemy_cell`, `local_vector`, `aim_direction`, `aim_source`, and `reason`.

## Future Usage

1. Imitation learning label generator: `positions -> aim_oracle -> continuous direction label`
2. PPO auxiliary target: regress or cosine-match the learned aim direction to the oracle
3. Hybrid controller: use oracle aim while RL learns movement/fire first
4. Debugging oracle: compare learned and oracle aim vectors to attribute aim mistakes
5. Baseline comparison: learned policy should not perform worse than oracle on simple aim scenarios
