# Stage 1 Combat-Only Reward

Stage 1 is a local combat micro environment. The stadium is fixed and safe-zone shrink is disabled so failures can be attributed to aim, projectile use, dodging, range control, and damage trade rather than macro rotation.

## Reward Formula

The main signal is:

```text
combat_score = damage_dealt_ratio - damage_taken_ratio
```

Per-step reward components:

- `damage_dealt_ratio = 1.0 * enemy_damage_delta / enemy_max_hp`
- `damage_taken_ratio = -1.2 * self_damage_delta / self_max_hp`
- `bullet_hit = 0.05` on an actual projectile hit
- `missed_shot = -0.03` when an actual fired projectile expires without hit
- `aim_bin_exact = 0.04` for exact aim-bin match
- `aim_bin_wrong = -0.04` only when a shot is fired with aim error >= 2
- `good_range = 0.01`, `too_close = -0.03`, `too_far = -0.01`
- `kill = 1.0`, `death = -1.0`
- `timeout_hp_lead = 0.5 * (total_damage_dealt_ratio - total_damage_taken_ratio)`
- `accuracy_bonus = 0.2 * hit_ratio * min(total_damage_dealt_ratio, 1.0)` at timeout

Aim and range shaping are intentionally small. A policy should not earn high return by aiming correctly without dealing damage.

## Metrics To Improve

- `damage_dealt_ratio`
- `damage_trade_ratio`
- `hit_ratio`
- `bullet_hit_per_shot`
- `missed_shot_rate`
- `exact_aim_match_rate`
- `good_range_rate`
- `self_hp_remaining_ratio`

## Reward Hacking Warnings

- High return but low `damage_dealt_ratio`
- High `hit_ratio` with very low `shot_fired_count` and low damage
- High aim accuracy but low `bullet_hit_per_shot`
- Low `damage_taken_ratio` only because the bot avoids combat entirely
- High survival with no enemy damage
