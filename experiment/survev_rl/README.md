# survev_rl — PPO against the survev CPC bridge

Python side of the RL setup for `duo2v2_field`: a PPO agent (shared policy for the two
controlled team-a agents) that fights the server's built-in scripted `chaser` duo, as the
first opponent/strength baseline. Everything talks to the game server through the
WebSocket bridge described in `docs/survev-bridge-v0.md` (protocol v0); a Python mock of
that bridge is included so the whole stack runs and is tested without the TS server.

Hard dependencies: `torch`, `numpy`, `websockets` (>= 13; 15.x tested). `pytest` for tests,
`onnxruntime` optional (ONNX round-trip test). No gymnasium / torchrl / tensordict.

## Layout

| file | purpose |
|---|---|
| `protocol.py` | typed protocol v0: `CpcAction`, `AgentObservation` (TypedDicts), `Event`, `ObsMessage`, `Metrics`, `Input` name<->number table, JSON builders/parsers, observation key **allowlist** validator |
| `bridge_client.py` | synchronous `BridgeClient` on `websockets.sync.client`: `reset`, `step`, `step_batch`, `close`, context manager, timeouts, `BridgeError` |
| `mock_bridge.py` | mock bridge server + `FieldSim` kinematic 2v2 simulation + scripted `chaser`/`idle`; `serve()`, `MockBridgeThread` |
| `featurizer.py` | `Featurizer(FeaturizerConfig)`: `AgentObservation -> float32[216]` with `vector_keys`, nearest-K slots + presence masks, optional facing-frame rotation, per-agent last-seen enemy memory |
| `actions.py` | `ActionSpace(mode="primitive"|"skill")`: MultiDiscrete `[9, 16, 2, 2]` -> `CpcAction` with the assist layer; skill mode mapping (`{"skill", "params"}`) |
| `rewards.py` | `RewardConfig` + `compute_rewards` / `compute_reward_breakdown` (per controlled agent, `team_mix`) |
| `env.py` | `SurvevVecEnv` (batched step, auto-reset, `infos[row]["episode_metrics"]`), `SurvevSingleEnv`, `EnvConfig`, `make_vec_env` |
| `ppo.py` | CleanRL-style PPO for MultiDiscrete: `ActorCritic` (obs -> 256 -> 256, one head per dim + value MLP), GAE, clipping, entropy, value clip, adv-norm, grad clip, LR anneal, CSV/JSONL/TensorBoard logging, checkpoints |
| `train_ppo.py` | training CLI (`--mock` for the in-process mock) |
| `eval.py` | evaluation CLI: win rate / survival / HP / damage over N episodes + per-episode JSONL, `to_harness_episode()` stub |
| `export_onnx.py` | actor -> ONNX (dynamic batch) + sidecar JSON with head offsets / bins / vector keys |
| `policy_server.py` | the inverse of the bridge: serves a checkpoint over WebSocket (`reset` / `act` with spec observations -> `CpcAction` wire form) so a live, client-rendered game can ask Python for the agent's actions |
| `tests/` | pytest suite (< 10 s) |

## Running on the Windows machine

```bat
conda activate agentic-ai
cd C:\repos\survev
pnpm cpc:bridge                      :: bridge on ws://127.0.0.1:8765 (leave running)

cd C:\repos\evolutionary-ai-battle
python -m experiment.survev_rl.train_ppo --bridge ws://127.0.0.1:8765 --n-envs 16 --ticks 10 ^
    --total-steps 2000000 --device cuda --out runs\ppo_v0
python -m experiment.survev_rl.eval --checkpoint runs\ppo_v0\checkpoint_final.pt ^
    --bridge ws://127.0.0.1:8765 --episodes 50 --out runs\ppo_v0\eval.jsonl
python -m experiment.survev_rl.export_onnx --checkpoint runs\ppo_v0\checkpoint_final.pt --out runs\ppo_v0\actor.onnx
python -m experiment.survev_rl.policy_server --checkpoint runs\ppo_v0\checkpoint_final.pt --port 8766
                                     :: a game process (survev live hook) connects and asks for actions
```

The bridge must include survev commit `5f118a6c` ("accept Input names in CpcAction"): before it,
the named inputs this client sends (`Interact`, `EquipPrimary`, `Reload`) were silently dropped by
the engine, so bridge-trained agents could not pick up, equip or reload (the `real_smoke`/first
`ppo_ep2` runs were affected; every number below is from the fixed bridge).

Mock mode (no game server; the mock runs in a thread of the same process):

```bat
python -m experiment.survev_rl.train_ppo --mock --n-envs 8 --total-steps 200000 --device cpu --out runs\mock
python -m experiment.survev_rl.eval --mock --checkpoint runs\mock\checkpoint_final.pt --episodes 20
python -m experiment.survev_rl.eval --mock --policy random --episodes 20         :: random baseline
python -m experiment.survev_rl.mock_bridge --port 8765                          :: standalone mock server
python -m pytest experiment\survev_rl\tests -q
```

Notes: run from the repo root so `experiment.core.cpc_actions` imports; `--total-steps`
counts agent transitions (rows x vec steps; with 16 envs x 2 controlled that is 32 per step);
the bridge hosts all envs of one connection in one process, so for more throughput start
several bridge processes (one per core) and one training process per bridge, or raise
`--n-envs` until the bridge is the bottleneck. `--resume <ckpt>` continues a run.

## Configuration

### `train_ppo.py` flags

| flag | default | meaning |
|---|---|---|
| `--bridge` / `--mock` / `--mock-port` | `ws://127.0.0.1:8765` | bridge URL, or start the mock in-process |
| `--n-envs` | 16 | envs on one connection (batched step) |
| `--ticks` | 10 | game ticks per decision (1 tick = 0.01 s; 10 = 0.1 s policy cadence) |
| `--controlled` | `team-a-0,team-a-1` | agents driven by the policy; the rest use `--scripted` (`chaser`/`idle`) |
| `--time-limit` / `--map-size` | 60 / 128 | reset options |
| `--loadout` | `fists` | `armed` spawns everyone with an ak47 + 90 rounds (curriculum). Reset option `loadout` (bridge + mock) |
| `--layout` | `fixed` | `random` rotates the spawn axis and draws the duo-to-center distance from [24, 44] u per seed (reset option `layout`, bridge; the mock accepts it but keeps its fixed geometry). Use it for any training run: on the fixed layout PPO learned to aim at the constant spawn direction |
| `--goal` | – | waypoint `center` (132,132) or `x,y`: appends the 6-float goal block to the observation and enables the waypoint reward terms (`goal_progress`, `goal_hold`, `goal_radius`, `enemy_at_goal`, `enemy_goal_radius` in `--reward-json`; presets in `configs/point_v1*.json`). The goal is stored in the checkpoint meta, so `eval` and `policy_server` reuse it |
| `--seed` | 0 | torch/numpy seed and base of the episode seeds (`cpc-duo2v2-seed-<base+env+n_envs*episode>`) |
| `--rotate-obs` | off | egocentric rotation into the facing frame (see featurizer) |
| `--no-memory` | off | drop the last-seen enemy memory block |
| `--no-assist` / `--auto-pickup` | assist on / pickup off | assist layer switches (see actions) |
| `--team-mix` | 0.0 | blend own reward with the team-mean reward |
| `--reward-json` | – | JSON file or inline JSON overriding `RewardConfig` fields |
| `--total-steps --rollout-steps --minibatches --epochs --lr --gamma --gae-lambda --clip --ent-coef --vf-coef --max-grad-norm --target-kl --hidden` | 2M / 128 / 4 / 4 / 3e-4 / 0.99 / 0.95 / 0.2 / 0.01 / 0.5 / 0.5 / none / 256 | PPO |
| `--device` | `cpu` | `cuda` / `cpu` / `auto` |
| `--checkpoint-every` / `--tensorboard` / `--resume` / `--out` | 10 / off / – / `runs/ppo_v0` | logging and checkpoints |

Outputs in `--out`: `config.json`, `progress.csv` + `log.jsonl` (per update: losses,
`approx_kl`, `clip_fraction`, `explained_variance`, and the window of episode metrics:
`win_rate`, `episode_return`, `survival_time`, `hp_mean`, `hp_end`, `damage_dealt`,
`damage_taken`, `kills`, ...), `episodes.jsonl` (one line per finished episode per agent),
`checkpoint_latest.pt` / `checkpoint_final.pt` (state dicts + all configs + `vector_keys`).

### `FeaturizerConfig`

| field | default | meaning |
|---|---|---|
| `max_teammates / max_enemies / max_loot / max_obstacles / max_bullets` | 1 / 3 / 6 / 4 / 4 | nearest-K slots |
| `rotate_to_facing` | False | rotate relative vectors into the facing frame (facing = +x) |
| `pos_scale` | 32 | divisor for relative positions (culling half-width at 1x) |
| `far_scale` | 128 | divisor for absolute position and gas distances |
| `clip_value` | 3 | clip for scaled distances |
| `memory` / `memory_decay_s` | True / 10 | last-seen enemy memory block and its recency time constant |
| `time_limit` | 60 | for `self_time_frac` |

### `ActionSpace`

`mode="primitive"` (default): MultiDiscrete `[9, 16, 2, 2]` = move bin (harness
`MOVE_VECTORS`, 0 = still), aim bin (harness `aim_bin_to_vec`, absolute world angle,
22.5 deg), fire (`fire.hold`), interact (`Interact` input). `assist=True` adds
`EquipPrimary`/`EquipSecondary` when a gun is carried but fists are equipped and `Reload`
when the equipped clip is empty (reserve > 0, not already reloading). `auto_pickup=True`
(off) also presses `Interact` whenever loot is within 2 u (curriculum aid).

`mode="skill"`: Discrete over `move_to_partner, loot_nearest, engage, retreat, take_cover,
revive, hold`; `to_skill_action(idx, obs)` yields `{"skill": name, "params": {...}}` with
targets resolved from the observation. Not executable through the bridge yet (see below).

### `RewardConfig` (defaults = the v0 decision; one step = 0.1 s at `ticks=10`)

| weight | default | applied to |
|---|---|---|
| `alive_per_step` | +0.01 | every step the agent is not dead (downed counts as alive) |
| `hp_delta` | 0.01 | change of *effective* HP (0 while downed/dead): damage is negative, heals positive, a down = losing what was left, a revive = +24 |
| `damage_dealt` | 0.02 | per HP of damage dealt to enemies (`damage` events with `source == agent`) |
| `damage_taken` | 0 | per HP taken (use a negative weight; `hp_delta` already covers it) |
| `kill` | 0 | per enemy `kill` event with `source == agent` |
| `death` | -1.0 | the agent's own `kill` event |
| `team_win` | +1.0 | at `done` when `info.winner_team` is the agent's team |
| `partner_hp_delta` / `partner_alive_per_step` | 0 / 0 | cooperation terms on the teammate's effective HP / survival |
| `cover_bonus` | 0 | placeholder (0 until obstacle/LOS features exist) |
| `time_penalty_after_s` / `time_penalty_per_step` | 0 / 0 | per-step penalty once `t` passes the threshold |
| `team_mix` | 0 | `r_i <- (1-mix) r_i + mix mean_team(r)` over controlled teammates |
| `goal_progress` | 0 | per world unit of distance to the waypoint removed this step (standing agents only; needs `--goal`) |
| `goal_hold` / `goal_radius` | 0 / 6 u | per step while standing within `goal_radius` of the waypoint |
| `enemy_at_goal` / `enemy_goal_radius` | 0 / 30 u | per step x sum over living enemies of max(0, 1 - d(enemy, waypoint) / radius), from the enemies' true positions (reward side only). Negative weight = the 'keep them off the point' pressure that only killing or repelling removes |

Rewards are for optimisation only. Evaluation and logging use the metric vector
(`info.metrics`: survival_time, hp_mean/hp_end, damage_dealt/taken, team_win, partner
survival, ...) exactly as the harness does — nothing is collapsed into the scalar.

## Observation vector (216 floats; `Featurizer.describe()` / `vector_keys`)

| block | slots | size | features |
|---|---|---|---|
| `self` | 1 | 31 | hp/100, boost/100, downed, dead, dir(x,y), abs pos (x,y from map center / 128), weapon class one-hot (fists, ak47, mp5, other_gun), has_primary, has_secondary, gun_equipped, clip/30, reserve/90, action one-hot (none, reload, use_item, revive), inventory x8 (bag caps: bandage 5, healthkit 1, soda 2, painkiller 1, 762mm 90, 9mm 120, 12gauge 15, 556mm 90), scope (zoom-28)/40, time fraction |
| `tm{i}` | 1 | 9 | present, dx, dy, dist, unit(x,y), hp/100, downed, dead |
| `en{i}` | 3 | 10 | present, dx, dy, dist, unit(x,y), facing dir(x,y), downed, armed |
| `loot{i}` | 6 | 11 | present, dx, dy, dist, class one-hot (gun, ammo, heal, armor, scope, other), count/30 |
| `ob{i}` | 4 | 8 | present, dx, dy, dist, collidable, scale, `los_blocked`*, `cover_score`* |
| `bl{i}` | 4 | 6 | present, dx, dy, dir(x,y), approaching |
| `gas` | 1 | 7 | active, inside, shrinking, edge distance, new-center unit(x,y), new-edge distance |
| `counts` | 1 | 5 | alive_count/4, alive_teams/2, enemies/K, loot/K, bullets/K |
| `mem{i}` | 3 | 4 | seen, last dx, last dy, recency = exp(-age/10 s) |

`*` placeholders (always 0 today). Relative positions are `(target - self) / 32`, clipped to
±3; with `rotate_to_facing` all relative vectors and directions are expressed in the
facing frame (default off because the aim bins are absolute). When more than one agent is
controlled the env appends an agent-id one-hot (`agent_is_<id>`), so the default training
observation is **218** floats. Action space: MultiDiscrete `[9, 16, 2, 2]` (29 logits).

## Mock bridge

`mock_bridge.FieldSim` reproduces the parts of the engine that matter for training
(see the module docstring for the full list and the constants for every number that is
not in the spec): 100 Hz ticks with held inputs, 12 u/s (+1 with fists) 8-direction
movement, seeded loot layout with the spec's item list, `Interact` pickup within 2 u
(gun -> slot with a full 30-round clip and auto-equip when holding fists, ammo/heals ->
inventory with level-0 bag caps, armor, scopes), firing with 0.10/0.11 s fireDelay and
13/9 damage (ak47/mp5), projectile bullets with angular spread (hit probability falls with
distance), armor reduction, auto-reload on empty, duo down/kill semantics, bleeding,
revive (5 u, 8 s, 24 HP), heals/boost, `fire/damage/down/kill` (+`heal`/`revive`) events,
`done`/`info`/metrics, a (zoom + 4) x 16:9 culling rectangle per agent, and the scripted
`chaser` (loot nearest gun/ammo -> approach to 22 u -> strafe -> hold fire within 30 u ->
revive when no enemy within 25 u; decides every 3 ticks from its own culled view).
It runs at roughly 200-300x real time per env in pure Python. It is a development stand-in,
not a physics clone: expect numbers (TTK, hit rates, timings) to differ from the real server.

## Extensibility

* **Cooperation rewards (co-playable teammate).** `RewardConfig.partner_hp_delta` and
  `partner_alive_per_step` already score the partner's effective HP (a revive counts +24
  for the reviver) and survival from the `teammates` HUD entries; `team_mix` shares
  credit. Staying near / reviving can be added as further components in
  `rewards.compute_reward_components` (the teammate slot `tm0_dist` is in the
  observation; `revive` events are already parsed). The featurizer's teammate block and
  the `revive` skill exist for the same reason.
* **Cover / obstacles.** `AgentObservation.obstacles` is featurized into 4 nearest slots
  with `collidable`, `scale`, and two placeholder features (`los_blocked`, `cover_score`)
  that keep the vector layout fixed. When the map has obstacles, fill them in
  `featurizer.py` (ray-cast against the obstacle list) and give `RewardConfig.cover_bonus`
  a term in `rewards._cover_term`. `ActionSpace` skill mode already has `take_cover`.
* **Skill mode.** `ActionSpace(mode="skill")` maps a Discrete choice over 7 named skills
  to `{"skill": name, "params": {...}}` with targets resolved from the observation. The
  bridge does not execute skills yet: once the TS System 1 skills land, add a `skill`
  field to the step message (or a `skill` action type) and route
  `to_skill_action(...)` there instead of `to_cpc_action(...)` in `SurvevVecEnv.step`.
  `ActorCritic` is generic over `nvec`, so PPO runs unchanged with `nvec = (7,)`.
* **ONNX -> onnxruntime-node.** `export_onnx.py` writes `actor.onnx` (input `obs[batch,
  obs_dim]`, output `logits[batch, 29]`, optional `value`) plus `actor.json` with head
  offsets, `move_vectors`, `aim_bins`, `vector_keys` and the featurizer config. The TS
  side re-implements `featurizer.py` from `vector_keys` (same order), runs the session,
  and per head samples/argmaxes `logits[offset:offset+size]`; move via `move_vectors`,
  aim via `aim_bins`, fire/interact as booleans, then applies the same assist rules.
* **Harness export.** `eval.py` writes per-episode JSONL (`obs` summaries or full
  observations with `--full-obs`, raw + `CpcAction` actions, rewards, events, metrics);
  `to_harness_episode()` maps the episode-level fields and `final_metrics` into the
  `EpisodeTrajectory` shape and leaves `steps` for the bridge's S6 JSONL export.

## What the mock runs showed (CPU, 200-300k agent steps, 8 envs)

* v0 defaults vs `chaser`: the policy learns to **flee** (survival 6 s -> 59 s, return 5.9,
  0 shots): fists give 13 u/s vs 12 u/s for armed chasers and `alive_per_step` x 600 steps
  outweighs everything else, so hiding at the map edge is optimal. 0 wins.
* no alive bonus + time penalty after 20 s (`--reward-json`) + `--auto-pickup`: still 0 shots
  in 300k steps; evasion remains the easy optimum, looting + aiming is a long sparse chain.
* vs `idle` opponents with `--auto-pickup`: a few shots early, then collapse to passive (return =
  alive bonus): the loot -> approach -> aim chain is too long for random exploration.
* **positive control** `--loadout armed --scripted idle --time-limit 30` (200k steps, 2.5 min on
  2 CPU cores): 85% win rate, 130 damage per agent, eliminations in ~15 s — the obs -> aim bin ->
  fire -> win pathway is learnable end to end. Against `chaser` the same policy shoots (32
  shots, 67 damage) but loses in 5.6 s.
* Suggested curriculum: `--loadout armed` + idle -> `--loadout armed` + chaser -> fists + chaser,
  with a larger `damage_dealt`/`kill` weight or a time penalty (`--reward-json`) to counter the
  flee optimum. These are mock numbers — the real server's TTK and hit model differ.

## What the real bridge runs showed (fixed bridge, 600k agent steps, 8 envs, CPU, ~5 min each)

Rendered frames of both runs are in `out/ep2/` and `out/ep2_armed/` (real client, camera player,
0.5 s / 0.2 s cadence; `live_episode.json` holds per-tick positions, HP, events and every PPO decision).

* **v0 defaults** (fists, `chaser`, default `RewardConfig`): survival 6 s -> 19.8 s, then a
  plateau; 0 shots, 0 damage dealt, win 0 %. The policy runs south-west from the first step,
  reaches the beach at ~5 s, the water at ~8 s, gets pinned in the map corner (1, 1) at ~12 s and
  is shot there (`out/ep2`). Fleeing is the local optimum of alive +0.01 / death -1; the
  loot -> equip -> aim chain is never explored on the real server either.
* **`--loadout armed`** (everyone spawns with an ak47, same reward): win 91 %, ~6 s episodes,
  hp_end 78, 120 damage dealt / 36 taken, 19 shots -> 8.5 hits. But the frames show *what* was
  learned (`out/ep2_armed`): stand still, aim east (aim bin 0) and hold fire from t = 0 — the
  chasers spawn on the same y and run straight into the stream (chaser fires only inside 30 u,
  so it usually dies first). When one survives and closes in, the policy switches to the flee
  behaviour (move west, no fire, random aim) and dies at the west edge. It is a spawn-geometry
  exploit, not tracking of a moving target.
* Consequences for the next iteration: randomize spawn sides / orientation and the loot layout
  per episode (seeded), give the featurizer a relative aim error toward the nearest enemy (or an
  aim-at-enemy assist), and use the harness `chaser` as a *curriculum* opponent rather than the
  benchmark — the metric vector (survival, HP, damage, team win, partner survival) is already
  produced per episode, so `eval.py` output can go straight into the harness comparison.
* Coordinates: `MOVE_LABELS` are the harness names with screen-y-down (`up` = (0, -1)); in survev
  world coordinates y grows upward on screen, so the label `up_left` is a south-west move. Use
  the vectors, not the names, when reading decisions.

## The "global point" objective (2026-09-09, random layout, 400k agent steps each, 2 CPU cores)

Objective proposed for v1: one global point (the field center; `--goal center`), reward for going
there, penalty as enemies get close to it, nothing else — the expectation being that killing the
enemies (the only way to remove the penalty) would make the agent farm a gun and fight.
Presets: `configs/point_v1.json` (progress 0.05/u, hold 0.01/step within 6 u, enemy-at-point
-0.02 x pressure), `point_v1_damage.json` (+ damage_dealt 0.02), `point_v2.json` (see below).
Frames: `out/ep3` (point_v1) and `out/ep4` (point_v2, armed).

| run | loadout | extras | survival | shots / hits | dmg dealt | hold frac | win | behaviour |
|---|---|---|---|---|---|---|---|---|
| A `point_v1` | fists | – | 4.6 s | 0.8 / 0.05 | 0.5 | 0.40 | 0 % | sprint to the point past the kit, die on it |
| B `point_v1_damage` | fists | – | 4.7 s | 1.5 / 0.1 | 0.9 | 0.40 | 0 % | same |
| C `point_v1_damage` | fists | `--auto-pickup` | 4.7 s | 3 / 0.2 | 2 | 0.41 | 0 % | same, a few shots on the way |
| D `point_v1_damage` | armed | – | 3.7 s | 17 / 2.5 | 32 | 0.00 | 2 % | sprint toward the point, brawl 10 u short of it, lose |
| E `point_v2` | armed | – | 17.7 s | 67 / 5.2 | 68 | 0.00 | 18 % | never approaches the point; retreats while shooting (kiting), still improving at 400k |

Why A-D collapse to "rush and die": `goal_progress` is potential-based and gets banked in the
first 2 s (+1.4), and death *ends* the `enemy_at_goal` stream, so dying on the point right after
banking the progress is the best return (~1.2) reachable without already knowing how to fight.
Staying alive on the point with enemies around costs -0.02 x pressure per step, i.e. more than
the hold bonus. The pressure term cannot teach shooting by itself: the credit path
(gun -> aim -> ~8 hits -> enemy dead -> penalty gone) is far too long for exploration.

`point_v2` keeps only the non-bankable terms — hold (+), enemy-at-point (-), damage dealt (+) —
and makes death (-1) cost more than the pressure an agent actually accumulates while it stays
alive (pressure follows the enemies, and the chasers follow the agent away from the point), so
ending the episode is no longer an escape; the options order as fight > flee > suicide. With guns in hand that is enough to start learning a real aim-and-fire
behaviour against moving targets on the random layout (run E: 8 -> 68 damage per agent over
400k steps, no spawn-direction exploit possible), but the agent then ignores the point: approaching
it means approaching the chasers. Farming never appears in any run because the kit is only picked
up by chance.

What this says about "can a completely simple reward get farm -> fight": not in one stage. The
minimal path that is consistent with the evidence is a curriculum with the same two-to-four
terms and one change per stage: (1) armed + random layout + `damage_dealt`/`death` until the
kiting fighter is stable (run E, more steps; 4 bridge processes on the Windows box make 2M steps
~15 min); (2) `--resume` it with fists + `--auto-pickup` — damage is now only reachable through
the kit, so farming is the first thing it has to learn; (3) add `goal_hold` / `enemy_at_goal` to
pull the fight onto the point. Spawn randomization stays on everywhere.

## The race objective (2026-09-09, `--objective race`, random layout, chaser opponent)

Design (준성's proposal, refined): one shared point at a time, +1 to the *team* whose member touches
it first, the point then moves 30-70 u away; points keep coming for the whole 60 s, also after a
wipe (`endOnElimination: false`), so dying forfeits every remaining point — the death penalty is
implicit. `configs/race_v1.json` = capture 1.0 + damage_dealt 0.02, everything else 0. A scripted
"go to the point" agent takes one point every ~4 s (~7 in 30 s), which is the racing ceiling.
Frames: `out/ep5` (run F, with the point drawn as a decal).

| run | loadout | steps | survival | captures / team | shots / hits | dmg dealt | kills | behaviour |
|---|---|---|---|---|---|---|---|---|
| F `race_v1` | fists | 600k | 9.1 s | 1.0 / 2.0 | 0.8 / 0.04 | 0.6 | 0 | sprints point to point at the scripted pace, no gun, dies to the chasers at ~11 s |
| F+ (resume) | fists | 2.0M | 10.8 s | 1.4 / 2.8 | 1.5 / 0.07 | 0.7 | 0 | same, slightly longer evasion; still no gun |
| G `race_v1` | armed | 600k | 10.2 s | 0.03 / 0.06 | 32 / 3.7 | 46 | 0.09 | fights the chasers (damage 8 -> 46, still rising), ignores the points |

What it shows. As a reward *structure* the race does what v1/v2 could not: no bankable term, no
suicide optimum, survival goes up instead of down, and the objective is learned in ~300k steps.
What it does not do by itself is bridge farm -> fight: with fists the fastest route to reward is
to run (fists are 1 u/s faster than a gun and the random points never pass the kit, so
`damage_dealt` never fires); with guns the fastest route is to shoot the chasers that come to you
(the capture needs a 3-5 s run under fire). Each loadout finds its own local optimum against an
opponent that hunts instead of racing. The lever is therefore the opponent, not another reward
term: a scripted `racer` that also takes points makes the enemy's captures cost us the point and
makes killing it pay in captures within the same episode, which is the credit path the current
chaser cannot provide. That, plus `--auto-pickup` or the armed stage as a warm start, is the next
iteration; the reward stays at two terms.

## Interpretations of the spec made here (to align with the bridge)

* Move speed: 13 u/s with fists (12 + 1), 12 u/s with a gun equipped.
* Picking up a gun while holding fists auto-equips it (the assist layer covers the other case).
* Firing on an empty clip triggers an automatic reload (mock); the assist layer sends `Reload` anyway.
* `Interact` prefers the nearest loot within 2 u, else a downed teammate within 5 u
  (`Revive` is the explicit key); revive takes 8 s and restores 24 HP.
* Downed players bleed 2 HP/s (x1.25 per additional down) as `damage` events with
  `source: "bleed"`; gas would be `source: "gas"`.
* The mock stops ticking at the terminal tick, so `t` on a `done` message can be earlier
  than `step start + ticks x 0.01`; a `step` after `done` returns an `error`.
* `bullets[].player_id` is a numeric engine id that Python cannot map to an agent id; the
  featurizer uses bullet geometry only. Adding `agent` to bullet entries would help.
* Metrics: `alive_at_end` = not dead (downed counts alive); `kills` includes the downed
  teammates killed by a team wipe (credited to the killer); `hp_mean` averages over all
  ticks while not dead.
* Loot positions: ring of radius 1.5 u around (110, 132), (154, 132), (132, 132) with a
  seeded rotation; ammo piles 1 u either side of each gun (30 rounds each).
