# survev CPC bridge protocol v0

The bridge lets Python drive agents inside the survev game server (Node/TS). It is a
WebSocket server started by the game repo (`pnpm cpc:bridge`, default `ws://127.0.0.1:8765`)
that hosts many independent offline games ("envs") in one process. Messages are JSON, one
request → one response, in order, per connection.

Design rules

- The server owns the simulation. Python never touches game state directly; controlled agents
  receive `CpcAction`s that go through the same `InputMsg → player.handleInput()` path as a
  human client (information-set parity, same physics).
- Stepping is decision-level by default: one `step` advances `ticks` game ticks (1 tick =
  1/100 s; `netSync` every 3 ticks). The last action of every controlled agent is held between
  ticks exactly like a held key. `ticks=1` (100 Hz), `ticks=3` (human input cadence, 33 Hz) and
  `ticks=10` (0.1 s, default policy cadence) are the supported presets; any positive integer
  works.
- Agents not listed in `controlled` are driven by a built-in server-side scripted policy
  (`chaser` — loot nearest gun/ammo → approach → strafe → hold fire; revives a downed teammate
  when no enemy is within 25 u). It is the opponent the PPO agent trains against.
- Observations are per agent and contain only what that agent's client would receive from the
  server (objects inside the agent's zoom-culled view rectangle, team status, gas, own state).
  Enemy HP is never included. Featurization to fixed-size vectors is the Python side's job.

## Scenario `duo2v2_field`

Map `test_normal` (264 × 264 world units, open grass, no obstacles, water outside the 48-unit
shore), scenario region 128 × 128 centered at (132, 132). Four agents: `team-a-0`, `team-a-1`
(spawn x = 100, y = 125.6 / 138.4) and `team-b-0`, `team-b-1` (spawn x = 164, same ys). World
y grows upward on screen. Everyone spawns with fists. Loot is dropped from a seeded layout:
an identical starter kit 10 u toward the center from each duo spawn (ak47, mp5, bandage×4,
soda×2, helmet01, chest01, 2xscope, plus the side ammo the engine drops next to guns: two
762mm piles by the ak47, two 9mm piles by the mp5) and a contested center kit (healthkit,
painkiller, 4xscope). Player move speed is 12 u/s (+1 with fists equipped).

Episode end: one team left alive (`reason: "elimination"`) or `t >= timeLimit`
(`reason: "time_limit"`, default 60 s). A downed player counts as alive for the team until the
whole team is downed/dead (engine semantics: last teammate death kills the downed ones).

## Messages

### reset

```json
{"type": "reset", "env_id": 0, "scenario": "duo2v2_field", "seed": "cpc-duo2v2-seed-0",
 "options": {"mapSize": 128, "timeLimit": 60,
             "controlled": ["team-a-0", "team-a-1"], "scripted": "chaser"}}
```

- `env_id`: integer chosen by the client; reusing an id replaces that env.
- `seed`: string or number; fixes the loot layout (and the map RNG, which does not matter on
  the empty field).
- `layout` (options): `"fixed"` (default; duos west/east of the center, 32 u out) or `"random"`
  (the spawn axis is rotated by a seeded angle, the distance to the center is drawn from [24, 44] u,
  duos stay mirror images through the center and face it, the team kits move with them). Use
  `"random"` for training so absolute directions carry no information across episodes.
- `loadout` (options): `"fists"` (default) or `"armed"` (everyone spawns with a loaded ak47 and
  90 rounds; curriculum helper).
- `objective` (options): `{"mode": "race", "radius": 4, "minDist": 30, "maxDist": 70, "margin": 12}`
  turns on the shared capture point (default `{"mode": "none"}`). One point exists at a time and every
  agent sees it (`obs.objective`); the first standing player within `radius` captures it for their team
  (a `capture` event) and the point moves to a seeded position `minDist..maxDist` u away, inside the
  region minus `margin`. Points keep coming until the time limit.
- `endOnElimination` (options): `true` (default) ends the episode when one team is left; `false` runs to
  the time limit even after a wipe (survivors keep taking points) and ends early only when every
  controlled agent is dead (`reason: "controlled_dead"`). In race mode `winner_team` is the team with
  more captures (null on a tie).
- `controlled`: agent ids Python will send actions for. Others use `scripted` (`"chaser"`,
  `"racer"` or `"idle"`).
- `scriptedOptions` (options): strength of the scripted opponents, default exact —
  `{"aimNoiseDeg": 10, "reactionDelay": 0.5, "engageDist": 25}`. `aimNoiseDeg` is the std-dev of a
  seeded Gaussian error added to their combat aim on every decision, `reactionDelay` the seconds an enemy
  must stay inside their 30 u fire range before they open fire, `engageDist` the racer's break-off
  distance. Against two idle targets the chasers' time to kill grows 6 s (exact) -> 9 s (5°) -> 14 s
  (10°) -> 26 s (20°); at 60° they do not finish in 30 s.
- Response: an `obs` message (below) with `t = 0`, `done = false`.

### step

```json
{"type": "step", "env_id": 0, "ticks": 10,
 "actions": {"team-a-0": {"move": {"x": 1, "y": 0}, "aim": {"x": 0.3, "y": 0.95},
                          "fire": {"start": false, "hold": true},
                          "inputs": ["Interact"], "useItem": ""},
             "team-a-1": {}}}
```

- Missing controlled agents keep their previous action (held input). `{}` means "release
  everything" (no movement, no fire).
- `move`: world-space direction, length ignored, quantized server-side to the 8 keyboard
  directions (`keys` mode). Zero / omitted = stand still.
- `aim`: world-space direction (normalized server-side).
- `fire.start`: edge-triggered single shot (send on one step only); `fire.hold`: level-triggered
  auto fire.
- `inputs`: list of `Input` names or numbers. Useful ones: `Interact` (7, pick up loot / open /
  revive), `Reload` (5), `Revive` (8), `EquipPrimary` (11), `EquipSecondary` (12), `EquipMelee`
  (13), `EquipNextWeap` (17), `UseBandage` (23), `UseHealthKit` (24), `UseSoda` (25),
  `UsePainkiller` (26). Applied once when the step starts (they are edge events, like key
  presses).
- `useItem`: item id to consume, e.g. `"bandage"`, `"healthkit"`, `"soda"`, `"painkiller"`,
  or a scope id to switch scopes (`"2xscope"`).
- Response: an `obs` message.

### step (batched)

```json
{"type": "step", "envs": {"0": {"ticks": 10, "actions": {...}}, "1": {"ticks": 10, "actions": {...}}}}
```

Response: `{"type": "obs_batch", "envs": {"0": <obs message>, "1": <obs message>}}`. Envs are
stepped sequentially in one process; use this for vectorized training to avoid one round trip
per env.

### close

`{"type": "close", "env_id": 0}` → `{"type": "closed", "env_id": 0}`. Closing a connection closes
all its envs.

### error

`{"type": "error", "env_id": 0, "message": "..."}` for malformed messages or unknown envs.

## obs message

```json
{"type": "obs", "env_id": 0, "t": 3.5, "tick": 350, "done": false,
 "agent_ids": ["team-a-0", "team-a-1", "team-b-0", "team-b-1"],
 "teams": {"team-a-0": "team-a", "team-a-1": "team-a", "team-b-0": "team-b", "team-b-1": "team-b"},
 "obs": {"team-a-0": <AgentObservation>, ...},
 "events": [<Event>, ...],
 "info": {"alive_teams": 2, "winner_team": null, "reason": null, "metrics": null}}
```

- `obs` is returned for every agent (controlled and scripted) so logs are complete.
- `events` are the fire / damage / down / kill events that happened during this step (see
  below); `t` is game time in seconds since reset.
- When `done` is true: `info.winner_team` is `"team-a"`, `"team-b"` or `null` (time limit with
  both alive; in race mode the team with more captures, null on a tie), `info.reason` is
  `"elimination"`, `"time_limit"` or `"controlled_dead"`, and `info.metrics` holds per agent metrics
  (below). `info.objective` (race mode) carries the current point and the captures per team on every
  message. After `done`, further `step`s return an error until `reset`.

### AgentObservation

All positions are world coordinates. `dist` is the distance from the observing agent.

```json
{"self": {"id": "team-a-0", "team": "team-a", "pos": {"x": 106.7, "y": 131.1},
          "dir": {"x": 0.93, "y": -0.37}, "hp": 87.5, "boost": 0, "downed": false, "dead": false,
          "weapon": "ak47", "clip": 21, "reserve": 45,
          "weapons": [{"slot": 0, "type": "ak47", "ammo": 21}, {"slot": 1, "type": "", "ammo": 0},
                      {"slot": 2, "type": "fists", "ammo": 0}, {"slot": 3, "type": "", "ammo": 0}],
          "inventory": {"bandage": 4, "healthkit": 0, "soda": 2, "painkiller": 0,
                        "762mm": 45, "9mm": 0, "12gauge": 0, "556mm": 0},
          "scope": "1xscope", "zoom": 28, "action": 0, "cur_weap_idx": 0},
 "teammates": [{"id": "team-a-1", "pos": {"x": 108.2, "y": 126.3}, "dist": 5.0, "hp": 100,
                "downed": false, "dead": false}],
 "players": [{"id": "team-b-0", "team": "team-b", "pos": {"x": 128.9, "y": 130.2}, "dist": 22.2,
              "dir": {"x": -1, "y": 0}, "downed": false, "dead": false, "weapon": "ak47"}],
 "loot": [{"id": 512, "type": "bandage", "pos": {"x": 110.4, "y": 128.8}, "dist": 4.4, "count": 4}],
 "obstacles": [{"id": 77, "type": "tree_01", "pos": {"x": 120.0, "y": 140.0}, "dist": 16.0,
                "collidable": true, "height": 1, "scale": 1}],
 "bullets": [{"pos": {"x": 118.0, "y": 131.0}, "dir": {"x": -1, "y": 0}, "player_id": 1027}],
 "dead_bodies": [{"pos": {"x": 130.0, "y": 129.0}, "dist": 23.3}],
 "gas": {"mode": 0, "rad": 196, "pos": {"x": 132, "y": 132}, "rad_new": 196,
         "pos_new": {"x": 132, "y": 132}},
 "shots_heard": [{"dir": "NE", "range": "mid"}],
 "alive_count": 4, "alive_teams": 2}
```

Rules

- `teammates` come from group status and are always present regardless of visibility (that is
  what the client shows on the team HUD). They carry HP; `players` (enemies) never do.
- `players`, `loot`, `obstacles`, `bullets`, `dead_bodies` contain only objects inside the
  agent's culling rectangle (`zoom + 4` half-width, 16:9), i.e. the same set the client renders.
  On the open field `obstacles` is empty; the field exists so cover features can be added later
  without a protocol change.
- `objective` is `null` unless the race objective is on; then `{"index": 3, "pos": {"x": 150.2, "y": 118.7},
  "radius": 4, "dist": 27.5}` — the same point for every agent (a shared HUD marker), `dist` from the
  observing agent.
- `hp` is 0–100. A downed player's HP is reset to 100 by the engine and then bleeds.
- `shots_heard` holds the shots *other* players fired during this step that this agent is close
  enough to hear, bucketed into one of 8 compass points and near / mid / far. The radius is 48 u,
  taken from the client's own audio: another player's shot plays on the `otherPlayers` channel whose
  `maxRange` is 48 (default `rangeMult` 1). It reaches past the view rectangle (zoom 28 -> 32 u
  half-width), which is exactly why it is a separate channel — an agent hears fights it cannot see.
  Agents further away than 48 u get an empty list, and you never hear your own shots. `dir` is
  **world** space, where y grows upward (`"N"` = +y), not screen space. It is a snapshot of the step,
  not a running log: a quiet step gives `[]`.
- `action` is the engine's `GameConfig.Action` enum (0 none, 1 reload, 2 useItem, 3 revive).
- Keys are an allowlist: a server-side test fails if any other key appears (M6).

### Event

```json
{"type": "fire",   "t": 3.4, "agent": "team-b-1", "weapon": "mp5", "pos": {"x": 139.0, "y": 125.8}, "dir": {"x": -1, "y": 0.1}}
{"type": "damage", "t": 3.52, "agent": "team-a-0", "source": "team-b-1", "weapon": "mp5",
                   "amount": 13, "hp_before": 100, "hp_after": 87, "downed": false, "dead": false, "pos": {"x": 121.7, "y": 130.6}}
{"type": "down",   "t": 5.68, "agent": "team-a-0", "source": "team-b-1"}
{"type": "kill",   "t": 6.03, "agent": "team-a-1", "source": "team-b-1"}
```

```json
{"type": "capture", "t": 4.3, "agent": "team-a-1", "team": "team-a", "index": 0, "pos": {"x": 150.2, "y": 118.7}, "time_to_capture": 4.3}
```

```json
{"type": "loot",   "t": 0.5,  "agent": "team-a-0", "item": "ak47", "count": 1, "pos": {"x": 107.0, "y": 129.1}}
{"type": "heal",   "t": 12.4, "agent": "team-a-0", "item": "bandage", "hp_before": 50, "hp_after": 65, "boost_before": 0, "boost_after": 0}
{"type": "revive", "t": 21.7, "agent": "team-a-1", "source": "team-a-0"}
```

`loot` is emitted only when the pickup actually changed the player's holdings (the engine destroys
the pile and re-drops whatever did not fit, so a refused pickup — inventory full, already owned,
better item equipped — produces no event); `count` is the pile's count, not necessarily the amount
taken. `heal` fires when the item is *consumed*, i.e. when the use action completes, and covers both
heals (`bandage`, `healthkit`: hp changes) and boosts (`soda`, `painkiller`: boost changes).
`revive` names the revived teammate as `agent` and the reviver as `source`, like `down` / `kill`.

`amount` is HP actually removed (after armor); a downing or killing hit removes everything that
was left. `capture` (race objective only) names the agent that touched the point first and its team;
credit is per team on the Python side. Heard shots are not events: they are per agent, so they live
in the observation (`obs.shots_heard`, below).

### metrics (in `info.metrics` when `done`)

Per agent:

```json
{"survival_time": 5.5, "alive_at_end": false, "downed_time": 0.9,
 "hp_mean": 71.3, "hp_end": 0, "damage_dealt": 53, "damage_taken": 120,
 "kills": 0, "shots": 29, "hits_given": 4, "team_win": false,
 "partner_survival_time": 5.5, "partner_hp_end": 0,
 "captures": 0, "team_captures": 0}
```

`hp_mean` averages HP over the agent's alive ticks (downed ticks included, at the engine's
downed HP). `team_win` is true for the winning team's agents only. These are the five headline
numbers (survival time, HP mean/end, damage, team win, partner survival) plus their inputs;
the harness cooperation metrics are computed offline from the JSONL log, not here.

## Throughput expectations

Simulation cost dominates at `ticks=10`; the target is ≥ 20× real time per process with 4
agents (S1 in the plan). At `ticks=1` the per-step JSON round trip dominates and 3–10× real
time is expected. One process can host many envs; run one process per CPU core for scale.
