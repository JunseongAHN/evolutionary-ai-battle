# Experiment

## Reward-selected Checkpoint

PPO training writes three reward-selection artifacts into each run directory:

- `checkpoint_latest.pt`: the most recent training update.
- `checkpoint_max_reward.pt`: the max-reward checkpoint when `selection_mode: max`.
- `checkpoint_selected.pt`: the currently selected reward checkpoint for the active selection mode.
- `selected_reward_checkpoint.json`: metadata describing the selected update, global step, metric, mode, and value.

Selection is configurable with:

- `selection_metric`: `eval_mean_episode_reward` or `episodic_return_mean`
- `selection_mode`: `min` or `max`

The default is `selection_metric: eval_mean_episode_reward` and `selection_mode: max`, so the selected checkpoint tracks the highest eval reward. Use `selection_mode: min` only when intentionally inspecting low-reward or failure behavior.

Train:

```powershell
python experiment/train_ppo.py --config experiment/configs/ppo_smoke.yaml --progress
```

Fast one-pass example:

```powershell
python experiment/train_ppo.py --config experiment/configs/ppo_example_tiny.yaml --smoke
```

Longer training config:

```powershell
python experiment/train_ppo.py --config experiment/configs/ppo_smoke.yaml --progress
```

Evaluate the selected checkpoint:

```powershell
python experiment/eval_ppo.py --checkpoint experiment/runs/<run>/checkpoint_selected.pt --episodes 10 --device cpu --deterministic
```

Play a checkpoint in the debug battle env and print the final result:

```powershell
python experiment/debug/model_gameplay.py --run-dir experiment/runs/<run> --device cpu
```

In Colab, the default checkpoint path is:

```python
from debug.model_gameplay import run_model_gameplay

run_model_gameplay(gui=False)
run_model_gameplay(pt_file="/content/drive/MyDrive/repos/evolutionary-ai-battle/experiment/runs/ppo_smoke_20260622_105638/checkpoint_latest.pt")
```

That default opens `/content/drive/MyDrive/repos/evolutionary-ai-battle/experiment/runs/ppo_smoke_20260622_105638/checkpoint_latest.pt`.
Add `gui=True` to show the pygame battle view when pygame/display support is available.

Try the two-agent model runner:

```powershell
python experiment/run_model_agents.py --checkpoint-a experiment/runs/<run>/checkpoint_selected.pt --checkpoint-b experiment/runs/<run>/checkpoint_selected.pt --episodes 1 --device cpu --deterministic --export experiment/runs/<run>/two_agent_selected_eval.json
```

The current toy `training.cpc_env.CPCEnv` exposes one controllable self agent. The runner therefore fails clearly for two-agent control until a multi-agent env with action mappings is available. Model loading and action production remain outside the environment.

## Model gameplay result export

Use `--save-result` to write a debug artifact for model-backed gameplay. The default log mode is compact and avoids duplicating full observations, env state, bullets, metrics, actions, and policy logits every step.

Single-agent export:

```powershell
python experiment/run_model_gameplay.py --checkpoint-a experiment/runs/<run>/checkpoint_min_reward.pt --episodes 1 --max-steps 100 --device cpu --deterministic --save-result experiment/runs/<run>/model_gameplay_result.json
```

Two-agent export is intentionally blocked until `CPCEnv` exposes two controlled agents:

```powershell
python experiment/run_model_gameplay.py --checkpoint-a experiment/runs/<run>/checkpoint_min_reward.pt --checkpoint-b experiment/runs/<run>/checkpoint_min_reward.pt --episodes 1 --max-steps 100 --device cpu --deterministic --save-result experiment/runs/<run>/two_agent_gameplay_result.json
```

That command fails clearly with a message explaining that a multi-agent env or runner-level controlled agent mapping is needed first. The env remains model-agnostic; checkpoint loading, model action selection, and serialization live in the runner layer.

## Gameplay log verbosity

`--save-result-mode` controls how much detail is written:

- `compact`: default, readable per-step debugging log.
- `full`: verbose raw debug export, preserving the original nested fields.
- `summary`: episode-level result only, no step details.
- `jsonl`: one compact JSON object per line for long episodes.

Compact default:

```powershell
python experiment/run_model_gameplay.py --checkpoint-a "C:\Users\PC\Downloads\checkpoint_latest.pt" --episodes 1 --max-steps 100 --device cpu --deterministic --save-result "experiment/runs/manual_debug/model_gameplay_result.compact.json"
```

Full debug:

```powershell
python experiment/run_model_gameplay.py --checkpoint-a "C:\Users\PC\Downloads\checkpoint_latest.pt" --episodes 1 --max-steps 100 --device cpu --deterministic --save-result "experiment/runs/manual_debug/model_gameplay_result.full.json" --save-result-mode full --policy-debug-mode full --include-state-before --include-observations --include-full-state
```

JSONL:

```powershell
python experiment/run_model_gameplay.py --checkpoint-a "C:\Users\PC\Downloads\checkpoint_latest.pt" --episodes 1 --max-steps 100 --device cpu --deterministic --save-result "experiment/runs/manual_debug/model_gameplay_result.jsonl" --save-result-mode jsonl
```

Summary:

```powershell
python experiment/run_model_gameplay.py --checkpoint-a "C:\Users\PC\Downloads\checkpoint_latest.pt" --episodes 1 --max-steps 100 --device cpu --deterministic --save-result "experiment/runs/manual_debug/model_gameplay_summary.json" --save-result-mode summary
```

Policy logits are omitted by default. Use `--policy-debug-mode topk` for the top 3 logits per action head, or `--policy-debug-mode full` for the raw logits. Inspect a saved log with:

```powershell
python experiment/analyze_gameplay_log.py --result "experiment/runs/manual_debug/model_gameplay_result.compact.json"
```

## Pygame model gameplay viewer

Use `--render-pygame` to open a simple pygame viewer while a loaded model plays. The viewer draws agent positions, HP, aim and movement direction, fire indicators, the shrinking safe zone, reward components, and basic metrics. `--save-result` still writes JSON while the viewer is open. If the pygame window is closed early, rollout stops cleanly and the partial JSON includes `"stopped_early": true`.

Install pygame for the viewer only:

```powershell
pip install pygame
```

Run gameplay with live rendering and JSON export:

```powershell
python experiment/run_model_gameplay.py --checkpoint-a "C:\Users\PC\Downloads\checkpoint_latest.pt" --episodes 1 --max-steps 100 --device cpu --deterministic --save-result "experiment/runs/manual_debug/model_gameplay_result.json" --render-pygame
```

Manual stepping opens the pygame viewer and waits at each generated step. Press `f` to move forward and `b` to review the previous generated step:

```powershell
python experiment/run_model_gameplay.py --checkpoint-a "C:\Users\PC\Downloads\checkpoint_latest.pt" --episodes 1 --max-steps 100 --device cpu --deterministic --save-result "experiment/runs/manual_debug/model_gameplay_result.json" --manual-step
```

Replay a saved result:

```powershell
python experiment/gui/replay_result.py --result "experiment/runs/manual_debug/model_gameplay_result.json" --fps 10
```

The viewer code lives under `experiment/gui/`. The env remains model-agnostic and pygame-agnostic; pygame is only used by the debug viewer and replay script.

## Projectile bullet simulation

The toy CPC env now uses projectile bullets instead of instant-hit fire. A fire action spawns a bullet at the shooter position, moves it each env step along the aim direction, expires it after `fire_range`, and applies damage only when the bullet hits the enemy. The saved gameplay JSON includes active bullets and per-step `bullet_events` such as `bullet_spawned`, `bullet_moved`, `bullet_hit`, and `bullet_expired`.

The pygame viewer draws each alive bullet as a moving circle with a short trail:

```powershell
python experiment/run_model_gameplay.py --checkpoint-a "C:\Users\PC\Downloads\checkpoint_latest.pt" --episodes 1 --max-steps 100 --device cpu --deterministic --save-result "experiment/runs/manual_debug/model_gameplay_result.json" --render-pygame
```

## Fire cooldown and shot timing

The policy can output `fire=1` every step, but the env treats that as a trigger request. A bullet is spawned only when weapon cooldown permits it. The default `fire_interval_steps` is `5`; during cooldown, `fire_requested` remains true if the policy holds fire, but `shot_fired` is false and no new bullet appears. Damage still comes only from `bullet_hit` events.

Saved gameplay results include:

- `fire.fire_requested`
- `fire.shot_fired`
- `fire.fire_blocked_reason`
- `fire.cooldown_remaining_steps_before`
- `fire.cooldown_remaining_steps_after`

Run with the pygame viewer and JSON export:

```powershell
python experiment/run_model_gameplay.py --checkpoint-a "C:\Users\PC\Downloads\checkpoint_latest.pt" --episodes 1 --max-steps 100 --device cpu --deterministic --save-result "experiment/runs/manual_debug/model_gameplay_result.json" --render-pygame
```

## Stage 1 Local Combat Micro

Stage 1 uses a fixed local combat stadium. Safe-zone shrink and zone rewards are disabled with `stage: local_combat`, `shrink_safe_zone: false`, and `use_zone_reward: false` so reward attribution stays focused on aim, fire, projectile hits, dodging, range control, damage trade, kill/death, and timeout HP lead.

The main reward signal is `damage_dealt_ratio - damage_taken_ratio`. Projectile hit and miss rewards are small, aim-bin shaping is capped at `0.04`, and range shaping is intentionally minor.

## Stage 1 reward gating against no-combat reward hacking

The policy previously obtained positive reward while firing zero shots, dealing zero damage, losing all HP, keeping `aim_bin=0`, and collecting aim/range shaping reward.

Stage 1 now gates shaping so no-combat behavior cannot look good:

- aim reward is only given when an actual shot is fired
- positive range reward requires combat engagement from a shot, damage dealt, or damage taken
- no-shot terminal episodes receive penalties, with extra penalties for dying without shooting or dying without damage dealt
- checkpoint selection uses `stage1_combat_quality`, not raw eval reward alone

A policy is only improving if `damage_trade_ratio` increases, `shot_fired_count` is nonzero, `bullet_hit_per_shot` increases, `damage_dealt_ratio` increases, `damage_taken_ratio` decreases, and warning count decreases.

Train Stage 1:

```powershell
python experiment/train_ppo.py --config experiment/configs/local_combat_micro.yaml
```

Evaluate/debug Stage 1:

```powershell
python experiment/run_model_gameplay.py --checkpoint-a "C:\Users\PC\Downloads\checkpoint_latest.pt" --episodes 1 --max-steps 100 --device cpu --deterministic --save-result "experiment/runs/manual_debug/local_combat_micro.combat_only.compact.json" --save-result-mode compact --render-pygame
```

Run tests:

```powershell
python -m pytest experiment -q
```

## Stage 1 Reward Hacking Checks

Stage 1 uses a fixed stadium and combat-only reward. The main evaluation metric is:

```text
damage_trade_ratio = damage_dealt_ratio - damage_taken_ratio
```

Warning signs:

- high reward but low damage dealt
- aim accuracy high but bullet hit rate low
- fire spam with few actual shots
- shaping reward dominates combat outcome
- high hit ratio from too few shots
- survival without combat

Analyze one compact gameplay result:

```powershell
python experiment/analyze_local_combat_eval.py --result experiment/runs/manual_debug/local_combat_micro.combat_only.compact.json --output-md experiment/runs/manual_debug/local_combat_micro.analysis.md
```

## Stage 1 eval analysis counters

Stage 1 eval analysis counters distinguish no fire request, cooldown-blocked fire requests, actual shots that miss, shots whose projectile lifecycle is not tracked, enemy bullets causing self damage, and aim-bin collapse.

The full eval analysis JSON includes fire counters, self/enemy bullet lifecycle counters, aim distributions, shot-time aim distributions, hit/miss-time aim distributions, range event-time rates, and warning counts. During PPO training, the compact `eval_analysis` progress log includes the high-level subset needed to tell whether the policy is failing before firing, after firing, or because aim has collapsed.

Example interpretations:

- If `damage_dealt_ratio = 0`, `damage_taken_ratio = 1`, and `shot_fired_count = 0`, the policy is not firing actual shots.
- If `shot_fired_count > 0`, `self_bullet_hit_count = 0`, and `self_bullet_missed_count = 0`, projectile lifecycle logging may be broken.
- If `aim_bin_0_rate = 1.0` and `exact_aim_match_rate` is low, the policy has collapsed to aim bin 0.

Evaluate baselines:

```powershell
python experiment/eval_local_combat_baselines.py --config experiment/configs/local_combat_micro.yaml --checkpoint-a "C:\Users\PC\Downloads\checkpoint_latest.pt" --episodes 20 --output-md experiment/runs/manual_debug/local_combat_baselines.md
```

## Aim-bin reward and aim-collapse debugging

Mean aim alignment can be misleading when enemies often appear to the right: a policy can collapse to `aim_bin=0`, hold fire, and still look partly aligned. The toy CPC env now computes `ideal_aim_bin`, circular `aim_bin_error`, exact/near/bad aim flags, and rewards discrete aim-bin accuracy instead of mostly rewarding dot-product alignment.

Exact aim is rewarded with `aim_bin_exact`, one-bin misses get a smaller `aim_bin_neighbor`, and clearly wrong aim receives `aim_bin_wrong`. Shot rewards only apply to actual `shot_fired` bullet spawns: exact shots receive `aligned_shot`, one-bin shots receive `near_aligned_shot`, and shots with two or more bins of error receive `off_target_shot`.

Projectile damage is still tied to real projectile hit events. `bullet_hit`, `damage_dealt`, and `missed_shot` come from bullets moving, hitting, or expiring, not from merely requesting fire. Generic `attack_intent` is small and only applies when fire is requested, the enemy is in range, aim is within one bin, and the weapon can fire.

The safe circle also has stronger pressure. Leaving the circle applies `zone_pressure`, moving back toward center gives `return_to_zone`, moving deeper outside is penalized, and moving outward near the edge gets a soft `near_edge_outward` penalty before the agent fully exits.

Saved gameplay JSON includes `aim_debug` and `zone_debug` per step. Compact gameplay logs include `aim.aim_bin_error`, exact/near flags, and the key aim-shot reward components. The pygame viewer draws the policy aim ray, target direction ray, bullet direction, bullets, and outside-safe-zone status, and shows the current aim bin, ideal bin, bin error, `EXACT AIM`, or `OFF TARGET`.

Training logs include reward components plus collapse indicators such as `aim_bin_0_rate`, `aim_bin_entropy`, `exact_aim_match_rate`, `within_1_bin_aim_rate`, `bad_aim_rate`, `shot_off_target_rate`, `bullet_hit_per_shot`, and `total_reward`.

To prevent an always-aim-right shortcut, training can randomize enemy spawn direction in YAML:

```yaml
randomize_enemy_spawn_direction: true
enemy_spawn_directions:
  - right
  - left
  - up
  - down
  - upper_right
  - lower_right
  - upper_left
  - lower_left
enemy_spawn_direction: null
```

For deterministic debug/eval, keep `randomize_enemy_spawn_direction: false` or set `enemy_spawn_direction` to a fixed value such as `left`.

Debug a trained checkpoint with a compact gameplay log:

```powershell
python experiment/run_model_gameplay.py --checkpoint-a "C:\Users\PC\Downloads\checkpoint_latest.pt" --episodes 1 --max-steps 100 --device cpu --deterministic --save-result "experiment/runs/manual_debug/model_gameplay_result.compact.json" --save-result-mode compact
```

## Combat-Forcing Toy Reward

The toy CPC training environment intentionally reduces the per-step survival reward to avoid no-op survival reward hacking. Center pressure now shrinks a safe zone over time, and the scripted enemy moves or fires weakly to force engagement instead of allowing the learner to idle.

Reward components are logged separately in `info["reward_components"]` and in `metrics.csv` with `reward_` prefixes, including damage, projectile hits, death/win, survival, enemy approach, aim alignment, shot alignment, missed shots, and zone pressure. These rewards are only PPO training signals; they are not the final CPC evaluation metric.
