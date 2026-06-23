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

Use `--save-result` to write a step-by-step JSON debug artifact for model-backed gameplay. The export records the model raw action, decoded engine action, environment debug state, observation after each step, reward components, metrics, and done/truncated status.

Single-agent export:

```powershell
python experiment/run_model_gameplay.py --checkpoint-a experiment/runs/<run>/checkpoint_min_reward.pt --episodes 1 --max-steps 100 --device cpu --deterministic --save-result experiment/runs/<run>/model_gameplay_result.json
```

Two-agent export is intentionally blocked until `CPCEnv` exposes two controlled agents:

```powershell
python experiment/run_model_gameplay.py --checkpoint-a experiment/runs/<run>/checkpoint_min_reward.pt --checkpoint-b experiment/runs/<run>/checkpoint_min_reward.pt --episodes 1 --max-steps 100 --device cpu --deterministic --save-result experiment/runs/<run>/two_agent_gameplay_result.json
```

That command fails clearly with a message explaining that a multi-agent env or runner-level controlled agent mapping is needed first. The env remains model-agnostic; checkpoint loading, model action selection, and serialization live in the runner layer.

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

## Aim learning and safe-zone reward

Fire selection is no longer treated as enough combat intent by itself. The PPO reward now separates `fire_requested`, `shot_fired`, projectile `bullet_hit`, and expired missed shots. The env gives stronger reward for aiming toward the enemy, aligned shots, and projectile hits, while off-target shots and bad aim near range are penalized.

The safe circle also has stronger pressure. Leaving the circle applies `zone_pressure`, moving back toward center gives `return_to_zone`, moving deeper outside is penalized, and moving outward near the edge gets a soft `near_edge_outward` penalty before the agent fully exits.

Saved gameplay JSON includes `aim_debug` and `zone_debug` per step. The pygame viewer draws the policy aim ray, target direction ray, bullet direction, bullets, and outside-safe-zone status. Training logs include reward components plus `mean_aim_alignment`, `off_target_shot_count`, `bullet_hit_count`, `outside_safe_zone_rate`, `near_edge_outward_count`, and `total_reward`.

To prevent an always-aim-right shortcut, training can randomize enemy spawn direction in YAML:

```yaml
randomize_enemy_spawn_direction: true
```

## Combat-Forcing Toy Reward

The toy CPC training environment intentionally reduces the per-step survival reward to avoid no-op survival reward hacking. Center pressure now shrinks a safe zone over time, and the scripted enemy moves or fires weakly to force engagement instead of allowing the learner to idle.

Reward components are logged separately in `info["reward_components"]` and in `metrics.csv` with `reward_` prefixes, including damage, projectile hits, death/win, survival, enemy approach, aim alignment, shot alignment, missed shots, and zone pressure. These rewards are only PPO training signals; they are not the final CPC evaluation metric.
