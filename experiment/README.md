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

Try the two-agent model runner:

```powershell
python experiment/run_model_agents.py --checkpoint-a experiment/runs/<run>/checkpoint_selected.pt --checkpoint-b experiment/runs/<run>/checkpoint_selected.pt --episodes 1 --device cpu --deterministic --export experiment/runs/<run>/two_agent_selected_eval.json
```

The current toy `training.cpc_env.CPCEnv` exposes one controllable self agent. The runner therefore fails clearly for two-agent control until a multi-agent env with action mappings is available. Model loading and action production remain outside the environment.

## Combat-Forcing Toy Reward

The toy CPC training environment intentionally reduces the per-step survival reward to avoid no-op survival reward hacking. Center pressure now shrinks a safe zone over time, and the scripted enemy moves or fires weakly to force engagement instead of allowing the learner to idle.

Reward components are logged separately in `info["reward_components"]` and in `metrics.csv` with `reward_` prefixes, including damage, death/win, survival, enemy approach, aim alignment, attack intent, and zone pressure. These rewards are only PPO training signals; they are not the final CPC evaluation metric.
