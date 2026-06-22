# Experiment

## Reward-selected Checkpoint

PPO training writes three reward-selection artifacts into each run directory:

- `checkpoint_latest.pt`: the most recent training update.
- `checkpoint_min_reward.pt`: the currently selected reward checkpoint.
- `selected_reward_checkpoint.json`: metadata describing the selected update, global step, metric, mode, and value.

Selection is configurable with:

- `selection_metric`: `eval_mean_episode_reward` or `episodic_return_mean`
- `selection_mode`: `min` or `max`

The default is `selection_metric: eval_mean_episode_reward` and `selection_mode: min`. This is useful for inspecting low-reward or failure behavior in GUI/debug tooling. It is not called the best policy because "best" depends on the chosen metric and mode.

Train:

```powershell
python experiment/train_ppo.py --config experiment/configs/ppo_smoke.yaml --smoke
```

Evaluate the selected checkpoint:

```powershell
python experiment/eval_ppo.py --checkpoint experiment/runs/<run>/checkpoint_min_reward.pt --episodes 10 --device cpu --deterministic
```

Try the two-agent model runner:

```powershell
python experiment/run_model_agents.py --checkpoint-a experiment/runs/<run>/checkpoint_min_reward.pt --checkpoint-b experiment/runs/<run>/checkpoint_min_reward.pt --episodes 1 --device cpu --deterministic --export experiment/runs/<run>/two_agent_min_reward_eval.json
```

The current toy `training.cpc_env.CPCEnv` exposes one controllable self agent. The runner therefore fails clearly for two-agent control until a multi-agent env with action mappings is available. Model loading and action production remain outside the environment.
