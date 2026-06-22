# CPC TorchRL Experiment Scaffold

This folder is the `training` Python module for the small TorchRL-oriented CPC experiment scaffold.

## Files

- `../training.ipynb` is the initiating notebook and demo entry point.
- `cpc_actions.py` defines the first multi-discrete action channels.
- `cpc_env.py` contains a deterministic toy CPC environment with `reset()` and `step()`.
- `cpc_metrics.py` accumulates basic teammate usefulness metrics.
- `torchrl_specs.py` keeps small TorchRL spec compatibility helpers.
- `torchrl_env.py` adapts the toy env to TorchRL `EnvBase` / `TensorDict`.
- `ppo_policy.py` defines the small multi-head actor-critic.
- `train_ppo.py` runs minimal PPO smoke training.
- `eval_ppo.py` evaluates a saved smoke checkpoint.
- `../configs/ppo_smoke.yaml` contains tiny smoke defaults.

## Action Model

The first action space is modeled as simultaneous channels:

```python
action = {
    "move": Discrete(9),
    "aim": Discrete(16),
    "fire": Discrete(2),
}
```

This keeps movement, aim, and firing separately inspectable. It also allows useful combinations such as moving up-right while aiming at an enemy and firing in the same step.

`decode_action()` converts the policy action into the engine-compatible shape:

```python
{
    "moveX": float,
    "moveY": float,
    "aimX": float,
    "aimY": float,
    "fire": 0 | 1,
}
```

## Running

Open `experiment/training.ipynb` and run the cells from top to bottom.

The notebook demonstrates:

- environment reset
- random actions
- explicit move + fire in the same step
- decoded action output
- observations, rewards, done flags, and reward components
- basic CPC metrics
- a sample trajectory JSON-like object
- optional TorchRL spec mapping if TorchRL is installed
- PR2 TorchRL wrapper reset, action sampling, stepping, and optional spec checks
- PR3 policy forward pass, rollout, smoke training, checkpoint load, and eval

## Tests

From the repo root:

```powershell
pytest experiment/tests/test_torchrl_env.py
python -m pytest experiment/tests/test_ppo_smoke.py
```

The TorchRL/PPO tests skip if `torch`, `torchrl`, or `tensordict` is not installed.

## PPO Smoke Training

Run:

```powershell
python experiment/training/train_ppo.py --config experiment/configs/ppo_smoke.yaml --smoke
```

This writes a run directory under `experiment/runs/ppo_smoke_<timestamp>/` with:

- `checkpoint_latest.pt`
- `checkpoint_min_reward.pt`
- `selected_reward_checkpoint.json`
- `checkpoint.pt`
- `metrics.csv`
- `config.json`

Evaluate:

```powershell
python experiment/training/eval_ppo.py --checkpoint experiment/runs/<run>/checkpoint.pt
```

## Reward-selected Checkpoint

Training also writes a reward-selected checkpoint. By default the config uses:

- `selection_metric: eval_mean_episode_reward`
- `selection_mode: min`

The selected files are:

- `checkpoint_latest.pt`: most recent update
- `checkpoint_min_reward.pt`: selected low-reward checkpoint
- `selected_reward_checkpoint.json`: selected update, global step, metric, mode, and value

The min-reward checkpoint is useful for inspecting failure behavior in GUI/debug eval. It is not called the best policy because the selection depends on the configured metric and mode.

```powershell
python experiment/train_ppo.py --config experiment/configs/ppo_smoke.yaml --smoke
python experiment/eval_ppo.py --checkpoint experiment/runs/<run>/checkpoint_min_reward.pt --episodes 10 --device cpu --deterministic
python experiment/run_model_agents.py --checkpoint-a experiment/runs/<run>/checkpoint_min_reward.pt --checkpoint-b experiment/runs/<run>/checkpoint_min_reward.pt --episodes 1 --device cpu --deterministic --export experiment/runs/<run>/two_agent_min_reward_eval.json
```

The current toy `CPCEnv` supports one controllable self agent. Two-agent model use is blocked in the runner with a clear error until a multi-agent Python env is available. The environment remains model-agnostic.

## Combat-Forcing Toy Reward

The toy CPC env now uses a much smaller survival reward so standing still no longer looks artificially strong. It also spawns the enemy near fire range, applies shrinking center safe-zone pressure, and gives the scripted enemy weak pressure behavior.

Reward components are exposed separately through `info["reward_components"]` and logged into `metrics.csv` with `reward_` prefixes. These reward components are PPO training signals, not the final CPC metric report.

PR3 uses a manual PyTorch PPO loss while still training through `TorchRLCPCEnv`. This is intentional: the acceptance target is smoke validation of rollouts, multi-discrete log-probs, PPO shapes, metrics, and checkpointing. A future PR can replace the manual loop with TorchRL collectors/loss modules after the adapter surface is stable.

## PR3 Acceptance Check

Run:

```powershell
python experiment/check_pr3_acceptance.py --config experiment/configs/ppo_smoke.yaml --seed 123 --eval-episodes 10 --device cpu
```

Optional CUDA run:

```powershell
python experiment/check_pr3_acceptance.py --config experiment/configs/ppo_smoke.yaml --seed 123 --eval-episodes 10 --device cuda
```

The CPU run is the merge gate. CUDA can have small floating-point variation, so the acceptance script does not judge policy quality or require strong returns.

The check verifies:

- same-seed CPU reproducibility for first sampled action and rollout action sequence
- forced `move != 0` plus `fire = 1` action
- raw and decoded action visibility
- decoded action bounds
- checkpoint load into a fresh model
- `metrics.csv` columns and multiple rows
- 10-episode eval robustness

## Design Constraint

The common schema stays framework-independent. TorchRL-specific code should live in this experiment/training layer, not in `schema.py` or the shared engine schema.

## PR Progression

PR1 proved the toy CPC loop and action decoding.

PR2 proves the toy CPC loop can be represented as a TorchRL environment. The wrapper is thin: it maps the existing toy observation/action/reward data into TensorDicts and specs, while keeping the environment logic in `cpc_env.py`.

PR3 adds PPO smoke training with separate policy heads for `move`, `aim`, and `fire`.

The goal is not policy performance. The goal is proving the training loop can reset/step the TorchRL wrapper, collect rollouts, compute advantages, run PPO loss, and save artifacts.

PR4 should add scenario GT evaluation, trajectory export, and a richer CPC metric report.

## TODO

- Add scenario GT evaluation.
- Add trajectory export compatible with the replay viewer.
- Add richer CPC metric summaries.
