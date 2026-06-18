# CPC TorchRL Experiment Scaffold

This folder contains the first small Python scaffold for a TorchRL-oriented CPC experiment.
It is intentionally not a full PPO pipeline yet.

## Files

- `training.ipynb` is the initiating notebook and demo entry point.
- `cpc_actions.py` defines the first multi-discrete action channels.
- `cpc_env.py` contains a deterministic toy CPC environment with `reset()` and `step()`.
- `cpc_metrics.py` accumulates basic teammate usefulness metrics.
- `torchrl_specs.py` keeps small TorchRL spec compatibility helpers.
- `torchrl_env.py` adapts the toy env to TorchRL `EnvBase` / `TensorDict`.

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

## Tests

From the repo root:

```powershell
pytest experiment/tests/test_torchrl_env.py
```

The TorchRL wrapper tests skip if `torch`, `torchrl`, or `tensordict` is not installed.

## Design Constraint

The common schema stays framework-independent. TorchRL-specific code should live in this experiment/training layer, not in `schema.py` or the shared engine schema.

## PR Progression

PR1 proved the toy CPC loop and action decoding.

PR2 proves the toy CPC loop can be represented as a TorchRL environment. The wrapper is thin: it maps the existing toy observation/action/reward data into TensorDicts and specs, while keeping the environment logic in `cpc_env.py`.

PR3 should add PPO smoke training with separate policy heads for `move`, `aim`, and `fire`.

PPO is intentionally not included in PR2 so the adapter and specs can be validated before adding collectors, losses, checkpointing, or policy architecture.

## TODO

- Add multi-head PPO actor outputs for `move`, `aim`, and `fire`.
- Add collector, GAE, PPO loss, and checkpointing.
- Bridge exported toy trajectories into the full replay viewer format.
