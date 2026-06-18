# CPC TorchRL Experiment Scaffold

This folder contains the first small Python scaffold for a TorchRL-oriented CPC experiment.
It is intentionally not a full PPO pipeline yet.

## Files

- `training.ipynb` is the initiating notebook and demo entry point.
- `cpc_actions.py` defines the first multi-discrete action channels.
- `cpc_env.py` contains a deterministic toy CPC environment with `reset()` and `step()`.
- `cpc_metrics.py` accumulates basic teammate usefulness metrics.

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

## Design Constraint

The common schema stays framework-independent. TorchRL-specific code should live in this experiment/training layer, not in `schema.py` or the shared engine schema.

## TODO

- Add a TorchRL `EnvBase` adapter around this toy environment.
- Add multi-head PPO actor outputs for `move`, `aim`, and `fire`.
- Add collector, GAE, PPO loss, and checkpointing.
- Bridge exported toy trajectories into the full replay viewer format.
