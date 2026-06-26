# Training Compatibility

The active CPC tactical-baseline workflow does not use this package for
training adapters.

Only lightweight compatibility shims live here:

- `cpc_actions.py` re-exports `experiment.core.cpc_actions`
- `cpc_env.py` re-exports `experiment.core.cpc_env`
- `cpc_metrics.py` re-exports `experiment.core.cpc_metrics`

Older PPO, TorchRL, checkpoint, and model-gameplay experiments were moved to
`legacy/python_experiments/`.
