"""Compatibility package for older CPC training import paths.

The active env/action modules now live in :mod:`experiment.core`. This package
keeps `training.cpc_env`, `training.cpc_actions`, and `training.cpc_metrics`
available for older notebooks or legacy scripts without importing TorchRL.
"""
