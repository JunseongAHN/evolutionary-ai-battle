"""Compatibility shim for the old training.cpc_env import path."""

try:
    from experiment.core.cpc_env import *  # noqa: F401,F403
except ModuleNotFoundError:
    from core.cpc_env import *  # noqa: F401,F403
