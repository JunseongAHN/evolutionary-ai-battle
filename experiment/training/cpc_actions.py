"""Compatibility shim for the old training.cpc_actions import path."""

try:
    from experiment.core.cpc_actions import *  # noqa: F401,F403
except ModuleNotFoundError:
    from core.cpc_actions import *  # noqa: F401,F403
