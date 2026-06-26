"""Compatibility shim for the old training.cpc_metrics import path."""

try:
    from experiment.core.cpc_metrics import *  # noqa: F401,F403
except ModuleNotFoundError:
    from core.cpc_metrics import *  # noqa: F401,F403
