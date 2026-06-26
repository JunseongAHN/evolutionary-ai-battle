"""Integrated tactical baseline for local CPC combat debugging."""

from .fire_rule import FireRule
from .tactical_baseline_bot import TacticalBaselineBot, build_tactical_baseline_bot

__all__ = ["FireRule", "TacticalBaselineBot", "build_tactical_baseline_bot"]
