from __future__ import annotations

import pathlib
import sys

EXPERIMENT_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parent
for path in (EXPERIMENT_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.cpc_env import CPCEnv
from core.env_config import load_env_config


STAY = {"move": 0, "aim_dx": 1.0, "aim_dy": 0.0, "fire": 0}


def test_enemy_death_does_not_end_episode():
    env = CPCEnv.from_config(load_env_config("configs/env/autoplay_goal_loop.yaml"))
    env.reset(seed=17)
    env.enemy_fire = False
    env.state["enemy_hp"] = 0.0

    _, _, done, _ = env.step(STAY)

    assert done is False
    assert env.state["self_hp"] > 0.0


def test_player_death_ends_episode_even_when_ally_is_alive():
    env = CPCEnv.from_config(load_env_config("configs/env/autoplay_goal_loop.yaml"))
    env.reset(seed=17)
    env.enemy_fire = False
    env.state["self_hp"] = 0.0
    env.state["ally_hp"] = env.ally_max_hp

    _, _, done, _ = env.step(STAY)

    assert done is True
