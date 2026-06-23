from __future__ import annotations

import json
import os
import pathlib
import sys
import builtins
from dataclasses import dataclass

import pytest

EXPERIMENT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from gameplay_serializer import save_gameplay_result, to_jsonable
from gui.geometry import world_to_screen
from gui.pygame_viewer import PygameCPCViewer
from run_model_agents import _run_single_agent_gameplay, build_arg_parser, run_model_gameplay
from training.cpc_env import CPCEnv
from training.cpc_actions import decode_action


class StaticDebugAgent:
    def act_with_debug(self, observation, deterministic: bool = True):
        del observation, deterministic
        raw_action = {"move": 4, "aim": 0, "fire": 1}
        decoded = decode_action(raw_action)
        return {
            "raw_action": raw_action,
            "decoded_action": {
                "move_x": decoded["moveX"],
                "move_y": decoded["moveY"],
                "aim_x": decoded["aimX"],
                "aim_y": decoded["aimY"],
                "fire": decoded["fire"],
            },
            "policy_debug": {
                "log_prob": None,
                "value": 0.5,
                "move_logits": [0.0],
                "aim_logits": [1.0],
                "fire_logits": [2.0],
            },
        }


@dataclass
class JsonableDataclass:
    value: int


def tiny_result() -> dict:
    return _run_single_agent_gameplay(
        agent=StaticDebugAgent(),
        checkpoint_a="checkpoint.pt",
        episodes=1,
        max_steps=2,
        device="cpu",
        deterministic=True,
    )


def test_world_to_screen_basic():
    assert world_to_screen({"x": 500.0, "y": 500.0}, {"width": 1000.0, "height": 1000.0}, (1000, 1000)) == (500, 500)


def test_debug_state_is_jsonable():
    env = CPCEnv(seed=0, max_steps=2)
    env.reset()

    json.dumps(to_jsonable(env.get_debug_state()))


def test_to_jsonable_tensor_numpy_nested():
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")

    converted = to_jsonable(
        {
            "tensor_scalar": torch.tensor([3.0]),
            "tensor_array": torch.tensor([[1, 2]]),
            "numpy_array": np.array([1.5, 2.5]),
            "numpy_scalar": np.float32(4.5),
            "dataclass": JsonableDataclass(value=7),
        }
    )

    assert converted == {
        "tensor_scalar": 3.0,
        "tensor_array": [[1, 2]],
        "numpy_array": [1.5, 2.5],
        "numpy_scalar": pytest.approx(4.5),
        "dataclass": {"value": 7},
    }


def test_save_result_file_created(tmp_path):
    path = tmp_path / "result.json"

    save_gameplay_result(tiny_result(), path)

    assert path.exists()


def test_run_model_gameplay_accepts_render_flag():
    parser = build_arg_parser()

    args = parser.parse_args(["--checkpoint-a", "checkpoint.pt", "--render-pygame", "--render-fps", "12"])

    assert args.render_pygame is True
    assert args.render_fps == 12


def test_save_result_still_works_without_gui(tmp_path):
    path = tmp_path / "result.json"

    save_gameplay_result(
        _run_single_agent_gameplay(
            agent=StaticDebugAgent(),
            checkpoint_a="checkpoint.pt",
            episodes=1,
            max_steps=1,
            device="cpu",
            deterministic=True,
            render_pygame=False,
        ),
        path,
    )

    assert json.loads(path.read_text(encoding="utf-8"))["episodes"][0]["episode_length"] == 1


def test_saved_result_contains_raw_and_decoded_action(tmp_path):
    path = tmp_path / "result.json"
    save_gameplay_result(tiny_result(), path)
    result = json.loads(path.read_text(encoding="utf-8"))
    agent_step = result["episodes"][0]["steps"][0]["agents"]["agent"]

    assert agent_step["raw_action"] == {"move": 4, "aim": 0, "fire": 1}
    assert set(agent_step["decoded_action"]) == {"move_x", "move_y", "aim_x", "aim_y", "fire"}


def test_saved_result_contains_env_state():
    result = tiny_result()
    env_step = result["episodes"][0]["steps"][0]["env"]

    assert "state" in env_step
    assert "state_before_step" in env_step
    assert "safe_zone" in env_step["state"]


def test_saved_result_contains_reward_components():
    result = tiny_result()
    info = result["episodes"][0]["steps"][0]["env"]["info"]

    assert "reward_components" in info
    assert "metrics" in info


def test_save_result_contains_bullets():
    result = tiny_result()
    first_env = result["episodes"][0]["steps"][0]["env"]

    assert "bullets" in first_env["state"]
    assert "bullet_events" in first_env["info"]
    assert first_env["info"]["bullet_events"][-1]["type"] == "bullet_spawned"


def test_save_result_contains_fire_debug():
    result = tiny_result()
    fire_info = result["episodes"][0]["steps"][0]["env"]["info"]["fire"]

    assert fire_info["fire_requested"] is True
    assert fire_info["shot_fired"] is True
    assert "cooldown_remaining_steps_before" in fire_info
    assert "cooldown_remaining_steps_after" in fire_info


def test_save_result_contains_aim_and_zone_debug():
    result = tiny_result()
    info = result["episodes"][0]["steps"][0]["env"]["info"]
    env_state = result["episodes"][0]["steps"][0]["env"]["state"]

    assert "aim_debug" in info
    assert "zone_debug" in info
    assert "aim_alignment" in info["aim_debug"]
    assert "move_toward_center" in info["zone_debug"]
    assert "aim_debug" in env_state
    assert "zone_debug" in env_state


def test_saved_result_is_json_serializable():
    json.dumps(tiny_result())


def test_env_does_not_import_pygame():
    source = (EXPERIMENT_ROOT / "training" / "cpc_env.py").read_text(encoding="utf-8").lower()

    assert "pygame" not in source


def test_renderer_import_fails_gracefully_if_pygame_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pygame":
            raise ImportError("missing pygame")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="pygame is required for --render-pygame"):
        PygameCPCViewer()


def test_pygame_viewer_can_render_bullets_with_dummy_driver(monkeypatch):
    pytest.importorskip("pygame")
    monkeypatch.setitem(os.environ, "SDL_VIDEODRIVER", "dummy")
    viewer = PygameCPCViewer(width=320, height=240, fps=60)
    try:
        assert viewer.render_step(
            {
                "map": {"width": 1000.0, "height": 1000.0, "center": {"x": 500.0, "y": 500.0}},
                "agents": {
                    "self": {"position": {"x": 100.0, "y": 100.0}, "hp": 100.0, "alive": True},
                    "enemy": {"position": {"x": 500.0, "y": 100.0}, "hp": 100.0, "alive": True},
                },
                "bullets": [
                    {
                        "bullet_id": "b0",
                        "owner_id": "self",
                        "spawn_pos": {"x": 100.0, "y": 100.0},
                        "previous_pos": {"x": 100.0, "y": 100.0},
                        "pos": {"x": 140.0, "y": 100.0},
                        "radius": 8.0,
                        "alive": True,
                    }
                ],
                "bullet_events": [],
            },
            None,
        )
    finally:
        viewer.close()


def test_two_agent_request_fails_clearly_if_env_does_not_support_it():
    with pytest.raises(NotImplementedError, match="Two-agent model gameplay requested"):
        run_model_gameplay(
            checkpoint_a="a.pt",
            checkpoint_b="b.pt",
            episodes=1,
            max_steps=1,
            device="cpu",
            deterministic=True,
        )
