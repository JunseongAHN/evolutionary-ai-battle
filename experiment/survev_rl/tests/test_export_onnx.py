from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiment.survev_rl.export_onnx import export_actor  # noqa: E402
from experiment.survev_rl.ppo import ActorCritic  # noqa: E402


def test_export_writes_onnx_and_sidecar(tmp_path):
    torch.manual_seed(0)
    agent = ActorCritic(obs_dim=218, nvec=(9, 16, 2, 2), hidden_dim=32)
    meta = {"vector_keys": [f"k{i}" for i in range(218)], "action_space": {"mode": "primitive", "assist": True}}
    path = export_actor(agent, tmp_path / "actor.onnx", meta=meta)
    assert path.exists() and path.stat().st_size > 1000
    side = json.loads(path.with_suffix(".json").read_text())
    assert side["obs_dim"] == 218 and side["nvec"] == [9, 16, 2, 2]
    assert [h["name"] for h in side["heads"]] == ["move", "aim", "fire", "interact"]
    assert side["heads"][1] == {"name": "aim", "size": 16, "offset": 9}
    assert side["move_vectors"]["4"]["x"] == 1.0 and side["aim_bins"]["4"]["y"] == pytest.approx(1.0)
    assert side["inputs"] == {"obs": ["batch", 218]} and side["outputs"] == {"logits": ["batch", 29]}
    assert len(side["vector_keys"]) == 218


def test_onnxruntime_roundtrip_dynamic_batch(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    torch.manual_seed(1)
    agent = ActorCritic(obs_dim=40, nvec=(9, 16, 2, 2), hidden_dim=32)
    path = export_actor(agent, tmp_path / "actor.onnx", with_value=True)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == ["obs"]
    assert [o.name for o in sess.get_outputs()] == ["logits", "value"]
    for batch in (1, 5, 33):
        x = np.random.default_rng(batch).standard_normal((batch, 40)).astype(np.float32)
        logits, value = sess.run(None, {"obs": x})
        with torch.no_grad():
            ref_logits, ref_value = agent(torch.from_numpy(x))
        assert logits.shape == (batch, 29) and value.shape == (batch, 1)
        np.testing.assert_allclose(logits, ref_logits.numpy(), atol=1e-5)
        np.testing.assert_allclose(value, ref_value.numpy(), atol=1e-5)
    # the sampled action from ONNX logits is a valid MultiDiscrete action
    x = np.zeros((1, 40), dtype=np.float32)
    logits = sess.run(None, {"obs": x})[0][0]
    offsets = np.cumsum([0, 9, 16, 2, 2])
    action = [int(np.argmax(logits[offsets[i]:offsets[i + 1]])) for i in range(4)]
    assert all(0 <= a < n for a, n in zip(action, (9, 16, 2, 2)))
