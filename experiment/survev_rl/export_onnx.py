"""Export the PPO actor to ONNX (dynamic batch) for ``onnxruntime-node`` in the TS server.

The exported graph maps ``obs[batch, obs_dim] (float32)`` to ``logits[batch, sum(nvec)]``
(and optionally ``value[batch, 1]``). A sidecar ``<name>.json`` records everything the TS
side needs to decode an action without Python: ``nvec``, per-dimension names, the harness
move vectors / aim bins, ``vector_keys`` and the featurizer / action-space configs.

Example::

    python -m experiment.survev_rl.export_onnx --checkpoint runs/ppo_v0/checkpoint_final.pt \
        --out runs/ppo_v0/actor.onnx
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import warnings
from pathlib import Path
from typing import Any

import torch
from torch import nn

from experiment.core.cpc_actions import AIM_BINS, MOVE_LABELS, MOVE_VECTORS

from .actions import PRIMITIVE_DIMS, SKILLS
from .ppo import ActorCritic, load_checkpoint


class ActorWrapper(nn.Module):
    """obs -> concatenated logits (+ value) for export."""

    def __init__(self, agent: ActorCritic, with_value: bool = False) -> None:
        super().__init__()
        self.agent = agent
        self.with_value = with_value

    def forward(self, obs: torch.Tensor):  # type: ignore[override]
        logits, value = self.agent(obs)
        if self.with_value:
            return logits, value
        return logits


def export_actor(
    agent: ActorCritic,
    out_path: str | Path,
    opset: int = 17,
    with_value: bool = False,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write ``out_path`` (.onnx) and ``out_path.with_suffix('.json')``; returns the .onnx path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = ActorWrapper(agent, with_value).eval().cpu()
    dummy = torch.zeros(1, agent.obs_dim, dtype=torch.float32)
    output_names = ["logits", "value"] if with_value else ["logits"]
    dynamic_axes = {"obs": {0: "batch"}, **{n: {0: "batch"} for n in output_names}}
    params = inspect.signature(torch.onnx.export).parameters
    legacy_kwargs: dict[str, Any] = dict(
        input_names=["obs"], output_names=output_names, dynamic_axes=dynamic_axes, opset_version=opset
    )
    if "dynamo" in params:
        legacy_kwargs["dynamo"] = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            torch.onnx.export(wrapper, (dummy,), str(out_path), **legacy_kwargs)
        except Exception as legacy_exc:  # fall back to the dynamo exporter (needs onnxscript)
            if "dynamo" not in params:
                raise
            try:
                batch = torch.export.Dim("batch")
                torch.onnx.export(
                    wrapper,
                    (dummy,),
                    str(out_path),
                    input_names=["obs"],
                    output_names=output_names,
                    dynamic_shapes={"obs": {0: batch}},
                    opset_version=opset,
                    dynamo=True,
                )
            except Exception as dynamo_exc:  # pragma: no cover
                raise RuntimeError(
                    f"ONNX export failed (legacy: {legacy_exc!r}; dynamo: {dynamo_exc!r})"
                ) from dynamo_exc
    sidecar = build_sidecar(agent, meta or {}, with_value, opset)
    out_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return out_path


def build_sidecar(agent: ActorCritic, meta: dict[str, Any], with_value: bool, opset: int) -> dict[str, Any]:
    mode = (meta.get("action_space") or {}).get("mode", "primitive")
    dims = list(PRIMITIVE_DIMS) if mode == "primitive" else ["skill"]
    heads = []
    offset = 0
    for name, n in zip(dims, agent.nvec):
        heads.append({"name": name, "size": int(n), "offset": offset})
        offset += int(n)
    return {
        "format": "survev_rl-actor-v0",
        "opset": opset,
        "inputs": {"obs": ["batch", agent.obs_dim]},
        "outputs": {"logits": ["batch", int(sum(agent.nvec))], **({"value": ["batch", 1]} if with_value else {})},
        "obs_dim": agent.obs_dim,
        "nvec": list(agent.nvec),
        "heads": heads,
        "action_mode": mode,
        "move_vectors": {str(k): {"x": v[0], "y": v[1], "label": MOVE_LABELS[k]} for k, v in MOVE_VECTORS.items()},
        "aim_bins": {
            str(k): {"x": math.cos(2 * math.pi * k / AIM_BINS), "y": math.sin(2 * math.pi * k / AIM_BINS)}
            for k in range(AIM_BINS)
        },
        "skills": list(SKILLS),
        "vector_keys": meta.get("vector_keys"),
        "featurizer": meta.get("featurizer"),
        "action_space": meta.get("action_space"),
        "env": meta.get("env"),
        "note": "sample each head independently (categorical over logits[offset:offset+size]) "
        "or take argmax; map move via move_vectors, aim via aim_bins; fire/interact are booleans.",
    }


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description="Export the PPO actor to ONNX.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True, help="output .onnx path")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--with-value", action="store_true", help="also export the value head")
    args = p.parse_args(argv)
    agent, ckpt = load_checkpoint(args.checkpoint, "cpu")
    path = export_actor(agent, args.out, opset=args.opset, with_value=args.with_value, meta=ckpt.get("meta") or {})
    print(f"wrote {path} and {path.with_suffix('.json')}")
    return path


if __name__ == "__main__":
    main()
