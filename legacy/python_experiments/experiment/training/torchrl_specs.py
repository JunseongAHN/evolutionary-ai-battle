from __future__ import annotations

from typing import Any


def import_torchrl_specs() -> dict[str, Any]:
    """Return TorchRL spec classes with fallbacks for older package versions."""
    import torchrl.data as torchrl_data

    composite = getattr(torchrl_data, "Composite", None) or getattr(torchrl_data, "CompositeSpec")
    categorical = getattr(torchrl_data, "Categorical", None) or getattr(torchrl_data, "DiscreteTensorSpec")
    unbounded = getattr(torchrl_data, "Unbounded", None) or getattr(torchrl_data, "UnboundedContinuousTensorSpec")

    return {
        "Composite": composite,
        "Categorical": categorical,
        "Unbounded": unbounded,
    }


def categorical_spec(n: int, *, device):
    import torch

    specs = import_torchrl_specs()
    categorical = specs["Categorical"]
    try:
        return categorical(n=n, shape=(), dtype=torch.int64, device=device)
    except TypeError:
        try:
            return categorical(n=n, shape=(), device=device)
        except TypeError:
            return categorical(n, shape=(), device=device)


def unbounded_spec(*, shape: tuple[int, ...], dtype, device):
    specs = import_torchrl_specs()
    unbounded = specs["Unbounded"]
    try:
        return unbounded(shape=shape, dtype=dtype, device=device)
    except TypeError:
        return unbounded(shape=shape, device=device)


def composite_spec(*, device, **fields):
    specs = import_torchrl_specs()
    composite = specs["Composite"]
    try:
        return composite(**fields, shape=(), device=device)
    except TypeError:
        return composite(**fields, shape=())


def import_check_env_specs():
    try:
        from torchrl.envs.utils import check_env_specs

        return check_env_specs
    except Exception:
        return None
