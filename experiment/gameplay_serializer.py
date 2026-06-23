from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))

    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]

    torch_jsonable = _torch_to_jsonable(value)
    if torch_jsonable is not _UNHANDLED:
        return torch_jsonable

    numpy_jsonable = _numpy_to_jsonable(value)
    if numpy_jsonable is not _UNHANDLED:
        return numpy_jsonable

    return str(value)


def serialize_gameplay_result(result: dict[str, Any]) -> dict[str, Any]:
    return to_jsonable(result)


def save_gameplay_result(result: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(serialize_gameplay_result(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


class _Unhandled:
    pass


_UNHANDLED = _Unhandled()


def _torch_to_jsonable(value: Any) -> Any:
    try:
        import torch
    except Exception:
        return _UNHANDLED

    if not isinstance(value, torch.Tensor):
        return _UNHANDLED

    tensor = value.detach().to("cpu")
    if tensor.numel() == 1:
        return tensor.reshape(-1)[0].item()
    return tensor.tolist()


def _numpy_to_jsonable(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:
        return _UNHANDLED

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return _UNHANDLED
