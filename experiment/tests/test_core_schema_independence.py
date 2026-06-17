from __future__ import annotations

import importlib
import pathlib
import sys

EXPERIMENT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))


def test_core_schema_imports_without_training_frameworks():
    module = importlib.import_module("core.schema")
    assert module.SCHEMA_VERSION == "cpc-common-v0"


def test_core_schema_source_has_no_framework_imports():
    source = (EXPERIMENT_ROOT / "core" / "schema.py").read_text(encoding="utf-8").lower()
    forbidden = ["torch", "torchrl", "tensordict", "gym"]
    assert not any(token in source for token in forbidden)


def test_no_active_python_source_imports_old_core_name():
    stale_import = "cpc" + "_core"
    offenders = []
    for path in EXPERIMENT_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if f"import {stale_import}" in text or f"from {stale_import}" in text:
            offenders.append(str(path.relative_to(EXPERIMENT_ROOT)))
    assert offenders == []
