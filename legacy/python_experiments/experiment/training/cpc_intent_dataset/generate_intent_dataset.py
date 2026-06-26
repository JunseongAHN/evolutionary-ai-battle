from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

from features import (
    DATASET_SCHEMA_VERSION,
    DEFAULT_THRESHOLDS,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    FEATURE_TYPES,
    LABELS,
    LABEL_SCHEMA_VERSION,
    LABEL_TO_INDEX,
    extract_model_features,
    feature_vector_from_features,
)
from scenario_dataset import SCENARIO_INTENTS, _compute_predicate_debug, generate_dataset

OUTPUT_FILES = {
    "train_json": "train_intent_dataset.json",
    "eval_json": "eval_intent_dataset.json",
    "train_jsonl": "train_intent_dataset.jsonl",
    "eval_jsonl": "eval_intent_dataset.jsonl",
    "train_csv": "train_intent_dataset.csv",
    "eval_csv": "eval_intent_dataset.csv",
    "feature_schema": "feature_schema.json",
    "label_schema": "label_schema.json",
    "manifest": "dataset_manifest.json",
}

BROWSER_UPLOAD_TARGETS = [
    OUTPUT_FILES["train_json"],
    OUTPUT_FILES["eval_json"],
]

CSV_COLUMNS = [
    "sampleId",
    *FEATURE_NAMES,
    "output",
]


def _json_dump(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _write_jsonl(path: Path, samples: Iterable[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False))
            handle.write("\n")


def _write_csv(path: Path, samples: Iterable[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for sample in samples:
            row = {"sampleId": sample["sampleId"], "output": sample["labelIndex"]}
            row.update(sample["features"])
            writer.writerow(row)


def _count_samples(samples: Iterable[Dict[str, object]]) -> Dict[str, Dict[str, int]]:
    by_split = Counter()
    by_scenario = Counter()
    by_label = Counter()
    for sample in samples:
        by_split[sample["split"]] += 1
        by_scenario[sample["scenarioId"]] += 1
        by_label[sample["label"]] += 1
    return {
        "bySplit": dict(by_split),
        "byScenario": dict(by_scenario),
        "byLabel": dict(by_label),
    }


def _init_stat_summary():
    return {"min": None, "max": None, "sum": 0.0, "count": 0}


def _update_stat_summary(summary, value):
    if value is None:
        return
    value = float(value)
    summary["min"] = value if summary["min"] is None else min(summary["min"], value)
    summary["max"] = value if summary["max"] is None else max(summary["max"], value)
    summary["sum"] += value
    summary["count"] += 1


def _finalize_stat_summary(summary):
    count = summary["count"]
    return {
        "min": summary["min"],
        "max": summary["max"],
        "mean": round(summary["sum"] / count, 6) if count else None,
        "count": count,
    }


def _build_distribution_summary(samples: List[Dict[str, object]]) -> Dict[str, object]:
    summary = {
        "allyDistanceNormByScenario": {},
        "enemy0HpNorm": _init_stat_summary(),
        "enemy1HpNorm": _init_stat_summary(),
        "enemy1DistanceNorm": _init_stat_summary(),
    }

    for sample in samples:
        scenario_id = sample["scenarioId"]
        ally_bucket = summary["allyDistanceNormByScenario"].setdefault(scenario_id, _init_stat_summary())
        _update_stat_summary(ally_bucket, sample["features"]["allyDistanceNorm"])
        _update_stat_summary(summary["enemy0HpNorm"], sample["features"]["enemy0HpNorm"])
        _update_stat_summary(summary["enemy1HpNorm"], sample["features"]["enemy1HpNorm"])
        _update_stat_summary(summary["enemy1DistanceNorm"], sample["features"]["enemy1DistanceNorm"])

    return {
        "allyDistanceNormByScenario": {scenario_id: _finalize_stat_summary(bucket) for scenario_id, bucket in summary["allyDistanceNormByScenario"].items()},
        "enemy0HpNorm": _finalize_stat_summary(summary["enemy0HpNorm"]),
        "enemy1HpNorm": _finalize_stat_summary(summary["enemy1HpNorm"]),
        "enemy1DistanceNorm": _finalize_stat_summary(summary["enemy1DistanceNorm"]),
    }


def _build_manifest(seed: int, num_per_scenario: int, eval_ratio: float, all_samples: List[Dict[str, object]], output_dir: Path) -> Dict[str, object]:
    return {
        "schemaVersion": DATASET_SCHEMA_VERSION,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "generator": "experiment.training.cpc_intent_dataset.generate_intent_dataset",
        "seed": seed,
        "numPerScenario": num_per_scenario,
        "evalRatio": eval_ratio,
        "thresholds": DEFAULT_THRESHOLDS,
        "featureSchemaVersion": FEATURE_SCHEMA_VERSION,
        "labelSchemaVersion": LABEL_SCHEMA_VERSION,
        "splitStrategy": "deterministic per-scenario shuffle using seeded hashes; eval samples are selected by the lowest random keys within each scenario",
        "sampleCounts": _count_samples(all_samples),
        "distributionSummary": _build_distribution_summary(all_samples),
        "outputFiles": OUTPUT_FILES,
        "browserUploadTargets": BROWSER_UPLOAD_TARGETS,
        "outputDir": str(output_dir),
    }


def _write_all_outputs(output_dir: Path, train_samples: List[Dict[str, object]], eval_samples: List[Dict[str, object]], manifest: Dict[str, object]) -> None:
    feature_schema = {
        "schemaVersion": FEATURE_SCHEMA_VERSION,
        "featureNames": FEATURE_NAMES,
        "featureTypes": [FEATURE_TYPES[name] for name in FEATURE_NAMES],
        "featureCount": len(FEATURE_NAMES),
    }
    label_schema = {
        "schemaVersion": LABEL_SCHEMA_VERSION,
        "labels": LABELS,
        "labelToIndex": LABEL_TO_INDEX,
    }

    _json_dump(output_dir / OUTPUT_FILES["train_json"], train_samples)
    _json_dump(output_dir / OUTPUT_FILES["eval_json"], eval_samples)
    _write_jsonl(output_dir / OUTPUT_FILES["train_jsonl"], train_samples)
    _write_jsonl(output_dir / OUTPUT_FILES["eval_jsonl"], eval_samples)
    _write_csv(output_dir / OUTPUT_FILES["train_csv"], train_samples)
    _write_csv(output_dir / OUTPUT_FILES["eval_csv"], eval_samples)
    _json_dump(output_dir / OUTPUT_FILES["feature_schema"], feature_schema)
    _json_dump(output_dir / OUTPUT_FILES["label_schema"], label_schema)
    _json_dump(output_dir / OUTPUT_FILES["manifest"], manifest)


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _assert_finite_numbers(value: object, path: str = "root") -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AssertionError(f"{path} contains non-finite float")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite_numbers(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_finite_numbers(item, f"{path}.{key}")


def self_check(output_dir: Path | None = None) -> None:
    temp_dir = None
    if output_dir is None:
        temp_dir = tempfile.TemporaryDirectory()
        output_dir = Path(temp_dir.name)

    all_samples, grouped = generate_dataset()
    manifest = _build_manifest(seed=42, num_per_scenario=100, eval_ratio=0.2, all_samples=all_samples, output_dir=output_dir)
    _write_all_outputs(output_dir, grouped["train"], grouped["eval"], manifest)

    feature_schema = _load_json(output_dir / OUTPUT_FILES["feature_schema"])
    label_schema = _load_json(output_dir / OUTPUT_FILES["label_schema"])
    manifest_loaded = _load_json(output_dir / OUTPUT_FILES["manifest"])
    train_json = _load_json(output_dir / OUTPUT_FILES["train_json"])
    eval_json = _load_json(output_dir / OUTPUT_FILES["eval_json"])

    assert feature_schema["featureNames"] == FEATURE_NAMES
    assert feature_schema["featureCount"] == len(FEATURE_NAMES)
    assert label_schema["labels"] == LABELS
    assert set(label_schema["labelToIndex"].keys()) == set(LABELS)
    assert train_json and eval_json
    assert {sample["label"] for sample in train_json + eval_json} == set(LABELS)

    combined = train_json + eval_json
    for sample in train_json + eval_json:
        recomputed_features = extract_model_features(sample["state"], DEFAULT_THRESHOLDS)
        assert sample["features"] == recomputed_features
        assert sample["featureVector"] == feature_vector_from_features(recomputed_features)
        assert len(sample["featureVector"]) == feature_schema["featureCount"]
        assert sample["predicateDebug"] == _compute_predicate_debug(sample["state"])
        _assert_finite_numbers(sample)

    scenario_checks = {
        "direct_enemy_contact": lambda sample: sample["predicateDebug"]["enemyNearby"] == 1,
        "teammate_under_pressure": lambda sample: sample["predicateDebug"]["allyUnderPressure"] == 1,
        "isolated_teammate": lambda sample: sample["predicateDebug"]["isIsolated"] == 1,
        "self_low_hp": lambda sample: sample["predicateDebug"]["selfLowHp"] == 1,
    }
    for scenario_id, check in scenario_checks.items():
        scenario_samples = [sample for sample in combined if sample["scenarioId"] == scenario_id]
        assert scenario_samples, scenario_id
        assert any(check(sample) for sample in scenario_samples), scenario_id
    distribution_summary = manifest_loaded["distributionSummary"]
    assert distribution_summary["enemy0HpNorm"]["min"] is not None
    assert distribution_summary["enemy0HpNorm"]["max"] is not None
    assert distribution_summary["enemy1HpNorm"]["min"] is not None
    assert distribution_summary["enemy1HpNorm"]["max"] is not None
    assert distribution_summary["enemy1DistanceNorm"]["min"] is not None
    assert distribution_summary["enemy1DistanceNorm"]["max"] is not None
    assert distribution_summary["enemy0HpNorm"]["min"] != distribution_summary["enemy0HpNorm"]["max"]
    assert distribution_summary["enemy1HpNorm"]["min"] != distribution_summary["enemy1HpNorm"]["max"]
    assert distribution_summary["enemy1DistanceNorm"]["min"] != distribution_summary["enemy1DistanceNorm"]["max"]
    teammate_stats = distribution_summary["allyDistanceNormByScenario"]["teammate_under_pressure"]
    assert teammate_stats["min"] is not None and teammate_stats["max"] is not None
    assert teammate_stats["min"] <= 0.5
    assert teammate_stats["max"] >= 0.15
    for scenario_id, stats in distribution_summary["allyDistanceNormByScenario"].items():
        assert stats["min"] is not None, scenario_id
        assert stats["max"] is not None, scenario_id

    assert grouped["train"]
    assert grouped["eval"]

    for name in (OUTPUT_FILES["train_json"], OUTPUT_FILES["eval_json"]):
        payload = _load_json(output_dir / name)
        assert isinstance(payload, list)

    for name in (OUTPUT_FILES["train_jsonl"], OUTPUT_FILES["eval_jsonl"]):
        with (output_dir / name).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                json.loads(line)

    for name in (OUTPUT_FILES["train_csv"], OUTPUT_FILES["eval_csv"]):
        with (output_dir / name).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            assert rows
            assert reader.fieldnames == CSV_COLUMNS

    assert manifest_loaded["schemaVersion"] == DATASET_SCHEMA_VERSION
    assert manifest_loaded["browserUploadTargets"] == BROWSER_UPLOAD_TARGETS
    assert manifest_loaded["sampleCounts"]["byLabel"]
    assert manifest_loaded["sampleCounts"]["bySplit"]["train"] > 0
    assert manifest_loaded["sampleCounts"]["bySplit"]["eval"] > 0

    if temp_dir is not None:
        temp_dir.cleanup()


def generate_and_write(num_per_scenario: int, eval_ratio: float, seed: int, output_dir: Path) -> None:
    all_samples, grouped = generate_dataset(num_per_scenario=num_per_scenario, eval_ratio=eval_ratio, seed=seed)
    manifest = _build_manifest(seed=seed, num_per_scenario=num_per_scenario, eval_ratio=eval_ratio, all_samples=all_samples, output_dir=output_dir)
    _write_all_outputs(output_dir, grouped["train"], grouped["eval"], manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a linear GT intent dataset.")
    parser.add_argument("--num-per-scenario", type=int, default=100)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("experiment/training/data/intent_dataset"))
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check()
        print("Self-check passed.")
        return
    generate_and_write(args.num_per_scenario, args.eval_ratio, args.seed, args.output_dir)
    print(f"Generated dataset at {args.output_dir}")


if __name__ == "__main__":
    main()
