# CPC Intent Dataset Generator

This package generates a deterministic experiment-side dataset for linear intent classification:

`observation features -> intent label`

It is intended for browser upload workflows and offline training experiments. It does not change runtime simulation or policy behavior.

## Intent Classes

The dataset contains four scenario families and four labels:

- `direct_enemy_contact` -> `attack_nearest_enemy`
- `teammate_under_pressure` -> `support_teammate_under_pressure`
- `isolated_teammate` -> `reduce_isolation`
- `self_low_hp` -> `retreat_when_low_hp`

## Feature Schema

Feature order is fixed and exported in `feature_schema.json`.

Model features, in exact order:

1. `selfHpNorm`
2. `canFire`
3. `allyHpNorm`
4. `allyDistanceNorm`
5. `enemy0HpNorm`
6. `enemy0DistanceNorm`
7. `enemy1HpNorm`
8. `enemy1DistanceNorm`

The JSON samples also include `predicateDebug` for validation/debugging only. It is not written to the CSV.

The intent model intentionally uses low-level features only. High-level predicates stay in JSON for validation and notebook inspection, not as training inputs.

## Label Schema

The label order is fixed in `label_schema.json`:

1. `attack_nearest_enemy`
2. `support_teammate_under_pressure`
3. `reduce_isolation`
4. `retreat_when_low_hp`

## Generation

```bash
python experiment/training/cpc_intent_dataset/generate_intent_dataset.py --num-per-scenario 100 --eval-ratio 0.2 --seed 42 --output-dir experiment/training/data/intent_dataset
```

## Self-Check

```bash
python experiment/training/cpc_intent_dataset/generate_intent_dataset.py --self-check
```

The self-check validates:

- all four labels are present
- train and eval splits are non-empty
- feature order is fixed
- `featureVector` length matches `featureCount`
- recomputed features match stored features
- `featureVector` matches the exported feature order
- no NaN or Infinity values exist
- scenario constraints hold for each family
- canFire is present in both states across the dataset
- enemy HP and enemy1 distance are not constant
- teammate_under_pressure covers a wider ally distance range
- JSON arrays are parseable by `JSON.parse`
- JSONL parses line by line
- CSV loads with Python's `csv` module
- `dataset_manifest.json` is valid

## Browser Upload Targets

The browser should upload the JSON array files:

- `train_intent_dataset.json`
- `eval_intent_dataset.json`

These are used because they are directly parseable with `JSON.parse` and preserve the full sample schema without extra decoding steps.

## Notebook Inspection

If you want to inspect a single JSON sample inside a notebook, import the helper:

```python
from visualize import format_sample_status, inspect_sample_status
```

Then pass a parsed JSON sample into `format_sample_status(sample)` to print the current status, or use `inspect_sample_status(sample)` for a structured dictionary.

If you want to render the sample state, call `plot_sample_state(sample)`.

## Generated Artifacts

Generated files are written to:

`experiment/training/data/intent_dataset/`
