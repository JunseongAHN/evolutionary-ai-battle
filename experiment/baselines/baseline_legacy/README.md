# Legacy Tactical Baseline

This is the old tactical baseline kept for comparison.

New development should happen in `experiment/baselines/hierarchical_baseline/`.

Run the preserved baseline with:

```bash
python experiment/baselines/baseline_legacy/run_tactical_autoplay.py \
  --config configs/env/autoplay_enemy_right.yaml \
  --steps 100 \
  --print-debug
```
