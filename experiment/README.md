# Experiment

The active Python CPC workflow is the tactical-baseline path documented in
`docs/ACTIVE_PATH.md`.

Current entry points:

```bash
python scripts/manual_env_debug.py --config configs/env/manual_enemy_right.yaml --steps 5 --actions stay,right,right,aim_right,fire --no-gui --no-grid-png
python experiment/baselines/aim_oracle/run_aim_oracle_debug.py --config configs/env/manual_enemy_right.yaml --steps 5
python experiment/baselines/move_score/run_move_score_debug.py --config configs/env/manual_enemy_far_right.yaml --steps 10
python experiment/baselines/tactical_baseline/run_tactical_autoplay.py --config configs/env/autoplay_enemy_right.yaml --steps 100 --fps 20 --print-debug --print-every 20
```

Older PPO, TorchRL, checkpoint, model-gameplay, scripted-baseline, and notebook
experiments were moved to `legacy/python_experiments/` for reference.
