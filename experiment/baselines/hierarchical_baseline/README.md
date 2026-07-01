# Hierarchical Baseline

The hierarchical baseline separates world context, global goal planning,
intent selection, local planning, control, and action construction.

`HierarchicalBaselineAgent.act()` is the only state mutation boundary. Every
other module exposes stateless functions that return values plus debug data.

```text
build_context
-> create_global_plan_if_needed
-> select_intent
-> create_local_plan
-> controller
-> build_action
```

Run goal navigation:

```bash
python experiment/baselines/hierarchical_baseline/run_hierarchical_autoplay.py \
  --config configs/env/autoplay_goal_loop.yaml \
  --steps 100 \
  --print-debug
```

Run a combat scenario by replacing the config with
`configs/env/autoplay_enemy_right.yaml`.
