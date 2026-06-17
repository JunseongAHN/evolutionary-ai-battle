# Common Interface v0

## Why This Schema Exists

The CPC common schema is the stable JSON bridge between a Survev-like world model, the TypeScript replay/dashboard viewer, future Survev.io adapters, and future training backends. It defines shared battle configs, snapshots, tactical observations, actions, steps, trajectories, events, and metric vectors without tying the project to one runtime.

## Connected Systems

Python world model:
Exports `BattleSnapshot`, `TacticalObservation`, `BattleAction`, `MultiAgentStep`, and `EpisodeTrajectory` records as plain JSON or JSONL. The lightweight Python helpers live in `experiment/core/`.

TypeScript viewer:
Consumes the same JSON-compatible records through `src/common/schema.ts` and validates them with `src/common/schemaValidation.ts`.

Survev.io adapter:
Can later convert Survev state into `BattleSnapshot` / `TacticalObservation`, send common `BattleAction` objects to an agent core, and adapt them back into Survev.io actions.

Future TorchRL backend:
Can batch `agent_ids` ordered observations and actions into tensors without changing the serialized schema.

## No TorchRL Dependency

This schema is TorchRL-friendly, not TorchRL-dependent. It imports no TorchRL, PyTorch, BenchMARL, Gymnasium, PettingZoo, JAX, or simulator-specific code. The serialized contract is plain JSON: arrays, objects, numbers, strings, booleans, and null-compatible optional fields.

## Observation Design

Each step stores observations as `Record<AgentId, TacticalObservation>`, with a deterministic `agent_ids` order in the step snapshot. Each observation includes:

- a numeric `vector` and matching `vector_keys`,
- bounded visible enemy/ally/obstacle/event arrays,
- masks for variable-length entity lists,
- structured entity fields for reasoning and debugging.

This supports both human-readable tactical inspection and later tensor conversion.

## Rewards And Metrics

Rewards are optional in `MultiAgentStep`. Metric-vector evaluation is primary and remains in `info.metrics` or `final_metrics`.

Combat, survival, cooperation, and movement metrics stay separate. The schema does not collapse evaluation into a scalar reward.

## Solo And Duo Modes

Solo mode uses `players_per_team = 1`; every agent is its own team. Cooperation metrics must use `cooperation.applicable = false`.

Duo mode uses `players_per_team = 2`; each team has exactly two agents. Cooperation metrics may include teammate pressure, support response, isolation, and ally-distance fields.

## TorchRL Adaptation Notes

- agent_ids defines deterministic agent ordering
- observations are Record<AgentId, TacticalObservation>
- vector field is already numeric
- variable-length visible entities are represented with max counts and masks
- entity arrays can be padded to fixed size
- actions use fixed numeric keys
- rewards are optional because metric-vector evaluation is primary
- metrics stay in info, not collapsed into reward

Future adapter only. Do not implement TorchRL now.

```python
agent_order = step["info"]["snapshot"]["agent_ids"]

obs_tensor = stack([
    step["observations"][agent_id]["vector"]
    for agent_id in agent_order
])

action_tensor = stack([
    [
        step["actions"][agent_id]["action"]["move_x"],
        step["actions"][agent_id]["action"]["move_y"],
        step["actions"][agent_id]["action"]["aim_x"],
        step["actions"][agent_id]["action"]["aim_y"],
        step["actions"][agent_id]["action"]["fire"],
    ]
    for agent_id in agent_order
])
```

## File Locations

- Serialized schema: `schemas/cpc_common_schema.v0.json`
- TypeScript types: `src/common/schema.ts`
- TypeScript validation: `src/common/schemaValidation.ts`
- Python dataclasses: `experiment/core/schema.py`
- Python validation: `experiment/core/schema_validation.py`
- Sample episode JSONL: `experiment/core/examples/sample_episode_v0.jsonl`
