"""survev_rl: PPO training against the survev CPC bridge (protocol v0).

Modules
-------
protocol       typed views of the bridge protocol + JSON helpers + key allowlist
bridge_client  synchronous WebSocket client (reset / step / step_batch / close)
mock_bridge    in-process mock bridge server with a kinematic 2v2 field simulation
featurizer     AgentObservation -> fixed-size float32 vector (+ vector_keys)
actions        MultiDiscrete / skill action spaces -> CpcAction
rewards        RewardConfig + per-agent reward computation
env            SurvevVecEnv / SurvevSingleEnv (no gymnasium dependency)
ppo            CleanRL-style PPO for MultiDiscrete actions (torch)
train_ppo      training CLI          eval        evaluation CLI
export_onnx    actor -> ONNX (dynamic batch) for onnxruntime-node
"""

__all__ = [
    "protocol",
    "bridge_client",
    "mock_bridge",
    "featurizer",
    "actions",
    "rewards",
    "env",
]
