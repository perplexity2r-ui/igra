#!/usr/bin/env python3
"""Export the terrain-aware deterministic PPO actor for the Forge client.

The Forge client runs inference with a tiny pure-Java MLP, so it does not need
Python, PyTorch, or Stable-Baselines3 at runtime. Re-run this after fine-tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import struct
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

OBS_DIM = 44
HIDDEN_1 = 192
HIDDEN_2 = 192
ACTION_DIM = 30
MAGIC = b"AIPP"
FORMAT_VERSION = 1


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "models" / "terrain_finetune" / "pvp_ppo_terrain_aggressive_50k.zip",
        help="Stable-Baselines3 PPO .zip model",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "src/main/resources/com/arena/aipilot/ppo_policy.bin",
        help="Binary actor resource packaged in the Forge JAR",
    )
    args = parser.parse_args()

    model = PPO.load(str(args.model), device="cpu")
    if tuple(model.observation_space.shape) != (OBS_DIM,):
        raise SystemExit(f"Expected Box({OBS_DIM},), got {model.observation_space}")
    if int(model.action_space.n) != ACTION_DIM:
        raise SystemExit(f"Expected Discrete({ACTION_DIM}), got {model.action_space}")

    state = model.policy.state_dict()
    expected = {
        "mlp_extractor.policy_net.0.weight": (HIDDEN_1, OBS_DIM),
        "mlp_extractor.policy_net.0.bias": (HIDDEN_1,),
        "mlp_extractor.policy_net.2.weight": (HIDDEN_2, HIDDEN_1),
        "mlp_extractor.policy_net.2.bias": (HIDDEN_2,),
        "action_net.weight": (ACTION_DIM, HIDDEN_2),
        "action_net.bias": (ACTION_DIM,),
    }
    chunks: list[bytes] = []
    for name, shape in expected.items():
        if name not in state or tuple(state[name].shape) != shape:
            raise SystemExit(f"Unexpected actor tensor {name}: {state.get(name, None)}")
        array = state[name].detach().cpu().numpy().astype("<f4", copy=False)
        chunks.append(np.ascontiguousarray(array).tobytes(order="C"))

    header = struct.pack(
        "<4sIIIII", MAGIC, FORMAT_VERSION, OBS_DIM, HIDDEN_1, HIDDEN_2, ACTION_DIM
    )
    payload = header + b"".join(chunks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(f"Exported {sum(np.prod(s) for s in expected.values()):,} actor parameters")
    print(f"Output: {args.output} ({len(payload):,} bytes)")
    print(f"SHA-256: {hashlib.sha256(payload).hexdigest()}")


if __name__ == "__main__":
    main()
