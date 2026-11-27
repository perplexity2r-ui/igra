#!/usr/bin/env python3
"""Compare the exported Java actor's NumPy forward pass with Stable-Baselines3."""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

OBS, H1, H2, ACTIONS = 44, 192, 192, 30


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        default=root / "models/terrain_finetune/pvp_ppo_terrain_aggressive_50k.zip")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "src/main/resources/com/arena/aipilot/ppo_policy.bin",
    )
    parser.add_argument("--samples", type=int, default=512)
    args = parser.parse_args()

    raw = args.weights.read_bytes()
    magic, version, obs_dim, h1, h2, actions = struct.unpack_from("<4sIIIII", raw, 0)
    if (magic, version, obs_dim, h1, h2, actions) != (b"AIPP", 1, OBS, H1, H2, ACTIONS):
        raise SystemExit("Unexpected exported actor header")
    flat = np.frombuffer(raw, dtype="<f4", offset=24).astype(np.float32)
    shapes = [(H1, OBS), (H1,), (H2, H1), (H2,), (ACTIONS, H2), (ACTIONS,)]
    tensors = []
    offset = 0
    for shape in shapes:
        size = int(np.prod(shape))
        tensors.append(flat[offset : offset + size].reshape(shape))
        offset += size
    if offset != flat.size:
        raise SystemExit("Export has trailing or missing actor parameters")
    w1, b1, w2, b2, w3, b3 = tensors

    model = PPO.load(str(args.model), device="cpu")
    rng = np.random.default_rng(20261125)
    observations = rng.uniform(-1.0, 2.0, (args.samples, OBS)).astype(np.float32)
    hidden1 = np.tanh(observations @ w1.T + b1)
    hidden2 = np.tanh(hidden1 @ w2.T + b2)
    exported_logits = hidden2 @ w3.T + b3

    with torch.no_grad():
        features = model.policy.extract_features(torch.as_tensor(observations))
        actor_latent = model.policy.mlp_extractor.forward_actor(features)
        reference_logits = model.policy.action_net(actor_latent).cpu().numpy()
    max_error = float(np.max(np.abs(exported_logits - reference_logits)))
    mismatches = int(np.count_nonzero(exported_logits.argmax(1) != reference_logits.argmax(1)))
    print(f"Compared {args.samples} observations; max |logit delta|={max_error:.8g}; action mismatches={mismatches}")
    if not np.allclose(exported_logits, reference_logits, rtol=2e-5, atol=2e-5) or mismatches:
        raise SystemExit("Exported actor does not match the SB3 policy")
    print("PPO weight export verified.")


if __name__ == "__main__":
    main()
