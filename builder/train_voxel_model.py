"""Train a coordinate-conditioned neural network to emit every mansion block."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from builder.mansion import render_mansion_blocks
from builder.voxel_model import (
    GRID_BOUNDS,
    HIDDEN_SIZES,
    VoxelBlockNet,
    all_grid_coordinates,
    encode_inputs,
)
from env.builder_env import BuilderEnv, DESIGN_CHOICES, THEMES


def _policy_designs(model: PPO, seed: int) -> dict[int, list[int]]:
    env = BuilderEnv()
    designs: dict[int, list[int]] = {}
    for theme_id in range(len(THEMES)):
        obs, _ = env.reset(seed=seed + theme_id, options={"theme": theme_id})
        choices = []
        for _ in DESIGN_CHOICES:
            action, _ = model.predict(obs, deterministic=True)
            choice = int(np.asarray(action).item())
            choices.append(choice)
            obs, _, terminated, truncated, _ = env.step(choice)
            if terminated or truncated:
                break
        if len(choices) != len(DESIGN_CHOICES):
            raise RuntimeError("policy returned an incomplete design")
        designs[theme_id] = choices
    env.close()
    return designs


def _make_condition_examples(count: int, seed: int, policy_designs: dict[int, list[int]]):
    rng = np.random.default_rng(seed)
    examples = []
    for index in range(count):
        theme_id = index % len(THEMES)
        choices = list(policy_designs[theme_id])
        if index >= len(THEMES):
            if index % 11 == 0:
                choices = [int(rng.integers(options)) for options in DESIGN_CHOICES]
            else:
                mutation_count = int(rng.integers(1, 5))
                stages = rng.choice(len(DESIGN_CHOICES), size=mutation_count, replace=False)
                for stage in stages:
                    choices[int(stage)] = int(rng.integers(DESIGN_CHOICES[int(stage)]))
        examples.append((theme_id, choices))
    return examples


def _flat_index(x: int, y: int, z: int) -> int:
    b = GRID_BOUNDS
    return ((x - b["x_min"]) * b["y_size"] + (y - b["y_min"])) * b["z_size"] + (z - b["z_min"])


def _build_dataset(examples, seed: int, negatives_per_positive: int):
    rng = np.random.default_rng(seed)
    coordinates = all_grid_coordinates()
    rendered = []
    block_set = {"minecraft:air"}
    for theme_id, choices in examples:
        blocks = render_mansion_blocks({"choices": choices})
        for x, y, z in blocks:
            b = GRID_BOUNDS
            if not (b["x_min"] <= x < b["x_min"] + b["x_size"]
                    and b["y_min"] <= y < b["y_min"] + b["y_size"]
                    and b["z_min"] <= z < b["z_min"] + b["z_size"]):
                raise ValueError(f"teacher placement {(x, y, z)} is outside voxel grid")
        block_set.update(blocks.values())
        rendered.append((theme_id, choices, blocks))

    block_names = ["minecraft:air"] + sorted(block_set - {"minecraft:air"})
    block_to_id = {name: index for index, name in enumerate(block_names)}
    full_counts = np.zeros(len(block_names), dtype=np.int64)
    sampled_features = []
    sampled_labels = []
    plan_ids = []

    for example_id, (theme_id, choices, blocks) in enumerate(rendered):
        target = np.zeros(len(coordinates), dtype=np.int32)
        positive_indices = np.fromiter(
            (_flat_index(x, y, z) for x, y, z in blocks), dtype=np.int64, count=len(blocks)
        )
        positive_labels = np.fromiter(
            (block_to_id[blocks[tuple(int(value) for value in coordinates[index])]]
             for index in positive_indices), dtype=np.int32, count=len(positive_indices)
        )
        target[positive_indices] = positive_labels
        full_counts += np.bincount(target, minlength=len(block_names))
        negative_pool = np.flatnonzero(target == 0)
        negative_count = min(len(negative_pool), len(positive_indices) * negatives_per_positive)
        negative_indices = rng.choice(negative_pool, size=negative_count, replace=False)
        selected_indices = np.concatenate((positive_indices, negative_indices))
        selected_labels = target[selected_indices]
        sampled_features.append(encode_inputs(coordinates[selected_indices], theme_id, choices))
        sampled_labels.append(selected_labels)
        plan_ids.append(np.full(len(selected_indices), example_id, dtype=np.int32))

    features = np.concatenate(sampled_features).astype(np.float32, copy=False)
    labels = np.concatenate(sampled_labels).astype(np.int64, copy=False)
    example_ids = np.concatenate(plan_ids)
    sampled_counts = np.bincount(labels, minlength=len(block_names)).astype(np.float64)
    full_prior = full_counts / max(1, int(full_counts.sum()))
    sampled_prior = sampled_counts / max(1.0, float(sampled_counts.sum()))
    prior_correction = np.log(np.maximum(full_prior, 1e-10)) - np.log(np.maximum(sampled_prior, 1e-10))
    return features, labels, example_ids, block_names, prior_correction.astype(np.float32), full_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="models/builder/builder_500000_episodes")
    parser.add_argument("--out", type=Path, default=Path("models/builder/voxel_mansion_net.pt"))
    parser.add_argument("--examples", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--negatives-per-positive", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261126)
    args = parser.parse_args()
    if args.examples < len(THEMES) * 2 or args.epochs <= 0 or args.batch_size < 2:
        raise SystemExit("need at least 8 examples, positive epochs, and batch size >= 2")
    if args.negatives_per_positive <= 0:
        raise SystemExit("--negatives-per-positive must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    started = time.monotonic()
    policy = PPO.load(args.policy, device="cpu")
    policy_designs = _policy_designs(policy, args.seed)
    condition_examples = _make_condition_examples(args.examples, args.seed, policy_designs)
    features, labels, example_ids, block_names, prior_correction, full_counts = _build_dataset(
        condition_examples, args.seed, args.negatives_per_positive
    )

    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(args.examples)
    val_count = max(1, args.examples // 10)
    val_plan_ids = set(int(value) for value in permutation[:val_count])
    val_mask = np.fromiter((int(value) in val_plan_ids for value in example_ids), dtype=bool)
    train_mask = ~val_mask
    train_x = torch.from_numpy(features[train_mask])
    train_y = torch.from_numpy(labels[train_mask])
    val_x = torch.from_numpy(features[val_mask])
    val_y = torch.from_numpy(labels[val_mask])
    train_loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True, drop_last=False
    )

    sample_counts = np.bincount(labels[train_mask], minlength=len(block_names)).astype(np.float32)
    class_weights = 1.0 / np.sqrt(np.maximum(sample_counts, 1.0))
    class_weights /= class_weights.mean()
    class_weights = np.clip(class_weights, 0.25, 5.0)
    class_weights[0] = min(class_weights[0], 0.7)

    model = VoxelBlockNet(len(block_names), HIDDEN_SIZES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(class_weights, dtype=torch.float32))
    best_state = None
    best_val_loss = float("inf")
    history = []
    print(
        f"Training coordinate-to-block net: examples={args.examples}, "
        f"sampled_voxels={len(features)}, grid={GRID_BOUNDS}, block_types={len(block_names)}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))

        model.eval()
        with torch.inference_mode():
            val_logits = model(val_x)
            val_loss = float(criterion(val_logits, val_y))
            val_accuracy = float((val_logits.argmax(dim=1) == val_y).float().mean())
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "validation_loss": val_loss,
            "validation_accuracy_balanced_samples": val_accuracy,
        })
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"epoch={epoch}/{args.epochs} train_loss={history[-1]['train_loss']:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_accuracy:.3f}", flush=True)

    if best_state is None:
        raise RuntimeError("voxel model did not produce a checkpoint")
    model.load_state_dict(best_state)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "block_names": block_names,
        "hidden_sizes": HIDDEN_SIZES,
        "grid_bounds": GRID_BOUNDS,
        "prior_correction": prior_correction,
        "condition_size": 60,
        "input_size": 63,
        "training_examples": args.examples,
        "epochs": args.epochs,
        "seed": args.seed,
        "policy_checkpoint": args.policy,
        "training_source": "procedural mansion teacher plans; inference predicts every block cell directly",
        "validation_loss": best_val_loss,
        "full_block_counts": full_counts.tolist(),
        "history": history,
        "elapsed_seconds": time.monotonic() - started,
    }
    torch.save(checkpoint, args.out)
    summary_path = args.out.with_suffix(".json")
    summary = {key: value for key, value in checkpoint.items() if key not in {"state_dict", "prior_correction"}}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "model": str(args.out),
        "block_types": len(block_names),
        "validation_loss": best_val_loss,
        "elapsed_seconds": checkpoint["elapsed_seconds"],
        "summary": str(summary_path),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
