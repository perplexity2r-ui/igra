"""Conditioned coordinate-to-block neural model for complete voxel plans."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from env.builder_env import N_OPTIONS, N_STAGES, THEMES

# Fixed local grid for the current mansion envelope: x -15..15, y 0..16, z -13..17.
GRID_BOUNDS = {"x_min": -15, "y_min": 0, "z_min": -13, "x_size": 31, "y_size": 17, "z_size": 31}
COORDINATE_COUNT = GRID_BOUNDS["x_size"] * GRID_BOUNDS["y_size"] * GRID_BOUNDS["z_size"]
CONDITION_SIZE = len(THEMES) + N_STAGES * N_OPTIONS
INPUT_SIZE = 3 + CONDITION_SIZE
HIDDEN_SIZES = (128, 128)


class VoxelBlockNet(nn.Module):
    """Predict the block token at any local 3D coordinate from style and position."""

    def __init__(self, num_blocks: int, hidden_sizes: tuple[int, ...] = HIDDEN_SIZES):
        super().__init__()
        layers: list[nn.Module] = []
        in_features = INPUT_SIZE
        for width in hidden_sizes:
            layers.extend((nn.Linear(in_features, width), nn.GELU()))
            in_features = width
        layers.append(nn.Linear(in_features, num_blocks))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def encode_condition(theme_id: int, choices: list[int]) -> np.ndarray:
    if not 0 <= int(theme_id) < len(THEMES):
        raise ValueError(f"theme_id must be between 0 and {len(THEMES) - 1}")
    if len(choices) != N_STAGES or any(not 0 <= int(choice) < N_OPTIONS for choice in choices):
        raise ValueError(f"choices must have {N_STAGES} entries in 0..{N_OPTIONS - 1}")
    encoded = np.zeros(CONDITION_SIZE, dtype=np.float32)
    encoded[int(theme_id)] = 1.0
    offset = len(THEMES)
    for stage, choice in enumerate(choices):
        encoded[offset + stage * N_OPTIONS + int(choice)] = 1.0
    return encoded


def all_grid_coordinates() -> np.ndarray:
    b = GRID_BOUNDS
    xs = np.arange(b["x_min"], b["x_min"] + b["x_size"], dtype=np.int16)
    ys = np.arange(b["y_min"], b["y_min"] + b["y_size"], dtype=np.int16)
    zs = np.arange(b["z_min"], b["z_min"] + b["z_size"], dtype=np.int16)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack((gx, gy, gz), axis=-1).reshape(-1, 3)


def encode_inputs(coords: np.ndarray, theme_id: int, choices: list[int]) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3) in local x,y,z order")
    b = GRID_BOUNDS
    normalized = np.column_stack((
        coords[:, 0] / max(abs(b["x_min"]), abs(b["x_min"] + b["x_size"] - 1)),
        (coords[:, 1] - b["y_min"]) / max(1, b["y_size"] - 1) * 2.0 - 1.0,
        (coords[:, 2] / max(abs(b["z_min"]), abs(b["z_min"] + b["z_size"] - 1))),
    )).astype(np.float32)
    condition = encode_condition(theme_id, choices)
    return np.concatenate((normalized, np.broadcast_to(condition, (len(coords), len(condition)))), axis=1)


def load_voxel_model(path: str | Path, device: str = "cpu") -> tuple[VoxelBlockNet, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = VoxelBlockNet(len(checkpoint["block_names"]), tuple(checkpoint["hidden_sizes"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def predict_block_ids(
    model: VoxelBlockNet,
    checkpoint: dict[str, Any],
    coords: np.ndarray,
    theme_id: int,
    choices: list[int],
    *,
    batch_size: int = 4096,
    device: str = "cpu",
    nonair_bias: float | None = None,
) -> np.ndarray:
    features = encode_inputs(coords, theme_id, choices)
    correction = torch.as_tensor(checkpoint["prior_correction"], dtype=torch.float32, device=device)
    if nonair_bias is None:
        nonair_bias = float(checkpoint.get("default_nonair_bias", 0.75))
    results = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.as_tensor(features[start:start + batch_size], device=device)
            logits = model(batch) + correction
            if logits.shape[1] > 1 and nonair_bias:
                logits[:, 1:] += float(nonair_bias)
            results.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(results).astype(np.int32, copy=False)


def generate_block_map(
    model: VoxelBlockNet,
    checkpoint: dict[str, Any],
    theme_id: int,
    choices: list[int],
    *,
    device: str = "cpu",
    nonair_bias: float | None = None,
) -> dict[tuple[int, int, int], str]:
    coords = all_grid_coordinates()
    block_ids = predict_block_ids(
        model, checkpoint, coords, theme_id, choices, device=device, nonair_bias=nonair_bias
    )
    block_names = checkpoint["block_names"]
    return {
        tuple(int(value) for value in coord): block_names[int(block_id)]
        for coord, block_id in zip(coords, block_ids)
        if int(block_id) != 0
    }
