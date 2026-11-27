"""Generate a mansion by neural per-voxel prediction and export a datapack."""
from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from builder.export_plan import placement_commands, plan_from_choices
from builder.render_preview import render_isometric
from builder.voxel_model import generate_block_map, load_voxel_model
from env.builder_env import BuilderEnv, DESIGN_CHOICES, THEMES, THEME_NAMES


def _sample_policy_plan(model: PPO, theme_id: int, seed: int) -> tuple[list[int], float, int]:
    env = BuilderEnv()
    obs, _ = env.reset(seed=seed, options={"theme": theme_id})
    choices: list[int] = []
    quality = 0.0
    for _ in DESIGN_CHOICES:
        action, _ = model.predict(obs, deterministic=True)
        choice = int(np.asarray(action).item())
        choices.append(choice)
        obs, _, terminated, truncated, info = env.step(choice)
        if terminated or truncated:
            quality = float(info["quality"])
            break
    season = env.season
    env.close()
    if len(choices) != len(DESIGN_CHOICES):
        raise RuntimeError("policy returned an incomplete style plan")
    return choices, quality, season


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="models/builder/builder_500000_episodes")
    parser.add_argument("--voxel-model", default="models/builder/voxel_mansion_net.pt")
    parser.add_argument("--theme", choices=THEMES, default="desert")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--nonair-bias", type=float, default=None,
                        help="override the checkpoint's logit bias between air and block predictions")
    parser.add_argument("--origin", nargs=3, type=int, default=(100, 64, 100), metavar=("X", "Y", "Z"))
    parser.add_argument("--out-dir", type=Path, default=Path("build-export/neural-mansion"))
    args = parser.parse_args()

    policy = PPO.load(args.policy, device="cpu")
    theme_id = THEMES.index(args.theme)
    choices, design_score, season = _sample_policy_plan(policy, theme_id, args.seed)
    voxel_model, voxel_checkpoint = load_voxel_model(args.voxel_model, device="cpu")
    nonair_bias = (float(voxel_checkpoint.get("default_nonair_bias", 0.75))
                   if args.nonair_bias is None else args.nonair_bias)
    blocks = generate_block_map(
        voxel_model, voxel_checkpoint, theme_id, choices, nonair_bias=nonair_bias
    )
    if not blocks:
        raise RuntimeError("voxel model predicted an empty structure")

    commands = placement_commands(blocks, tuple(args.origin))
    block_histogram = Counter(block.split("[", 1)[0] for block in blocks.values())
    plan = plan_from_choices(
        theme_id, choices, rotation=0, lot_scale=2, season=season,
        reward=design_score, checkpoint=str(args.policy),
    )
    plan.update({
        "structure": "neural voxel mansion",
        "block_generator": str(args.voxel_model),
        "generation_method": "the voxel network predicts a block token independently for every grid coordinate; no mansion renderer is called at inference",
        "origin": list(args.origin),
        "block_count": len(blocks),
        "block_histogram": dict(sorted(block_histogram.items())),
    })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.out_dir / "mansion-plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    function_path = args.out_dir / "mansion.mcfunction"
    function_path.write_text("\n".join(commands) + "\n", encoding="utf-8")

    datapack_root = args.out_dir / "mansion-datapack"
    function_in_pack = datapack_root / "data" / "builder" / "functions" / "mansion.mcfunction"
    function_in_pack.parent.mkdir(parents=True, exist_ok=True)
    function_in_pack.write_text("\n".join(commands) + "\n", encoding="utf-8")
    pack_meta = {"pack": {"pack_format": 6, "description": "Neural Voxel Builder Mansion"}}
    (datapack_root / "pack.mcmeta").write_text(json.dumps(pack_meta, indent=2) + "\n", encoding="utf-8")
    datapack_zip = args.out_dir / "neural-mansion-datapack.zip"
    with zipfile.ZipFile(datapack_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(datapack_root / "pack.mcmeta", "pack.mcmeta")
        archive.write(function_in_pack, "data/builder/functions/mansion.mcfunction")

    preview_path = args.out_dir / "mansion.png"
    render_isometric(
        blocks, f"Neural {THEME_NAMES[args.theme]} Mansion", design_score, preview_path,
        quality_label="Policy style-plan score",
    )
    print(json.dumps({
        "theme": plan["theme"],
        "policy_choices": plan["design"],
        "policy_style_score_not_voxel_quality": round(design_score, 6),
        "voxel_model": args.voxel_model,
        "nonair_bias": nonair_bias,
        "predicted_blocks": len(blocks),
        "command_count": len(commands),
        "function": str(function_path),
        "datapack_zip": str(datapack_zip),
        "preview": str(preview_path),
        "plan": str(plan_path),
        "origin": list(args.origin),
        "block_histogram": plan["block_histogram"],
    }, indent=2))


if __name__ == "__main__":
    main()
