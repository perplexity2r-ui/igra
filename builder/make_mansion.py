"""Use the trained policy to design and export a large mansion datapack."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from builder.export_plan import placement_commands, plan_from_choices
from builder.mansion import render_mansion_blocks
from builder.render_preview import render_isometric
from env.builder_env import BuilderEnv, DESIGN_CHOICES, THEMES, THEME_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/builder/builder_500000_episodes")
    parser.add_argument("--theme", choices=THEMES, default="desert")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--origin", nargs=3, type=int, default=(100, 64, 100), metavar=("X", "Y", "Z"))
    parser.add_argument("--out-dir", type=Path, default=Path("build-export/mansion"))
    args = parser.parse_args()

    model = PPO.load(args.model, device="cpu")
    theme_id = THEMES.index(args.theme)
    env = BuilderEnv()
    obs, _ = env.reset(seed=args.seed, options={"theme": theme_id})
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

    plan = plan_from_choices(
        theme_id, choices, rotation=0, lot_scale=2, season=season,
        reward=quality, checkpoint=str(args.model),
    )
    plan.update({
        "structure": "two-storey U-shaped mansion with courtyard, central fountain, gate, and twin turrets",
        "layout_note": "The policy selected the theme/style; the mansion renderer expands its courtyard-villa plan to estate scale.",
        "origin": list(args.origin),
    })
    blocks = render_mansion_blocks(plan)
    commands = placement_commands(blocks, tuple(args.origin))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plan_path = args.out_dir / "mansion-plan.json"
    plan_path.write_text(json.dumps({**plan, "block_count": len(blocks)}, indent=2) + "\n", encoding="utf-8")
    function_path = args.out_dir / "mansion.mcfunction"
    function_path.write_text("\n".join(commands) + "\n", encoding="utf-8")

    datapack_root = args.out_dir / "mansion-datapack"
    function_in_pack = datapack_root / "data" / "builder" / "functions" / "mansion.mcfunction"
    function_in_pack.parent.mkdir(parents=True, exist_ok=True)
    function_in_pack.write_text("\n".join(commands) + "\n", encoding="utf-8")
    pack_meta = {"pack": {"pack_format": 6, "description": "PPO Builder - Desert Courtyard Mansion"}}
    (datapack_root / "pack.mcmeta").write_text(json.dumps(pack_meta, indent=2) + "\n", encoding="utf-8")
    datapack_zip = args.out_dir / "mansion-datapack.zip"
    with zipfile.ZipFile(datapack_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(datapack_root / "pack.mcmeta", "pack.mcmeta")
        archive.write(function_in_pack, "data/builder/functions/mansion.mcfunction")

    preview_path = args.out_dir / "mansion.png"
    render_isometric(blocks, f"{THEME_NAMES[args.theme]} Mansion", quality, preview_path)
    print(json.dumps({
        "theme": plan["theme"],
        "choices": plan["design"],
        "heuristic_quality": round(quality, 6),
        "block_count": len(blocks),
        "command_count": len(commands),
        "function": str(function_path),
        "datapack_zip": str(datapack_zip),
        "preview": str(preview_path),
        "plan": str(plan_path),
        "origin": list(args.origin),
    }, indent=2))


if __name__ == "__main__":
    main()
