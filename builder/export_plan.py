"""Export build plans as JSON or Minecraft Java 1.16.5 function files."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from builder.render import render_blocks
from env.builder_env import DESIGN_CHOICES, OPTION_NAMES, THEMES, THEME_NAMES, design_quality


def plan_from_choices(
    theme: int | str,
    choices: list[int],
    *,
    rotation: int = 0,
    lot_scale: int = 1,
    season: int | None = None,
    reward: float | None = None,
    checkpoint: str | None = None,
) -> dict[str, Any]:
    if isinstance(theme, str):
        if theme not in THEMES:
            raise ValueError(f"unknown theme {theme!r}; choose from {THEMES}")
        theme_id = THEMES.index(theme)
    else:
        theme_id = int(theme)
    if not 0 <= theme_id < len(THEMES):
        raise ValueError(f"theme must be between 0 and {len(THEMES) - 1}")
    if len(choices) != len(DESIGN_CHOICES):
        raise ValueError(f"expected {len(DESIGN_CHOICES)} design choices, got {len(choices)}")
    normalized = [int(value) for value in choices]
    for stage, choice in enumerate(normalized):
        if not 0 <= choice < DESIGN_CHOICES[stage]:
            raise ValueError(f"choice {choice} out of range at stage {stage}")
    if not 0 <= int(lot_scale) < 3:
        raise ValueError("lot_scale must be 0 (small), 1 (normal), or 2 (large)")
    if season is not None and not 0 <= int(season) < 4:
        raise ValueError("season must be between 0 and 3")
    return {
        "theme_id": theme_id,
        "theme": THEME_NAMES[THEMES[theme_id]],
        "choices": normalized,
        "design": {OPTION_NAMES[i][normalized[i]]: normalized[i] for i in range(len(normalized))},
        "rotation": int(rotation) % 4,
        "lot_scale": int(lot_scale),
        "season": season,
        "reward": float(reward) if reward is not None else design_quality(normalized, theme_id),
        "checkpoint": checkpoint,
    }


def placement_commands(
    blocks: dict[tuple[int, int, int], str], origin: tuple[int, int, int]
) -> list[str]:
    if len(origin) != 3:
        raise ValueError("origin must contain three integer coordinates")
    ox, oy, oz = (int(value) for value in origin)
    # Supports/floors are placed before upper structures. The first command clears
    # only this build's bounding box; callers should choose an empty build site.
    ordered = sorted(blocks.items(), key=lambda item: (item[0][1], item[0][2], item[0][0]))
    if not ordered:
        return []
    xs = [pos[0] for pos, _ in ordered]
    ys = [pos[1] for pos, _ in ordered]
    zs = [pos[2] for pos, _ in ordered]
    lo = (min(xs), min(ys), min(zs))
    hi = (max(xs), max(ys), max(zs))
    fill_volume = (hi[0] - lo[0] + 1) * (hi[1] - lo[1] + 1) * (hi[2] - lo[2] + 1)
    if fill_volume > 32_768:
        raise ValueError(f"Minecraft 1.16.5 fill limit exceeded: {fill_volume} blocks")
    commands = [
        f"fill {ox + lo[0]} {oy + lo[1]} {oz + lo[2]} "
        f"{ox + hi[0]} {oy + hi[1]} {oz + hi[2]} air"
    ]
    for (x, y, z), block in ordered:
        commands.append(f"setblock {ox + x} {oy + y} {oz + z} {block} replace")
    if len(commands) > 65_536:
        raise ValueError(f"Minecraft function command-chain limit exceeded: {len(commands)}")
    return commands


def export_plan(
    plan: dict[str, Any],
    out_dir: str | Path,
    origin: tuple[int, int, int],
    function_name: str = "generated_house",
) -> dict[str, Any]:
    if not re.fullmatch(r"[a-z0-9_/-]+", function_name) or function_name.startswith("/"):
        raise ValueError("function_name must use lowercase namespace-safe characters")
    if any(part in ("", ".", "..") for part in function_name.split("/")):
        raise ValueError("function_name cannot contain empty or traversal path segments")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    blocks = render_blocks(plan)
    commands = placement_commands(blocks, origin)
    (out / "plan.json").write_text(
        json.dumps({**plan, "block_count": len(blocks)}, indent=2) + "\n", encoding="utf-8"
    )
    function_path = out / f"{function_name}.mcfunction"
    function_path.parent.mkdir(parents=True, exist_ok=True)
    function_path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    return {
        "block_count": len(blocks),
        "command_count": len(commands),
        "fill_volume": _fill_volume(blocks),
        "function": str(function_path),
    }


def _fill_volume(blocks: dict[tuple[int, int, int], str]) -> int:
    if not blocks:
        return 0
    coords = list(blocks)
    return (
        (max(p[0] for p in coords) - min(p[0] for p in coords) + 1)
        * (max(p[1] for p in coords) - min(p[1] for p in coords) + 1)
        * (max(p[2] for p in coords) - min(p[2] for p in coords) + 1)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="JSON plan containing a seven-entry choices array")
    parser.add_argument("--out", default="build-export")
    parser.add_argument("--origin", nargs=3, type=int, default=(0, 64, 0), metavar=("X", "Y", "Z"))
    parser.add_argument("--function-name", default="generated_house")
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    result = export_plan(plan, args.out, tuple(args.origin), args.function_name)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
