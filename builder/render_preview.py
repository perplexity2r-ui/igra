"""Render the trained builder's actual block placements as isometric previews."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO

from builder.export_plan import plan_from_choices
from builder.render import render_blocks
from env.builder_env import BuilderEnv, DESIGN_CHOICES, THEMES, THEME_NAMES


TILE_W = 32
TILE_H = 16
BLOCK_H = 16


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _shade(rgb: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, round(channel * factor))) for channel in rgb)


def _block_color(block: str) -> tuple[int, int, int]:
    name = block.split(":", 1)[-1].split("[", 1)[0]
    if name == "air":
        return 0, 0, 0
    if "glass" in name:
        return 130, 202, 222
    if name.endswith("_door"):
        return 81, 52, 34
    if name == "water":
        return 55, 133, 205
    if "leaves" in name or name == "warped_wart_block":
        return (58, 126, 73) if "warped" not in name else (46, 123, 113)
    if name in {"dandelion", "sunflower"}:
        return 231, 195, 52
    if name in {"blue_orchid", "allium"}:
        return 96, 131, 202
    if name in {"poppy", "orange_tulip", "oxeye_daisy"}:
        return (213, 82, 61) if name == "poppy" else (224, 137, 67)
    if "sandstone" in name or name == "sand":
        return (199, 173, 120) if "red_" not in name else (176, 91, 57)
    if "blackstone" in name or "nether_brick" in name:
        return 73, 54, 58
    if "prismarine" in name or "warped" in name:
        return 58, 128, 119
    if "spruce" in name:
        return 102, 75, 49
    if "dark_oak" in name:
        return 75, 51, 34
    if "birch" in name:
        return 210, 193, 139
    if "acacia" in name:
        return 170, 91, 50
    if "oak" in name or "dandelion" in name:
        return 168, 126, 69
    if "cobble" in name or "stone" in name or name in {"iron_bars", "iron_block"}:
        return 126, 132, 135
    if "brick" in name:
        return 153, 77, 65
    if name in {"glowstone", "lantern", "campfire"}:
        return 220, 158, 72
    if name == "sugar_cane":
        return 116, 166, 83
    if name == "grass_block":
        return 104, 158, 83
    if name == "dead_bush":
        return 150, 118, 72
    if name == "water":
        return 55, 133, 205
    return 155, 144, 126


def render_isometric(
    blocks: dict[tuple[int, int, int], str],
    title: str,
    quality: float,
    output: str | Path,
    *,
    quality_label: str = "Heuristic quality",
) -> Image.Image:
    visible = {pos: block for pos, block in blocks.items() if block.split(":", 1)[-1].split("[", 1)[0] != "air"}
    if not visible:
        raise ValueError("cannot render an empty block plan")

    def project(x: float, y: float, z: float) -> tuple[float, float]:
        return ((x - z) * TILE_W / 2, (x + z) * TILE_H / 2 - y * BLOCK_H)

    xs = [pos[0] for pos in visible]
    ys = [pos[1] for pos in visible]
    zs = [pos[2] for pos in visible]
    ground = (min(xs) - 2, max(xs) + 3, min(zs) - 2, max(zs) + 3)
    gx0, gx1, gz0, gz1 = ground
    world_points: list[tuple[float, float]] = []
    for x, y, z in visible:
        for corner in ((x, y, z), (x + 1, y, z), (x, y + 1, z), (x, y, z + 1),
                       (x + 1, y + 1, z), (x + 1, y, z + 1), (x, y + 1, z + 1),
                       (x + 1, y + 1, z + 1)):
            world_points.append(project(*corner))
    for corner in ((gx0, -0.15, gz0), (gx1, -0.15, gz0),
                   (gx1, -0.15, gz1), (gx0, -0.15, gz1)):
        world_points.append(project(*corner))
    min_px = min(point[0] for point in world_points)
    max_px = max(point[0] for point in world_points)
    min_py = min(point[1] for point in world_points)
    max_py = max(point[1] for point in world_points)

    margin = 48
    header = 96
    footer = 48
    width = max(900, math.ceil(max_px - min_px + margin * 2))
    height = max(700, math.ceil(header + max_py - min_py + margin + footer))
    offset_x = (width - (max_px - min_px)) / 2 - min_px
    offset_y = header + margin - min_py

    def screen(point: tuple[float, float]) -> tuple[int, int]:
        return round(point[0] + offset_x), round(point[1] + offset_y)

    image = Image.new("RGB", (width, height), (220, 234, 244))
    draw = ImageDraw.Draw(image)
    draw.text((32, 22), f"PPO BUILDER  /  {title.upper()}", fill=(35, 47, 58), font=_font(28))
    draw.text((34, 60), f"{quality_label} {quality:.3f}  •  generated Minecraft block plan",
              fill=(68, 86, 99), font=_font(17))

    ground_poly = [screen(project(gx0, -0.15, gz0)), screen(project(gx1, -0.15, gz0)),
                   screen(project(gx1, -0.15, gz1)), screen(project(gx0, -0.15, gz1))]
    draw.polygon(ground_poly, fill=(127, 167, 103), outline=(76, 106, 68))

    for (x, y, z), block in sorted(visible.items(), key=lambda item: (item[0][0] + item[0][2], item[0][1], item[0][0])):
        color = _block_color(block)
        top = [screen(project(x, y + 1, z)), screen(project(x + 1, y + 1, z)),
               screen(project(x + 1, y + 1, z + 1)), screen(project(x, y + 1, z + 1))]
        right = [screen(project(x + 1, y, z)), top[1], top[2], screen(project(x + 1, y, z + 1))]
        left = [screen(project(x, y, z + 1)), top[3], top[2], screen(project(x + 1, y, z + 1))]
        outline = _shade(color, 0.48)
        draw.polygon(right, fill=_shade(color, 0.68), outline=outline)
        draw.polygon(left, fill=_shade(color, 0.82), outline=outline)
        draw.polygon(top, fill=_shade(color, 1.12), outline=outline)

    footer_text = "Isometric rendering of the model's block placements — not a live Minecraft screenshot."
    draw.text((32, height - 32), footer_text, fill=(77, 91, 102), font=_font(14))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)
    return image


def sample_choices(model: PPO, theme_id: int, seed: int) -> tuple[list[int], float]:
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
    env.close()
    return choices, quality


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="trained SB3 checkpoint path")
    parser.add_argument("--out-dir", type=Path, default=Path("build-previews"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rotation", type=int, choices=range(4), default=0)
    parser.add_argument("--lot-scale", type=int, choices=range(3), default=1)
    args = parser.parse_args()

    model = PPO.load(args.model, device="cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pages: list[Image.Image] = []
    for theme_id, theme_key in enumerate(THEMES):
        choices, quality = sample_choices(model, theme_id, args.seed)
        plan = plan_from_choices(
            theme_id, choices, rotation=args.rotation, lot_scale=args.lot_scale,
            reward=quality, checkpoint=str(args.model),
        )
        blocks = render_blocks(plan)
        image_path = args.out_dir / f"{theme_key}.png"
        page = render_isometric(blocks, THEME_NAMES[theme_key], quality, image_path)
        pages.append(page)
        (args.out_dir / f"{theme_key}-plan.json").write_text(
            json.dumps({**plan, "block_count": len(blocks)}, indent=2) + "\n", encoding="utf-8"
        )
        print(f"{theme_key}: quality={quality:.3f}, blocks={len(blocks)}, image={image_path}")

    pdf_path = args.out_dir / "builder-houses.pdf"
    pages[0].save(pdf_path, "PDF", save_all=True, append_images=pages[1:], resolution=120.0)
    print(f"Saved four-page visual preview: {pdf_path}")


if __name__ == "__main__":
    main()
