"""Render a learned seven-choice plan into Minecraft Java 1.16.5 blocks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable



@dataclass(frozen=True)
class Palette:
    foundation: str
    floor: str
    wall: str
    roof: str
    stairs: str
    accent: str
    glass: str
    door: str
    leaves: str
    flower: str


PALETTES = (
    Palette("cobblestone", "oak_planks", "oak_planks", "spruce_planks", "spruce_stairs",
            "oak_log", "glass_pane", "oak_door", "oak_leaves", "dandelion"),
    Palette("stone_bricks", "spruce_planks", "spruce_planks", "stone_bricks", "stone_brick_stairs",
            "spruce_log", "glass_pane", "spruce_door", "spruce_leaves", "blue_orchid"),
    Palette("sandstone", "cut_sandstone", "sandstone", "red_sandstone", "red_sandstone_stairs",
            "acacia_log", "yellow_stained_glass_pane", "acacia_door", "acacia_leaves", "dead_bush"),
    Palette("polished_blackstone_bricks", "dark_oak_planks", "dark_oak_planks", "nether_bricks",
            "nether_brick_stairs", "dark_oak_log", "red_stained_glass_pane", "dark_oak_door",
            "dark_oak_leaves", "poppy"),
    Palette("bricks", "birch_planks", "birch_planks", "bricks", "brick_stairs", "birch_log",
            "white_stained_glass_pane", "birch_door", "birch_leaves", "oxeye_daisy"),
    Palette("stone_bricks", "oak_planks", "stone_bricks", "stone_bricks", "stone_brick_stairs",
            "oak_log", "light_blue_stained_glass_pane", "oak_door", "oak_leaves", "allium"),
    Palette("dark_prismarine", "warped_planks", "warped_planks", "prismarine_bricks",
            "prismarine_brick_stairs", "warped_stem", "cyan_stained_glass_pane", "warped_door",
            "warped_wart_block", "azure_bluet"),
    Palette("red_sandstone", "acacia_planks", "orange_terracotta", "red_sandstone",
            "red_sandstone_stairs", "acacia_log", "orange_stained_glass_pane", "acacia_door",
            "acacia_leaves", "orange_tulip"),
)


def _footprint(choice: int) -> set[tuple[int, int]]:
    def rect(x0: int, x1: int, z0: int, z1: int) -> set[tuple[int, int]]:
        return {(x, z) for x in range(x0, x1 + 1) for z in range(z0, z1 + 1)}

    if choice == 0:  # cozy square
        return rect(-3, 3, -3, 3)
    if choice == 1:  # broad chalet
        return rect(-4, 4, -3, 3)
    if choice == 2:  # open courtyard ring
        return rect(-4, 4, -4, 4) - rect(-1, 1, -1, 1)
    if choice == 3:  # compact tower
        return rect(-3, 3, -3, 3)
    if choice == 4:  # L-shaped lodge
        return rect(-4, 0, -4, 4) | rect(-4, 4, 0, 4)
    if choice == 5:  # pavilion with a central hall
        return rect(-1, 1, -4, 4) | rect(-4, 4, -1, 1)
    if choice == 6:  # twin wings, connected by a neck
        return rect(-5, -2, -3, 3) | rect(2, 5, -3, 3) | rect(-1, 1, -1, 1)
    # modern studio
    return rect(-4, 4, -3, 3)


def _rotate(point: tuple[int, int], turns: int) -> tuple[int, int]:
    x, z = point
    for _ in range(turns % 4):
        x, z = -z, x
    return x, z


def _rotate_block_state(block: str, turns: int) -> str:
    if turns % 4 == 0 or "[" not in block:
        return block
    name, raw_state = block.split("[", 1)
    state = raw_state.rstrip("]")
    directions = ("north", "east", "south", "west")
    states = []
    for item in state.split(","):
        key, value = item.split("=", 1)
        if key == "facing" and value in directions:
            value = directions[(directions.index(value) + turns) % 4]
        states.append(f"{key}={value}")
    return f"{name}[{','.join(states)}]"


def render_blocks(plan: dict) -> dict[tuple[int, int, int], str]:

    """Return local (x,y,z)->block-name placements, including exterior landscaping."""
    choices = [int(v) for v in plan["choices"]]
    if len(choices) != 7 or any(not 0 <= choice < 8 for choice in choices):
        raise ValueError("plan must contain seven option indices, each in 0..7")
    footprint = _footprint(choices[0])
    palette = PALETTES[choices[1]]
    roof_style, window_style, entry_style, ornament, landscaping = choices[2:]
    blocks: dict[tuple[int, int, int], str] = {}

    def put(x: int, y: int, z: int, block: str) -> None:
        blocks[(int(x), int(y), int(z))] = "minecraft:" + block

    def fill(coords: Iterable[tuple[int, int]], y: int, block: str) -> None:
        for x, z in coords:
            put(x, y, z, block)

    min_x = min(x for x, _ in footprint)
    max_x = max(x for x, _ in footprint)
    min_z = min(z for _, z in footprint)
    max_z = max(z for _, z in footprint)
    center_x = round((min_x + max_x) / 2)
    floor_cells = set(footprint)
    # The courtyard's open center is grass, not an interior room floor.
    if choices[0] == 2:
        floor_cells -= {(x, z) for x in range(-1, 2) for z in range(-1, 2)}

    perimeter = {
        (x, z) for x, z in footprint
        if any((x + dx, z + dz) not in footprint for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    }
    # Foundation and raised interior floor.
    fill(footprint, 0, palette.foundation)
    fill(floor_cells, 1, palette.floor)

    # Three-block walls. Corner posts and top trim add readable structure.
    for x, z in perimeter:
        corner = sum((x + dx, z + dz) not in footprint
                     for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))) >= 2
        for y in range(2, 5):
            put(x, y, z, palette.accent if corner or y == 4 else palette.wall)

    front_cells = sorted((x, z) for x, z in perimeter if z == max_z)
    door_x = min((x for x, _ in front_cells), key=lambda x: abs(x - center_x)) if front_cells else center_x
    door_cell = (door_x, max_z)
    # Window positions along each exterior side. The selector makes a balanced rhythm.
    window_cells: set[tuple[int, int]] = set()
    sides = [
        [(x, max_z) for x in range(min_x + 1, max_x)],
        [(x, min_z) for x in range(min_x + 1, max_x)],
        [(max_x, z) for z in range(min_z + 1, max_z)],
        [(min_x, z) for z in range(min_z + 1, max_z)],
    ]
    for side_i, side in enumerate(sides):
        side = [pos for pos in side if pos in perimeter and pos != door_cell]
        for i, pos in enumerate(side):
            if window_style == 0:
                chosen = (i % 3 == 1)
            elif window_style == 1:
                chosen = (i % 4 == 1)
            elif window_style == 2:
                chosen = (i % 3 == 0)
            elif window_style == 3:
                chosen = (i % 2 == 0)
            elif window_style == 4:
                chosen = (i % 3 == 1)
            elif window_style == 5:
                chosen = (i % 4 in (1, 2))
            elif window_style == 6:
                chosen = (i % 3 == 0)
            else:
                chosen = (i % 4 == 2)
            if chosen:
                window_cells.add(pos)
    for x, z in window_cells:
        put(x, 3, z, palette.glass)
        if window_style in (1, 3, 5):
            put(x, 2, z, palette.glass)
        put(x, 4, z, palette.accent)

    # Door (lower + upper half) faces out of the front wall.
    put(door_x, 2, max_z, f"{palette.door}[facing=south,half=lower,hinge=left,open=false,powered=false]")
    put(door_x, 3, max_z, f"{palette.door}[facing=south,half=upper,hinge=left,open=false,powered=false]")

    # Roof: broad eaves with several recognizable silhouettes. Courtyard roofs
    # leave the central 3x3 open so the courtyard and fountain remain uncovered.
    roof_min_x, roof_max_x = min_x - 1, max_x + 1
    roof_min_z, roof_max_z = min_z - 1, max_z + 1
    def roof_put(x: int, y: int, z: int, block: str) -> None:
        if choices[0] == 2 and -1 <= x <= 1 and -1 <= z <= 1:
            return
        put(x, y, z, block)

    if roof_style == 0:  # pitched gable with a solid stepped cross-section
        half_span = max(abs(roof_min_x - center_x), abs(roof_max_x - center_x))
        for x in range(roof_min_x, roof_max_x + 1):
            peak_height = 5 + half_span - abs(x - center_x)
            for y in range(5, peak_height + 1):
                for z in range(roof_min_z, roof_max_z + 1):
                    roof_put(x, y, z, palette.roof)
    elif roof_style == 1:  # hip roof
        for level, inset in ((5, 0), (6, 1), (7, 2)):
            for x in range(roof_min_x + inset, roof_max_x - inset + 1):
                for z in range(roof_min_z + inset, roof_max_z - inset + 1):
                    roof_put(x, level, z, palette.roof)
    elif roof_style == 2:  # mansard with a lower eave band
        for level, inset in ((5, 0), (6, 0), (7, 1)):
            for x in range(roof_min_x + inset, roof_max_x - inset + 1):
                for z in range(roof_min_z + inset, roof_max_z - inset + 1):
                    roof_put(x, level, z, palette.roof)
    elif roof_style == 3:  # flat roof terrace and parapet
        for x in range(roof_min_x, roof_max_x + 1):
            for z in range(roof_min_z, roof_max_z + 1):
                roof_put(x, 5, z, palette.floor)
                if x in (roof_min_x, roof_max_x) or z in (roof_min_z, roof_max_z):
                    roof_put(x, 6, z, palette.accent)
    elif roof_style == 4:  # pagoda/eaves
        for level, inset in ((5, 0), (6, 1), (7, 0)):
            for x in range(roof_min_x + inset, roof_max_x - inset + 1):
                for z in range(roof_min_z + inset, roof_max_z - inset + 1):
                    roof_put(x, level, z, palette.stairs if level != 6 else palette.roof)
    elif roof_style == 5:  # shed roof rising toward the back
        for z in range(roof_min_z, roof_max_z + 1):
            level = 5 + min(2, (z - roof_min_z) * 3 // max(1, roof_max_z - roof_min_z + 1))
            for x in range(roof_min_x, roof_max_x + 1):
                roof_put(x, level, z, palette.roof)
    elif roof_style == 6:  # stepped crown
        for level, inset in ((5, 0), (6, 1), (7, 2), (8, 3)):
            for x in range(roof_min_x + inset, roof_max_x - inset + 1):
                for z in range(roof_min_z + inset, roof_max_z - inset + 1):
                    roof_put(x, level, z, palette.roof)
    else:  # spire
        for level, inset in ((5, 0), (6, 1), (7, 2), (8, 3), (9, 4)):
            for x in range(roof_min_x + inset, roof_max_x - inset + 1):
                for z in range(roof_min_z + inset, roof_max_z - inset + 1):
                    roof_put(x, level, z, palette.roof)
        spire_z = roof_min_z + 2 if choices[0] == 2 else 0
        put(center_x, 10, spire_z, "iron_bars")

    # Select simple additions tied to the model's ornament decision.
    back_z = min_z + 1
    left_x = min_x + 1
    right_x = max_x - 1
    if ornament == 0:  # chimney
        for y in range(4, 9):
            put(left_x, y, back_z, palette.foundation)
        put(left_x, 9, back_z, "campfire[lit=true,signal_fire=false,waterlogged=false]")
    elif ornament == 1:  # front dormer
        for x in range(center_x - 1, center_x + 2):
            put(x, 6, max_z - 1, palette.accent)
            put(x, 7, max_z - 1, palette.wall)
        put(center_x, 7, max_z - 1, palette.glass)
        for x in range(center_x - 1, center_x + 2):
            put(x, 8, max_z - 1, palette.stairs)
    elif ornament == 2:  # cupola
        cupola_z = min_z + 1 if choices[0] == 2 else 0
        for x in range(center_x - 1, center_x + 2):
            for z in range(cupola_z - 1, cupola_z + 2):
                put(x, 8, z, palette.accent if abs(x-center_x) == 1 or abs(z-cupola_z) == 1 else palette.glass)
        put(center_x, 9, cupola_z, palette.roof)
    elif ornament == 3:  # flower boxes under windows
        for x, z in window_cells:
            put(x, 2, z, "oak_trapdoor[facing=south,half=top,open=false,powered=false,waterlogged=false]")
            put(x, 2, z + (1 if z == max_z else -1 if z == min_z else 0), palette.flower)
    elif ornament == 4:  # corner turret
        tx, tz = right_x, back_z
        for y in range(5, 8):
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    put(tx + dx, y, tz + dz,
                        palette.glass if y == 6 and (dx == 0 or dz == 0) else palette.accent)
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                put(tx + dx, 8, tz + dz, palette.stairs)
    elif ornament == 5:  # warm lantern beams
        for x in (door_x - 2, door_x + 2):
            put(x, 5, max_z + 2, palette.accent)
            put(x, 6, max_z + 2, "lantern[hanging=false,waterlogged=false]")
        put(door_x, 6, max_z + 2, palette.accent)
    elif ornament == 6:  # carved trim/corner braces
        for x, z in perimeter:
            if (x + z) % 2 == 0:
                put(x, 4, z, palette.accent)
        for x in range(min_x, max_x + 1):
            put(x, 5, max_z, palette.accent)
    else:  # tiny lined basin outside the courtyard or beside the house
        fx, fz = max_x + 3, 0
        for x in range(fx - 1, fx + 2):
            for z in range(fz - 1, fz + 2):
                put(x, 0, z, "stone_bricks")
        for x in range(fx - 1, fx + 2):
            for z in range(fz - 1, fz + 2):
                if x != fx or z != fz:
                    put(x, 1, z, "stone_bricks")
        put(fx, 1, fz, "water")

    # Entrance variants. A two-block path leads toward the front in every plan.
    porch_z = max_z + 1
    if entry_style in (0, 2, 3, 4):
        width = 5 if entry_style == 2 else 3
        x0 = door_x - width // 2
        for x in range(x0, x0 + width):
            for z in range(max_z + 1, max_z + (4 if entry_style == 2 else 3)):
                put(x, 0, z, palette.foundation)
                put(x, 1, z, palette.floor)
        if entry_style in (0, 2, 3):
            post_positions = (x0, x0 + width - 1)
            post_z = max_z + 2
            for x in post_positions:
                for y in range(2, 5):
                    put(x, y, post_z, palette.accent)
            for x in range(x0, x0 + width):
                put(x, 5, post_z, palette.roof)
        if entry_style == 4:
            for z in range(max_z + 1, max_z + 4):
                put(door_x, 0, z, palette.stairs)
    elif entry_style == 1:  # stone arch
        for x in (door_x - 2, door_x + 2):
            for y in range(2, 6):
                put(x, y, max_z + 1, palette.foundation)
        for x in range(door_x - 2, door_x + 3):
            put(x, 5, max_z + 1, palette.foundation)
    elif entry_style == 6:  # paired doors
        for x in (door_x - 1, door_x + 1):
            put(x, 2, max_z, f"{palette.door}[facing=south,half=lower,hinge=left,open=false,powered=false]")
            put(x, 3, max_z, f"{palette.door}[facing=south,half=upper,hinge=left,open=false,powered=false]")
    else:  # upper balcony
        for x in range(door_x - 2, door_x + 3):
            put(x, 4, max_z + 1, palette.floor)
            put(x, 5, max_z + 1, "oak_fence")
        put(door_x, 5, max_z + 1, "air")

    # Low paths and landscaping leave open ground around the house.
    if landscaping == 0:  # cottage flower garden
        for x in range(min_x - 2, max_x + 3):
            if x in (min_x - 2, max_x + 2) or x % 2 == 0:
                put(x, 0, max_z + 5, palette.flower)
        for x in (min_x - 2, max_x + 2):
            put(x, 0, max_z + 4, "oak_leaves")
    elif landscaping == 1:  # alpine pines
        for tx, tz in ((min_x - 3, min_z), (max_x + 3, min_z), (min_x - 3, max_z + 2), (max_x + 3, max_z + 2)):
            _tree(put, tx, tz, "spruce_log", "spruce_leaves", height=5)
    elif landscaping == 2:  # courtyard fountain, moved outside non-courtyard homes
        fx, fz = (0, 0) if choices[0] == 2 else (max_x + 3, 0)
        for x in range(fx - 1, fx + 2):
            for z in range(fz - 1, fz + 2):
                put(x, 0, z, palette.foundation)
                if x != fx or z != fz:
                    put(x, 1, z, "stone_bricks")
        put(fx, 1, fz, "water")
    elif landscaping == 3:  # enchanted grove
        for x, z in ((min_x - 3, min_z), (max_x + 3, max_z), (min_x - 3, max_z + 3)):
            _tree(put, x, z, "warped_stem", "warped_wart_block", height=4)
        for x in range(door_x - 2, door_x + 3):
            put(x, 0, max_z + 4, "glowstone" if x == door_x else "mossy_cobblestone")
    elif landscaping == 4:  # stone path
        for z in range(max_z + 1, max_z + 8):
            put(door_x, 0, z, "stone_bricks")
            if z % 2 == 0:
                put(door_x - 1, 0, z, "mossy_stone_bricks")
                put(door_x + 1, 0, z, "mossy_stone_bricks")
    elif landscaping == 5:  # lined pond with sugar-cane banks
        px, pz = max_x + 3, max_z + 2
        for x in range(px - 2, px + 3):
            for z in range(pz - 2, pz + 3):
                put(x, 0, z, "stone_bricks")
        for x in range(px - 2, px + 3):
            for z in range(pz - 2, pz + 3):
                if abs(x - px) == 2 or abs(z - pz) == 2:
                    if (abs(x - px), abs(z - pz)) == (2, 0):
                        put(x, 0, z, "sand")
                        put(x, 1, z, "sugar_cane")
                    else:
                        put(x, 1, z, "stone_bricks")
                else:
                    put(x, 1, z, "water")
    elif landscaping == 6:  # orchard
        for tx in (min_x - 3, 0, max_x + 3):
            _tree(put, tx, max_z + 3, "oak_log", "oak_leaves", height=4)
    else:  # formal garden / mirrored hedges
        for x in range(min_x - 3, max_x + 4):
            for z in (max_z + 3, max_z + 5):
                put(x, 0, z, "stone_bricks" if x % 2 else palette.flower)
        for x in (min_x - 3, max_x + 3):
            for z in range(max_z + 3, max_z + 6):
                put(x, 0, z, "oak_leaves")

    # All builds sit on a modest path start, helping the entrance read from outside.
    for z in range(max_z + 1, max_z + 4):
        if (door_x, 0, z) not in blocks:
            put(door_x, 0, z, palette.foundation)

    # Apply deterministic site rotation and optional size variation after geometry.
    turns = int(plan.get("rotation", 0)) % 4
    scale = (0.8, 1.0, 1.2)[int(plan.get("lot_scale", 1)) % 3]
    transformed: dict[tuple[int, int, int], str] = {}
    for (x, y, z), block in blocks.items():
        sx, sz = round(x * scale), round(z * scale)
        rx, rz = _rotate((sx, sz), turns)
        transformed[(rx, y, rz)] = _rotate_block_state(block, turns)
    return transformed


def _tree(put, x: int, z: int, trunk: str, leaves: str, height: int = 4) -> None:
    for y in range(height):
        put(x, y, z, trunk)
    for y in range(max(1, height - 2), height + 1):
        radius = 1 if y != height else 2
        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if abs(dx) + abs(dz) <= radius + 1:
                    put(x + dx, y, z + dz, leaves)
