"""Expand a policy-selected courtyard-villa style into a large mansion plan."""
from __future__ import annotations

from builder.render import PALETTES


def render_mansion_blocks(plan: dict) -> dict[tuple[int, int, int], str]:
    """Create a two-storey U-shaped estate using the model's learned style choices."""
    choices = [int(value) for value in plan["choices"]]
    if len(choices) != 7 or any(not 0 <= value < 8 for value in choices):
        raise ValueError("mansion plan must contain seven option indices, each in 0..7")
    palette = PALETTES[choices[1]]
    roof_style, window_style, ornament, landscape = choices[2], choices[3], choices[5], choices[6]
    blocks: dict[tuple[int, int, int], str] = {}

    def put(x: int, y: int, z: int, block: str) -> None:
        blocks[(x, y, z)] = "minecraft:" + block

    def fill(cells: set[tuple[int, int]], y: int, block: str) -> None:
        for x, z in cells:
            put(x, y, z, block)

    # Three connected wings enclose a broad, open-air courtyard.
    rear_wing = {(x, z) for x in range(-10, 11) for z in range(-11, -2)}
    west_wing = {(x, z) for x in range(-14, -7) for z in range(-3, 8)}
    east_wing = {(x, z) for x in range(8, 15) for z in range(-3, 8)}
    footprint = rear_wing | west_wing | east_wing
    perimeter = {
        (x, z) for x, z in footprint
        if any((x + dx, z + dz) not in footprint for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    }
    exposed = {
        cell: [(dx, dz) for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))
               if (cell[0] + dx, cell[1] + dz) not in footprint]
        for cell in perimeter
    }

    # Foundation, ground floor, upper floor, and two-storey exterior shell.
    fill(footprint, 0, palette.foundation)
    fill(footprint, 1, palette.floor)
    fill(footprint, 5, palette.floor)
    for x, z in perimeter:
        corner = len(exposed[(x, z)]) >= 2
        for y in (2, 3, 4, 6, 7, 8):
            trim = corner or (ornament == 6 and y in (4, 8) and (x + 2 * z) % 5 == 0)
            put(x, y, z, palette.accent if trim else palette.wall)

    # Repeated paired windows on both the street and courtyard facades.
    for x, z in perimeter:
        if len(exposed[(x, z)]) != 1 or (3 * x + 2 * z) % (5 if window_style in (0, 2, 4, 6) else 6):
            continue
        for y in (3, 7):
            put(x, y, z, palette.glass)
            if window_style in (1, 3, 5):
                put(x, y - 1, z, palette.glass)

    # Main double entrance opens from the courtyard into the rear hall.
    door_z = -3
    door_xs = (-1, 0) if choices[4] == 6 else (0,)
    for x in door_xs:
        hinge = "right" if x < 0 else "left"
        put(x, 2, door_z, f"{palette.door}[facing=south,half=lower,hinge={hinge},open=false,powered=false]")
        put(x, 3, door_z, f"{palette.door}[facing=south,half=upper,hinge={hinge},open=false,powered=false]")

    # Grand portico and balcony above the double doors.
    for x in (-3, 2):
        for y in range(2, 6):
            put(x, y, -2, palette.accent)
    for x in range(-3, 3):
        put(x, 6, -2, palette.roof)
    for x in range(-3, 3):
        for z in (-2, -1):
            put(x, 5, z, palette.floor)
    for x in range(-3, 3):
        put(x, 6, -1, "oak_fence")

    # Roof terrace over the side wings and stepped crown over the main hall.
    fill(footprint, 9, palette.roof)
    for x, z in perimeter:
        put(x, 10, z, palette.accent if (x + z) % 3 == 0 else palette.roof)
    if roof_style == 0:  # policy-selected gable
        for x in range(-11, 12):
            rise = max(0, 5 - abs(x) * 5 // 11)
            for y in range(10, 11 + rise):
                for z in range(-12, -2):
                    put(x, y, z, palette.roof)
    elif roof_style == 5:  # shed/rising roof
        for z in range(-12, -1):
            height = 10 + min(4, (z + 12) * 4 // 11)
            for x in range(-11, 12):
                put(x, height, z, palette.roof)
    else:  # stepped crown, pagoda, or terrace selected by the policy
        layers = ((10, 0), (11, 1), (12, 2)) if roof_style in (4, 6, 7) else ((10, 0), (11, 2))
        for y, inset in layers:
            for x in range(-11 + inset, 12 - inset):
                for z in range(-12 + inset, -2 - inset):
                    put(x, y, z, palette.roof)
        for x in range(-5, 6):
            put(x, 13, -7, palette.accent if abs(x) in (4, 5) else palette.roof)

    # Two matching roof turrets anchor the long wings.
    for cx in (-11, 11):
        cz = 4
        turret = {(x, z) for x in range(cx - 2, cx + 3) for z in range(cz - 2, cz + 3)}
        turret_edge = {
            (x, z) for x, z in turret
            if x in (cx - 2, cx + 2) or z in (cz - 2, cz + 2)
        }
        for x, z in turret_edge:
            for y in range(10, 13):
                put(x, y, z, palette.accent if (x in (cx - 2, cx + 2) and z in (cz - 2, cz + 2))
                    or y == 12 else palette.wall)
        for x, z in ((cx, cz - 2), (cx, cz + 2), (cx - 2, cz), (cx + 2, cz)):
            put(x, 11, z, palette.glass)
        for y, inset in ((13, 0), (14, 1), (15, 2)):
            for x in range(cx - 2 + inset, cx + 3 - inset):
                for z in range(cz - 2 + inset, cz + 3 - inset):
                    put(x, y, z, palette.roof)
        put(cx, 16, cz, "lantern[hanging=false,waterlogged=false]")

    # Formal desert courtyard and a contained fountain.
    if landscape == 2:
        fx, fz = 0, 3
        for x in range(fx - 2, fx + 3):
            for z in range(fz - 2, fz + 3):
                edge = abs(x - fx) == 2 or abs(z - fz) == 2
                put(x, 0, z, palette.foundation)
                put(x, 1, z, palette.foundation if edge else "water")
    else:
        # The learned landscape choice still controls the courtyard treatment.
        for x in (-5, 5):
            for z in range(0, 7):
                put(x, 0, z, palette.foundation)
                put(x, 1, z, palette.leaves)
                if landscape in (1, 3, 6, 7):
                    put(x, 2, z, palette.leaves)

    # Symmetric clipped hedges, planted beds, and lamps around the courtyard.
    for x in (-6, 6):
        for z in range(-1, 7):
            put(x, 0, z, palette.foundation)
            put(x, 1, z, palette.leaves)
            put(x, 2, z, palette.leaves)
    for x in (-5, -4, 4, 5):
        for z in (0, 6):
            if (x, z) not in footprint:
                put(x, 0, z, palette.flower)
    for x in (-5, 5):
        for z in (0, 6):
            put(x, 0, z, palette.foundation)
            for y in range(1, 4):
                put(x, y, z, palette.accent)
            put(x, 4, z, "lantern[hanging=false,waterlogged=false]")

    # Gate, long processional drive, and paths that skirt the fountain.
    for x in (-7, 7):
        for y in range(0, 6):
            put(x, y, 8, palette.accent if y in (1, 2, 3, 4) else palette.foundation)
    for x in range(-7, 8):
        put(x, 6, 8, palette.roof)
    for z in range(9, 17):
        for x in range(-2, 3):
            put(x, 0, z, palette.floor)
    for z in range(-1, 7):
        for x in (-4, 4):
            put(x, 0, z, palette.floor)
    for z in (-1, 6):
        for x in range(-4, 5):
            put(x, 0, z, palette.floor)
    for z in (7, 8):
        for x in range(-2, 3):
            put(x, 0, z, palette.floor)

    # Formal estate trees beyond the gate line.
    for tx in (-11, 11):
        for y in range(0, 5):
            put(tx, y, 13, palette.accent)
        for y in range(2, 7):
            radius = 1 if y < 5 else 2
            for dx in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if abs(dx) + abs(dz) <= radius + 1:
                        put(tx + dx, y, 13 + dz, palette.leaves)

    return blocks
