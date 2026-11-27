"""Record a 360-degree Multi-Camera 3D Minecraft PvP Showcase Video!

Runs the trained 50k Asymmetric Randomizer PPO policy (`models/pvp_ppo_latest.zip`)
and renders a 1920x1080 30fps MP4 (`videos/pvp_360_8cam_showcase.mp4`) featuring
8 bot-attached 360° cameras (4 cameras attached to LearnerBot at 0°/90°/180°/270°
that also smoothly orbit 360°, and 4 cameras attached to EnemyBot at 0°/90°/180°/270°
that also smoothly orbit 360°), complete with Minecraft 3D blocky characters,
armor tiers, weapons, shields, placed block barricades, cobwebs, lava/water,
particles, HP/absorption bars, and live combat event feed.
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO

from env.sim_env import BotState, SimulatedPvPEnv

FONT_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_MONO_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
FONT_REG_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def load_font(path: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


FONTS = {
    "title": load_font(FONT_BOLD_PATH, 24),
    "sub": load_font(FONT_REG_PATH, 16),
    "cam_title": load_font(FONT_BOLD_PATH, 15),
    "cam_meta": load_font(FONT_MONO_PATH, 13),
    "hud": load_font(FONT_MONO_PATH, 14),
    "banner": load_font(FONT_BOLD_PATH, 34),
    "tag": load_font(FONT_BOLD_PATH, 12),
}

# Precomputed unit cube vertices [-0.5, 0.5]^3 and 6 quad faces with outward normals
CUBE_VERTS = np.array(
    [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ],
    dtype=np.float64,
)

# Each face: (vertex_indices_tuple, normal_vec, shade_factor)
CUBE_FACES: list[tuple[tuple[int, int, int, int], np.ndarray, float]] = [
    ((4, 5, 6, 7), np.array([0.0, 0.0, 1.0]), 0.92),    # +Z front
    ((1, 0, 3, 2), np.array([0.0, 0.0, -1.0]), 0.72),   # -Z back
    ((5, 1, 2, 6), np.array([1.0, 0.0, 0.0]), 0.82),    # +X right
    ((0, 4, 7, 3), np.array([-1.0, 0.0, 0.0]), 0.78),   # -X left
    ((7, 6, 2, 3), np.array([0.0, 1.0, 0.0]), 1.05),    # +Y top
    ((0, 1, 5, 4), np.array([0.0, -1.0, 0.0]), 0.55),   # -Y bottom
]

ARMOR_COLORS: dict[str, tuple[int, int, int]] = {
    "leather": (176, 104, 58),
    "chain_gold": (232, 198, 72),
    "iron": (212, 218, 226),
    "diamond_mix": (68, 228, 224),
}

WEAPON_COLORS: dict[str, tuple[int, int, int]] = {
    "stick_kb": (168, 124, 68),
    "wooden_shovel": (156, 114, 62),
    "golden_hoe": (248, 208, 56),
    "stone_sword": (150, 154, 160),
    "iron_pickaxe": (215, 220, 228),
    "iron_sword": (225, 230, 238),
    "iron_axe": (210, 215, 225),
    "diamond_sword": (64, 238, 230),
    "trident": (82, 206, 188),
    "diamond_axe": (56, 228, 220),
    "netherite_sword_fire": (245, 108, 48),
    "netherite_axe_sharp": (118, 86, 128),
}

BLOCK_COLORS: dict[str, tuple[int, int, int]] = {
    "minecraft:cobblestone": (126, 128, 134),
    "minecraft:oak_planks": (182, 142, 86),
    "minecraft:obsidian": (46, 32, 68),
    "minecraft:hay_block": (214, 184, 58),
}


@dataclass
class HazardVisual:
    pos: np.ndarray
    kind: str  # 'cobweb', 'lava', 'water', 'pillar'
    ttl: float
    color: tuple[int, int, int] = (230, 230, 235)


@dataclass
class ProjectileVisual:
    start: np.ndarray
    end: np.ndarray
    progress: float
    kind: str


@dataclass
class StepSnapshot:
    round_idx: int
    step_idx: int
    time_sec: float
    l_pos: np.ndarray
    l_y: float
    l_yaw: float
    l_hp: float
    l_abs: float
    l_hurt: float
    l_shield: bool
    l_shield_broken: float
    l_fire: float
    l_cobweb: float
    l_buff: float
    l_swing: float
    l_weapon: str
    l_armor: str
    l_blocks: int
    l_gaps: int
    l_splashes: int
    l_totem_flash: float

    e_pos: np.ndarray
    e_y: float
    e_yaw: float
    e_hp: float
    e_abs: float
    e_hurt: float
    e_shield: bool
    e_shield_broken: float
    e_fire: float
    e_cobweb: float
    e_buff: float
    e_swing: float
    e_weapon: str
    e_armor: str
    e_blocks: int
    e_gaps: int
    e_splashes: int
    e_totem_flash: float

    obstacles: list[tuple[np.ndarray, str]]
    hazards: list[tuple[np.ndarray, str, tuple[int, int, int]]]
    projectiles: list[tuple[np.ndarray, str]]
    events: list[str]
    banner_text: str = ""
    banner_color: tuple[int, int, int] = (90, 240, 140)


def shade_rgb(rgb: tuple[int, int, int], factor: float, hurt: bool = False) -> tuple[int, int, int]:
    r, g, b = rgb
    if hurt:
        r = min(255, int(r * 0.45 + 245 * 0.55))
        g = int(g * 0.35)
        b = int(b * 0.35)
    return (
        max(0, min(255, int(r * factor))),
        max(0, min(255, int(g * factor))),
        max(0, min(255, int(b * factor))),
    )


class Camera3D:
    def __init__(self, cam_pos: np.ndarray, look_at: np.ndarray, width: int, height: int, fov_deg: float = 64.0):
        self.cam_pos = cam_pos
        self.width = width
        self.height = height
        fwd = look_at - cam_pos
        norm = float(np.linalg.norm(fwd))
        self.fwd = fwd / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])
        up_ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(self.fwd, up_ref)
        r_norm = float(np.linalg.norm(right))
        self.right = right / r_norm if r_norm > 1e-6 else np.array([1.0, 0.0, 0.0])
        self.up = np.cross(self.right, self.fwd)
        self.focal = (0.5 * width) / math.tan(math.radians(fov_deg) * 0.5)
        self.cx = width * 0.5
        self.cy = height * 0.5

    def world_to_cam(self, pts: np.ndarray) -> np.ndarray:
        rel = pts - self.cam_pos
        x = rel @ self.right
        y = rel @ self.up
        z = rel @ self.fwd
        return np.stack([x, y, z], axis=-1)

    def project_cam_pts(self, cam_pts: np.ndarray) -> np.ndarray:
        z = np.maximum(cam_pts[:, 2], 0.12)
        sx = self.cx + (cam_pts[:, 0] / z) * self.focal
        sy = self.cy - (cam_pts[:, 1] / z) * self.focal
        return np.stack([sx, sy], axis=-1)

    def project_point(self, pt: np.ndarray) -> tuple[float, float, float] | None:
        rel = pt - self.cam_pos
        z = float(np.dot(rel, self.fwd))
        if z <= 0.25:
            return None
        x = float(np.dot(rel, self.right))
        y = float(np.dot(rel, self.up))
        sx = self.cx + (x / z) * self.focal
        sy = self.cy - (y / z) * self.focal
        return sx, sy, z


def add_oriented_box(
    polys: list[tuple[float, list[tuple[float, float]], tuple[int, int, int], tuple[int, int, int] | None]],
    cam: Camera3D,
    center: np.ndarray,
    size: tuple[float, float, float],
    yaw: float,
    color: tuple[int, int, int],
    hurt: bool = False,
    pitch_local: float = 0.0,
    pivot_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    outline: tuple[int, int, int] | None = (22, 26, 34),
) -> None:
    verts = CUBE_VERTS * np.array(size, dtype=np.float64)
    if abs(pitch_local) > 1e-4:
        verts = verts + np.array(pivot_offset, dtype=np.float64)
        cp, sp = math.cos(pitch_local), math.sin(pitch_local)
        rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float64)
        verts = verts @ rot_x.T
        verts = verts - np.array(pivot_offset, dtype=np.float64)

    cy, sy = math.cos(yaw), math.sin(yaw)
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    world_verts = (verts @ rot_y.T) + center

    cam_verts = cam.world_to_cam(world_verts)
    if float(np.max(cam_verts[:, 2])) <= 0.22:
        return
    # Near-plane clip safety: skip boxes straddling right behind lens
    if float(np.min(cam_verts[:, 2])) <= 0.14:
        return

    scr_verts = cam.project_cam_pts(cam_verts)

    for idx_quad, local_normal, shade in CUBE_FACES:
        world_normal = rot_y @ local_normal
        face_center = np.mean(world_verts[list(idx_quad)], axis=0)
        to_cam = cam.cam_pos - face_center
        if float(np.dot(world_normal, to_cam)) <= 0.0:
            continue
        depth = float(np.mean(cam_verts[list(idx_quad), 2]))
        pts = [(float(scr_verts[i, 0]), float(scr_verts[i, 1])) for i in idx_quad]
        polys.append((depth, pts, shade_rgb(color, shade, hurt=hurt), outline))


def build_floor_tiles() -> list[tuple[np.ndarray, tuple[int, int, int]]]:
    tiles: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    step = 2.0
    coords = np.arange(-9.0, 9.01, step)
    for ix, x in enumerate(coords):
        for iz, z in enumerate(coords):
            q = np.array(
                [
                    [x - 1.0, 0.0, z - 1.0],
                    [x + 1.0, 0.0, z - 1.0],
                    [x + 1.0, 0.0, z + 1.0],
                    [x - 1.0, 0.0, z + 1.0],
                ],
                dtype=np.float64,
            )
            checker = (ix + iz) % 2 == 0
            dist_edge = max(abs(x), abs(z))
            if dist_edge >= 8.5:
                col = (58, 64, 78) if checker else (50, 56, 70)
            else:
                col = (74, 88, 76) if checker else (64, 78, 68)
            tiles.append((q, col))
    return tiles


FLOOR_TILES = build_floor_tiles()


def add_bot_mesh(
    polys: list[tuple[float, list[tuple[float, float]], tuple[int, int, int], tuple[int, int, int] | None]],
    cam: Camera3D,
    pos_xz: np.ndarray,
    y: float,
    yaw: float,
    is_learner: bool,
    armor: str,
    weapon: str,
    shield_up: bool,
    hurt: bool,
    swing_phase: float,
    walk_phase: float,
    on_fire: bool,
    in_cobweb: bool,
    buffed: bool,
) -> None:
    bx, bz = float(pos_xz[0]), float(pos_xz[1])
    base = np.array([bx, y, bz], dtype=np.float64)
    cy, sy = math.cos(yaw), math.sin(yaw)
    right = np.array([cy, 0.0, -sy], dtype=np.float64)
    fwd = np.array([sy, 0.0, cy], dtype=np.float64)

    skin_col = (56, 178, 255) if is_learner else (245, 86, 86)
    shirt_col = (34, 118, 212) if is_learner else (188, 44, 48)
    pants_col = (28, 72, 138) if is_learner else (112, 30, 36)
    armor_col = ARMOR_COLORS.get(armor)

    torso_col = armor_col if armor_col else shirt_col
    leg_col = armor_col if armor in ("iron", "diamond_mix", "chain_gold") else pants_col
    helm_col = armor_col if armor in ("chain_gold", "iron", "diamond_mix") else None

    leg_swing = math.sin(walk_phase) * 0.45
    arm_swing = -0.95 * math.sin(swing_phase * math.pi) if swing_phase > 0.0 else -leg_swing * 0.6

    # Left & Right Legs
    add_oriented_box(
        polys, cam,
        base + np.array([0.0, 0.38, 0.0]) - right * 0.14,
        (0.24, 0.75, 0.24), yaw, leg_col, hurt=hurt,
        pitch_local=leg_swing, pivot_offset=(0.0, -0.32, 0.0),
    )
    add_oriented_box(
        polys, cam,
        base + np.array([0.0, 0.38, 0.0]) + right * 0.14,
        (0.24, 0.75, 0.24), yaw, leg_col, hurt=hurt,
        pitch_local=-leg_swing, pivot_offset=(0.0, -0.32, 0.0),
    )

    # Torso / Chestplate
    add_oriented_box(
        polys, cam,
        base + np.array([0.0, 1.12, 0.0]),
        (0.52, 0.74, 0.28), yaw, torso_col, hurt=hurt,
    )

    # Head + Helmet shell
    add_oriented_box(
        polys, cam,
        base + np.array([0.0, 1.74, 0.0]),
        (0.48, 0.48, 0.48), yaw, skin_col, hurt=hurt,
    )
    # Eyes (facing +fwd)
    add_oriented_box(
        polys, cam,
        base + np.array([0.0, 1.76, 0.0]) + fwd * 0.245,
        (0.34, 0.10, 0.04), yaw, (245, 250, 255), hurt=False, outline=None,
    )
    if helm_col is not None:
        add_oriented_box(
            polys, cam,
            base + np.array([0.0, 1.84, 0.0]),
            (0.54, 0.32, 0.54), yaw, helm_col, hurt=hurt,
        )

    # Right Arm (Weapon hand)
    r_arm_center = base + np.array([0.0, 1.12, 0.0]) + right * 0.38 + fwd * (0.12 if swing_phase > 0 else 0.0)
    add_oriented_box(
        polys, cam, r_arm_center,
        (0.22, 0.70, 0.22), yaw, torso_col, hurt=hurt,
        pitch_local=-0.55 + arm_swing, pivot_offset=(0.0, -0.28, 0.0),
    )

    # Held Weapon blade/tool extending forward from right hand
    w_col = WEAPON_COLORS.get(weapon, (210, 215, 225))
    is_axe = "axe" in weapon
    w_pos = base + np.array([0.0, 1.05, 0.0]) + right * 0.42 + fwd * (0.52 + (0.22 if swing_phase > 0 else 0.0))
    add_oriented_box(
        polys, cam, w_pos,
        (0.10, 0.14 if not is_axe else 0.28, 0.72), yaw, w_col, hurt=False,
        pitch_local=0.35 + arm_swing * 0.6,
    )

    # Left Arm + Off-hand Shield
    l_arm_center = base + np.array([0.0, 1.12, 0.0]) - right * 0.38
    add_oriented_box(
        polys, cam, l_arm_center,
        (0.22, 0.70, 0.22), yaw, torso_col, hurt=hurt,
        pitch_local=-0.45 if shield_up else leg_swing * 0.5, pivot_offset=(0.0, -0.28, 0.0),
    )
    if shield_up:
        sh_pos = base + np.array([0.0, 1.16, 0.0]) + fwd * 0.44 - right * 0.16
        add_oriented_box(
            polys, cam, sh_pos,
            (0.62, 0.86, 0.10), yaw, (186, 142, 78), hurt=False, outline=(240, 215, 110),
        )

    # Fire / Cobweb / Buff visual effects on bot
    if on_fire:
        add_oriented_box(
            polys, cam, base + np.array([0.0, 0.95, 0.0]),
            (0.68, 1.35, 0.68), yaw + 0.4, (255, 118, 28), hurt=False, outline=(255, 210, 60),
        )
    if in_cobweb:
        add_oriented_box(
            polys, cam, base + np.array([0.0, 0.48, 0.0]),
            (0.86, 0.86, 0.86), 0.25, (236, 240, 246), hurt=False, outline=(180, 188, 198),
        )
    if buffed:
        add_oriented_box(
            polys, cam, base + np.array([0.0, 0.06, 0.0]),
            (0.95, 0.05, 0.95), yaw, (186, 84, 255), hurt=False, outline=None,
        )


def render_viewport(
    snap: StepSnapshot,
    anchor_bot: str,
    base_angle_deg: float,
    orbit_deg: float,
    vp_w: int,
    vp_h: int,
    cam_label: str,
) -> Image.Image:
    sky_bg = (22, 28, 42) if anchor_bot == "learner" else (34, 24, 34)
    img = Image.new("RGB", (vp_w, vp_h), sky_bg)
    draw = ImageDraw.Draw(img)

    # Subtle horizon band
    draw.rectangle([0, 0, vp_w, int(vp_h * 0.42)], fill=(18, 24, 38) if anchor_bot == "learner" else (28, 20, 32))

    if anchor_bot == "learner":
        a_pos, a_y, a_yaw = snap.l_pos, snap.l_y, snap.l_yaw
        o_pos, o_y = snap.e_pos, snap.e_y
        accent = (64, 190, 255)
    else:
        a_pos, a_y, a_yaw = snap.e_pos, snap.e_y, snap.e_yaw
        o_pos, o_y = snap.l_pos, snap.l_y
        accent = (255, 96, 96)

    # Camera rig attached to anchor bot at 0/90/180/270 deg + smooth 360° orbit!
    theta = a_yaw + math.radians(base_angle_deg + orbit_deg)
    radius = 4.9
    cam_x = float(a_pos[0]) - math.sin(theta) * radius
    cam_z = float(a_pos[1]) - math.cos(theta) * radius
    # Gently keep camera inside/above bedrock walls so view is never occluded by outer wall
    cam_x = max(-10.8, min(10.8, cam_x))
    cam_z = max(-10.8, min(10.8, cam_z))
    cam_pos = np.array([cam_x, a_y + 2.65, cam_z], dtype=np.float64)

    # Look slightly weighted between the attached bot (72%) and its opponent (28%) so the bot stays framed while showing incoming action
    look_xz = a_pos * 0.74 + o_pos * 0.26
    look_at = np.array([float(look_xz[0]), a_y * 0.75 + o_y * 0.25 + 1.15, float(look_xz[1])], dtype=np.float64)

    cam = Camera3D(cam_pos, look_at, vp_w, vp_h, fov_deg=66.0)

    # 1. Draw Arena Floor Checkerboard first
    floor_polys: list[tuple[float, list[tuple[float, float]], tuple[int, int, int]]] = []
    for quad_3d, col in FLOOR_TILES:
        c_pts = cam.world_to_cam(quad_3d)
        if float(np.min(c_pts[:, 2])) <= 0.18:
            continue
        s_pts = cam.project_cam_pts(c_pts)
        depth = float(np.mean(c_pts[:, 2]))
        pts = [(float(s_pts[i, 0]), float(s_pts[i, 1])) for i in range(4)]
        floor_polys.append((depth, pts, col))

    floor_polys.sort(key=lambda item: item[0], reverse=True)
    for _, pts, col in floor_polys:
        draw.polygon(pts, fill=col, outline=(42, 50, 58))

    # 2. Collect 3D scene boxes (bedrock perimeter walls, placed barricades, hazards, projectiles, both bots)
    polys: list[tuple[float, list[tuple[float, float]], tuple[int, int, int], tuple[int, int, int] | None]] = []

    # Low bedrock arena border curbs so orientation & walls are unmistakable
    for wall_center, wall_size in [
        (np.array([0.0, 0.55, -10.1]), (20.6, 1.1, 0.5)),
        (np.array([0.0, 0.55, 10.1]), (20.6, 1.1, 0.5)),
        (np.array([-10.1, 0.55, 0.0]), (0.5, 1.1, 20.6)),
        (np.array([10.1, 0.55, 0.0]), (0.5, 1.1, 20.6)),
    ]:
        add_oriented_box(polys, cam, wall_center, wall_size, 0.0, (44, 48, 58), outline=(28, 32, 40))

    # Corner pillars for clear 360° spatial reference as cameras rotate!
    for cx, cz, p_col in [
        (-9.6, -9.6, (80, 150, 235)),
        (9.6, -9.6, (235, 175, 65)),
        (9.6, 9.6, (225, 85, 95)),
        (-9.6, 9.6, (75, 210, 145)),
    ]:
        add_oriented_box(polys, cam, np.array([cx, 1.3, cz]), (0.7, 2.6, 0.7), 0.0, p_col)

    # Placed 2-block high tactical barricades
    for obs_pos, blk_name in snap.obstacles:
        b_col = BLOCK_COLORS.get(blk_name, (132, 136, 142))
        add_oriented_box(
            polys, cam,
            np.array([float(obs_pos[0]), 0.5, float(obs_pos[1])]),
            (1.0, 1.0, 1.0), 0.0, b_col,
        )
        add_oriented_box(
            polys, cam,
            np.array([float(obs_pos[0]), 1.5, float(obs_pos[1])]),
            (1.0, 1.0, 1.0), 0.0, b_col,
        )

    # Hazards & Pillar blocks (lava, water, cobwebs, pillar jump blocks)
    for h_pos, h_kind, h_col in snap.hazards:
        if h_kind == "lava":
            add_oriented_box(
                polys, cam, np.array([float(h_pos[0]), 0.14, float(h_pos[1])]),
                (1.15, 0.28, 1.15), 0.0, (255, 102, 20), outline=(255, 200, 50),
            )
        elif h_kind == "water":
            add_oriented_box(
                polys, cam, np.array([float(h_pos[0]), 0.12, float(h_pos[1])]),
                (1.35, 0.24, 1.35), 0.0, (48, 142, 255), outline=(130, 205, 255),
            )
        elif h_kind == "cobweb":
            add_oriented_box(
                polys, cam, np.array([float(h_pos[0]), 0.45, float(h_pos[1])]),
                (0.9, 0.9, 0.9), 0.3, (235, 238, 245), outline=(175, 182, 195),
            )
        elif h_kind == "pillar":
            add_oriented_box(
                polys, cam, np.array([float(h_pos[0]), 0.5, float(h_pos[1])]),
                (0.95, 1.0, 0.95), 0.0, h_col,
            )

    # Flying projectiles (snowballs, eggs, arrows, harming potions)
    for p_pos, p_kind in snap.projectiles:
        p_col = (245, 248, 255) if p_kind in ("snowball", "egg") else ((195, 65, 225) if "harm" in p_kind else (235, 200, 90))
        add_oriented_box(polys, cam, p_pos, (0.28, 0.28, 0.28), snap.time_sec * 5.0, p_col, outline=None)

    walk_phase = snap.time_sec * 10.5
    add_bot_mesh(
        polys, cam,
        snap.l_pos, snap.l_y, snap.l_yaw,
        is_learner=True,
        armor=snap.l_armor,
        weapon=snap.l_weapon,
        shield_up=snap.l_shield,
        hurt=snap.l_hurt > 0.0,
        swing_phase=snap.l_swing,
        walk_phase=walk_phase,
        on_fire=snap.l_fire > 0.0,
        in_cobweb=snap.l_cobweb > 0.0,
        buffed=snap.l_buff > 0.0,
    )
    add_bot_mesh(
        polys, cam,
        snap.e_pos, snap.e_y, snap.e_yaw,
        is_learner=False,
        armor=snap.e_armor,
        weapon=snap.e_weapon,
        shield_up=snap.e_shield,
        hurt=snap.e_hurt > 0.0,
        swing_phase=snap.e_swing,
        walk_phase=walk_phase + 1.4,
        on_fire=snap.e_fire > 0.0,
        in_cobweb=snap.e_cobweb > 0.0,
        buffed=snap.e_buff > 0.0,
    )

    # Painter's algorithm sort back-to-front
    polys.sort(key=lambda item: item[0], reverse=True)
    for _, pts, col, out_col in polys:
        draw.polygon(pts, fill=col, outline=out_col)

    # Floating Nameplates + HP bars above bots
    for b_pos, b_y, b_name, b_hp, b_abs, b_col in [
        (snap.l_pos, snap.l_y, "LearnerBot", snap.l_hp, snap.l_abs, (78, 198, 255)),
        (snap.e_pos, snap.e_y, "EnemyBot", snap.e_hp, snap.e_abs, (255, 102, 102)),
    ]:
        proj = cam.project_point(np.array([float(b_pos[0]), b_y + 2.38, float(b_pos[1])]))
        if proj is not None:
            sx, sy, z = proj
            if 10 <= sx <= vp_w - 10 and 18 <= sy <= vp_h - 10 and z < 26.0:
                bw = 68
                bx0 = int(sx - bw // 2)
                by0 = int(sy)
                draw.rectangle([bx0 - 2, by0 - 15, bx0 + bw + 2, by0 + 6], fill=(12, 15, 22))
                draw.text((bx0, by0 - 14), b_name, fill=b_col, font=FONTS["tag"])
                hp_frac = max(0.0, min(1.0, b_hp / 20.0))
                draw.rectangle([bx0, by0, bx0 + bw, by0 + 4], fill=(48, 52, 62))
                if hp_frac > 0:
                    draw.rectangle(
                        [bx0, by0, bx0 + int(bw * hp_frac), by0 + 4],
                        fill=(64, 228, 118) if b_name == "LearnerBot" else (245, 84, 84),
                    )
                if b_abs > 0.1:
                    abs_w = min(bw, int(bw * (b_abs / 10.0)))
                    draw.rectangle([bx0, by0 - 2, bx0 + abs_w, by0], fill=(250, 212, 64))

    # Camera bezel & 360° telemetry badge
    draw.rectangle([0, 0, vp_w - 1, vp_h - 1], outline=accent, width=2)
    draw.rectangle([4, 4, vp_w - 5, 28], fill=(12, 16, 26))
    total_az = int((base_angle_deg + orbit_deg) % 360)
    draw.text((10, 7), cam_label, fill=accent, font=FONTS["cam_title"])
    az_txt = f"360° Azimuth: {total_az:03d}°"
    draw.text((vp_w - 148, 8), az_txt, fill=(210, 220, 235), font=FONTS["cam_meta"])

    return img


def simulate_showcase_rounds(
    model: PPO, num_rounds: int = 3, base_seed: int = 2026
) -> list[StepSnapshot]:
    env = SimulatedPvPEnv(max_steps=260)
    snapshots: list[StepSnapshot] = []

    # Curate exciting asymmetric matchups so the viewer sees diverse weapons, blocks, traps & consumables
    candidate_seeds = [11, 29, 44, 58, 77, 93, 108, 125, 141]
    chosen_runs: list[list[StepSnapshot]] = []

    for seed in candidate_seeds:
        if len(chosen_runs) >= num_rounds:
            break
        obs, reset_info = env.reset(seed=base_seed + seed)
        round_idx = len(chosen_runs) + 1
        round_snaps: list[StepSnapshot] = []
        hazards: list[HazardVisual] = []
        projectiles: list[ProjectileVisual] = []
        recent_events: list[str] = [
            f"R{round_idx}: {env.learner.weapon.weapon_id} ({env.learner.armor.name}) vs {env.enemy.weapon.weapon_id} ({env.enemy.armor.name})"
        ]

        l_yaw = math.atan2(float(env.enemy.pos[0] - env.learner.pos[0]), float(env.enemy.pos[1] - env.learner.pos[1]))
        e_yaw = l_yaw + math.pi
        l_swing = 0.0
        e_swing = 0.0
        l_totem_flash = 0.0
        e_totem_flash = 0.0

        done = False
        used_blocks_or_traps = 0

        while not done:
            prev_l_cd = env.learner.cd
            prev_e_cd = env.enemy.cd
            prev_obs_count = len(env.obstacles)
            prev_l_pos = env.learner.pos.copy()
            prev_e_pos = env.enemy.pos.copy()

            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            L = env.learner
            E = env.enemy

            vec = E.pos - L.pos
            if float(np.linalg.norm(vec)) > 1e-4:
                l_yaw = math.atan2(float(vec[0]), float(vec[1]))
                e_yaw = math.atan2(float(-vec[0]), float(-vec[1]))

            if L.cd > prev_l_cd:
                l_swing = 1.0
            else:
                l_swing = max(0.0, l_swing - 0.34)

            if E.cd > prev_e_cd:
                e_swing = 1.0
            else:
                e_swing = max(0.0, e_swing - 0.34)

            used = info.get("used_consumable")
            if used:
                if used == "block_barricade":
                    used_blocks_or_traps += 1
                    recent_events.append(f"Learner placed {L.block_item.split(':')[1]} Barricade!")
                elif used == "pillar_jump":
                    used_blocks_or_traps += 1
                    hazards.append(HazardVisual(prev_l_pos.copy(), "pillar", 1.8, BLOCK_COLORS.get(L.block_item, (160, 140, 95))))
                    recent_events.append(f"Learner Pillar-Jumped ({L.block_item.split(':')[1]})!")
                elif used == "cobweb_trap":
                    used_blocks_or_traps += 1
                    hazards.append(HazardVisual(E.pos.copy(), "cobweb", 2.6))
                    recent_events.append("Learner trapped Enemy in Cobweb!")
                elif used == "lava_trap":
                    used_blocks_or_traps += 1
                    hazards.append(HazardVisual(E.pos.copy(), "lava", 1.6))
                    recent_events.append("Learner dumped Lava Bucket on Enemy!")
                elif used == "water_bucket":
                    used_blocks_or_traps += 1
                    hazards.append(HazardVisual(L.pos.copy(), "water", 1.0))
                    recent_events.append("Learner used Water Bucket push/clear!")
                elif used in ("snowball", "egg", "crossbow", "splash_harming"):
                    projectiles.append(ProjectileVisual(prev_l_pos.copy(), E.pos.copy(), 0.0, used))
                    recent_events.append(f"Learner fired {used}!")
                elif used == "ender_pearl":
                    recent_events.append("Learner blinked with Ender Pearl!")
                else:
                    recent_events.append(f"Learner used {used}!")

            if len(env.obstacles) > prev_obs_count and used != "block_barricade":
                recent_events.append(f"Enemy built {E.block_item.split(':')[1]} cover!")
            if info.get("shield_broken"):
                recent_events.append("AXE SHIELD BREAK! Enemy shield disabled 5s!")
            if info.get("blocked_hit"):
                recent_events.append("Learner blocked incoming attack!")
            if info.get("totem_popped_learner"):
                l_totem_flash = 1.0
                recent_events.append("Learner popped TOTEM OF UNDYING!")
            if info.get("totem_popped_enemy"):
                e_totem_flash = 1.0
                recent_events.append("Enemy popped TOTEM OF UNDYING!")

            recent_events = recent_events[-5:]

            # Update hazard & projectile visuals
            hazards = [
                HazardVisual(h.pos, h.kind, h.ttl - env.DT, h.color)
                for h in hazards
                if h.ttl - env.DT > 0.0
            ]
            next_proj: list[ProjectileVisual] = []
            for p in projectiles:
                np_prog = p.progress + 0.45
                if np_prog < 1.0:
                    next_proj.append(ProjectileVisual(p.start, p.end, np_prog, p.kind))
            projectiles = next_proj

            obs_vis = [(op.copy(), L.block_item) for (op, _) in env.obstacles]
            haz_vis = [(h.pos.copy(), h.kind, h.color) for h in hazards]
            proj_vis = [
                (
                    np.array(
                        [
                            float(p.start[0] * (1 - p.progress) + p.end[0] * p.progress),
                            1.25 + 0.4 * math.sin(p.progress * math.pi),
                            float(p.start[1] * (1 - p.progress) + p.end[1] * p.progress),
                        ]
                    ),
                    p.kind,
                )
                for p in projectiles
            ]

            banner = ""
            b_col = (90, 240, 140)
            if done:
                if info.get("enemy_dead") and not info.get("learner_dead"):
                    banner = f"ROUND {round_idx}: LEARNERBOT WINS!"
                    b_col = (72, 238, 138)
                elif info.get("learner_dead"):
                    banner = f"ROUND {round_idx}: ENEMYBOT WINS!"
                    b_col = (255, 96, 96)
                else:
                    banner = f"ROUND {round_idx}: TIME LIMIT DRAW"
                    b_col = (240, 200, 80)

            snap = StepSnapshot(
                round_idx=round_idx,
                step_idx=env.step_count,
                time_sec=env.step_count * env.DT,
                l_pos=L.pos.copy(),
                l_y=L.y,
                l_yaw=l_yaw,
                l_hp=L.hp,
                l_abs=L.absorption,
                l_hurt=L.hurt_cd,
                l_shield=bool(L.has_shield and L.shield_up and L.shield_disabled <= 0.0),
                l_shield_broken=L.shield_disabled,
                l_fire=L.fire_timer + L.lava_timer,
                l_cobweb=L.cobweb_timer,
                l_buff=L.buff_timer,
                l_swing=l_swing,
                l_weapon=L.weapon.weapon_id,
                l_armor=L.armor.name,
                l_blocks=L.blocks,
                l_gaps=L.gaps,
                l_splashes=L.splashes,
                l_totem_flash=l_totem_flash,
                e_pos=E.pos.copy(),
                e_y=E.y,
                e_yaw=e_yaw,
                e_hp=E.hp,
                e_abs=E.absorption,
                e_hurt=E.hurt_cd,
                e_shield=bool(E.has_shield and E.shield_up and E.shield_disabled <= 0.0),
                e_shield_broken=E.shield_disabled,
                e_fire=E.fire_timer + E.lava_timer,
                e_cobweb=E.cobweb_timer,
                e_buff=E.buff_timer,
                e_swing=e_swing,
                e_weapon=E.weapon.weapon_id,
                e_armor=E.armor.name,
                e_blocks=E.blocks,
                e_gaps=E.gaps,
                e_splashes=E.splashes,
                e_totem_flash=e_totem_flash,
                obstacles=obs_vis,
                hazards=haz_vis,
                projectiles=proj_vis,
                events=list(recent_events),
                banner_text=banner,
                banner_color=b_col,
            )
            round_snaps.append(snap)
            l_totem_flash = max(0.0, l_totem_flash - 0.25)
            e_totem_flash = max(0.0, e_totem_flash - 0.25)

        won = bool(round_snaps[-1].banner_text.endswith("LEARNERBOT WINS!"))
        if won and 22 <= len(round_snaps) <= 135 and used_blocks_or_traps >= 2:
            # Hold victory pose for 8 steps so viewer can read final state & watch 360° sweep
            final_snap = round_snaps[-1]
            for extra in range(8):
                held = StepSnapshot(**{**final_snap.__dict__, "time_sec": final_snap.time_sec + (extra + 1) * 0.1})
                round_snaps.append(held)
            chosen_runs.append(round_snaps)

    # Fallback if fewer than num_rounds matched strict filter
    if not chosen_runs:
        raise RuntimeError("Could not record showcase rounds")

    for r_snaps in chosen_runs:
        snapshots.extend(r_snaps)
    return snapshots


def interpolate_snapshots(s0: StepSnapshot, s1: StepSnapshot, alpha: float) -> StepSnapshot:
    if alpha <= 0.01 or s0.round_idx != s1.round_idx:
        return s0
    if alpha >= 0.99:
        return s1

    # Shortest-angle interpolation for yaws
    def lerp_angle(a: float, b: float, t: float) -> float:
        diff = (b - a + math.pi) % (2.0 * math.pi) - math.pi
        return a + diff * t

    return StepSnapshot(
        round_idx=s0.round_idx,
        step_idx=s0.step_idx,
        time_sec=s0.time_sec * (1.0 - alpha) + s1.time_sec * alpha,
        l_pos=s0.l_pos * (1.0 - alpha) + s1.l_pos * alpha,
        l_y=s0.l_y * (1.0 - alpha) + s1.l_y * alpha,
        l_yaw=lerp_angle(s0.l_yaw, s1.l_yaw, alpha),
        l_hp=s0.l_hp * (1.0 - alpha) + s1.l_hp * alpha,
        l_abs=s0.l_abs,
        l_hurt=s0.l_hurt,
        l_shield=s0.l_shield,
        l_shield_broken=s0.l_shield_broken,
        l_fire=s0.l_fire,
        l_cobweb=s0.l_cobweb,
        l_buff=s0.l_buff,
        l_swing=s0.l_swing * (1.0 - alpha) + s1.l_swing * alpha,
        l_weapon=s0.l_weapon,
        l_armor=s0.l_armor,
        l_blocks=s0.l_blocks,
        l_gaps=s0.l_gaps,
        l_splashes=s0.l_splashes,
        l_totem_flash=s0.l_totem_flash,
        e_pos=s0.e_pos * (1.0 - alpha) + s1.e_pos * alpha,
        e_y=s0.e_y * (1.0 - alpha) + s1.e_y * alpha,
        e_yaw=lerp_angle(s0.e_yaw, s1.e_yaw, alpha),
        e_hp=s0.e_hp * (1.0 - alpha) + s1.e_hp * alpha,
        e_abs=s0.e_abs,
        e_hurt=s0.e_hurt,
        e_shield=s0.e_shield,
        e_shield_broken=s0.e_shield_broken,
        e_fire=s0.e_fire,
        e_cobweb=s0.e_cobweb,
        e_buff=s0.e_buff,
        e_swing=s0.e_swing * (1.0 - alpha) + s1.e_swing * alpha,
        e_weapon=s0.e_weapon,
        e_armor=s0.e_armor,
        e_blocks=s0.e_blocks,
        e_gaps=s0.e_gaps,
        e_splashes=s0.e_splashes,
        e_totem_flash=s0.e_totem_flash,
        obstacles=s0.obstacles,
        hazards=s0.hazards,
        projectiles=s0.projectiles,
        events=s0.events,
        banner_text=s0.banner_text,
        banner_color=s0.banner_color,
    )


def render_composite_frame(snap: StepSnapshot, global_frame_idx: int, fps: int) -> np.ndarray:
    W, H = 1920, 1080
    canvas = Image.new("RGB", (W, H), (11, 14, 22))
    draw = ImageDraw.Draw(canvas)

    # Top Header Bar
    draw.rectangle([0, 0, W, 62], fill=(16, 21, 34))
    draw.line([(0, 62), (W, 62)], fill=(48, 64, 96), width=2)
    draw.text(
        (20, 10),
        f"MINECRAFT 1.16.5 PvP AI — 360° MULTI-CAM RIG (ROUND {snap.round_idx})",
        fill=(240, 246, 255),
        font=FONTS["title"],
    )
    draw.text(
        (20, 38),
        "Row 1: 4x 360° Cameras Attached to LearnerBot (Blue)   |   Row 2: 4x 360° Cameras Attached to EnemyBot (Red)",
        fill=(165, 185, 215),
        font=FONTS["sub"],
    )

    dist = float(np.linalg.norm(snap.e_pos - snap.l_pos))
    draw.text(
        (W - 430, 18),
        f"Step: {snap.step_idx:03d}  |  Dist: {dist:4.1f}m  |  Obstacles: {len(snap.obstacles)}",
        fill=(215, 230, 250),
        font=FONTS["hud"],
    )

    # Continuous 360° camera rotation (36 deg/sec -> full 360° sweep every 10s, plus 4 orthogonal rigs at 0/90/180/270!)
    orbit_deg = (global_frame_idx / float(fps)) * 36.0

    vp_w = 468
    vp_h = 412
    pad_x = 12
    start_x = 12

    learner_cams = [
        (0.0, "L-CAM 1 [FRONT CHASE + 360°]"),
        (90.0, "L-CAM 2 [RIGHT FLANK + 360°]"),
        (180.0, "L-CAM 3 [FACE / REVERSE + 360°]"),
        (270.0, "L-CAM 4 [LEFT FLANK + 360°]"),
    ]
    enemy_cams = [
        (0.0, "E-CAM 1 [FRONT CHASE + 360°]"),
        (90.0, "E-CAM 2 [RIGHT FLANK + 360°]"),
        (180.0, "E-CAM 3 [FACE / REVERSE + 360°]"),
        (270.0, "E-CAM 4 [LEFT FLANK + 360°]"),
    ]

    # Row 1: 4 cameras attached to LearnerBot
    y_row1 = 72
    for idx, (base_deg, label) in enumerate(learner_cams):
        vp = render_viewport(snap, "learner", base_deg, orbit_deg, vp_w, vp_h, label)
        x = start_x + idx * (vp_w + pad_x)
        canvas.paste(vp, (x, y_row1))

    # Row 2: 4 cameras attached to EnemyBot
    y_row2 = y_row1 + vp_h + 10
    for idx, (base_deg, label) in enumerate(enemy_cams):
        vp = render_viewport(snap, "enemy", base_deg, orbit_deg, vp_w, vp_h, label)
        x = start_x + idx * (vp_w + pad_x)
        canvas.paste(vp, (x, y_row2))

    # Bottom Telemetry, Asymmetric Loadouts, Radar Minimap & Combat Feed (y = 912..1080)
    hud_y = 914
    draw.rectangle([12, hud_y, W - 12, H - 10], fill=(16, 20, 32), outline=(45, 58, 86), width=2)

    # LearnerBot Panel (left)
    draw.text((28, hud_y + 10), "LEARNERBOT (PPO 50k Policy)", fill=(78, 198, 255), font=FONTS["cam_title"])
    draw.text(
        (28, hud_y + 34),
        f"Weapon : {snap.l_weapon:<22} | Armor: {snap.l_armor}",
        fill=(228, 236, 248),
        font=FONTS["hud"],
    )
    draw.text(
        (28, hud_y + 56),
        f"HP     : {snap.l_hp:4.1f}/20 (+{snap.l_abs:3.1f} Abs)  | Blocks: {snap.l_blocks:2d} | GApple: {snap.l_gaps} | Heals: {snap.l_splashes}",
        fill=(180, 240, 200),
        font=FONTS["hud"],
    )
    l_states = []
    if snap.l_shield:
        l_states.append("SHIELD UP")
    if snap.l_buff > 0:
        l_states.append("STR+SPD II")
    if snap.l_fire > 0:
        l_states.append("BURNING")
    if snap.l_cobweb > 0:
        l_states.append("IN COBWEB")
    draw.text(
        (28, hud_y + 78),
        f"Status : {', '.join(l_states) if l_states else 'Combat Ready'}",
        fill=(145, 195, 255),
        font=FONTS["hud"],
    )
    # Learner HP bar
    draw.rectangle([28, hud_y + 106, 560, hud_y + 124], fill=(38, 44, 58), outline=(70, 82, 106))
    l_fill = int(532 * max(0.0, min(1.0, snap.l_hp / 20.0)))
    if l_fill > 0:
        draw.rectangle([28, hud_y + 106, 28 + l_fill, hud_y + 124], fill=(56, 210, 255))

    # EnemyBot Panel (middle)
    ex0 = 615
    draw.text((ex0, hud_y + 10), "ENEMYBOT (Asymmetric Loadout)", fill=(255, 102, 102), font=FONTS["cam_title"])
    draw.text(
        (ex0, hud_y + 34),
        f"Weapon : {snap.e_weapon:<22} | Armor: {snap.e_armor}",
        fill=(228, 236, 248),
        font=FONTS["hud"],
    )
    draw.text(
        (ex0, hud_y + 56),
        f"HP     : {snap.e_hp:4.1f}/20 (+{snap.e_abs:3.1f} Abs)  | Blocks: {snap.e_blocks:2d} | GApple: {snap.e_gaps} | Heals: {snap.e_splashes}",
        fill=(255, 195, 195),
        font=FONTS["hud"],
    )
    e_states = []
    if snap.e_shield:
        e_states.append("SHIELD UP")
    if snap.e_shield_broken > 0:
        e_states.append(f"SHIELD BROKEN ({snap.e_shield_broken:.1f}s)")
    if snap.e_fire > 0:
        e_states.append("BURNING")
    if snap.e_cobweb > 0:
        e_states.append("TRAPPED IN COBWEB")
    draw.text(
        (ex0, hud_y + 78),
        f"Status : {', '.join(e_states) if e_states else 'Aggressive AI'}",
        fill=(255, 150, 150),
        font=FONTS["hud"],
    )
    draw.rectangle([ex0, hud_y + 106, ex0 + 532, hud_y + 124], fill=(38, 44, 58), outline=(70, 82, 106))
    e_fill = int(532 * max(0.0, min(1.0, snap.e_hp / 20.0)))
    if e_fill > 0:
        draw.rectangle([ex0, hud_y + 106, ex0 + e_fill, hud_y + 124], fill=(245, 82, 88))

    # Top-Down 360° Camera & Arena Radar Minimap
    map_x, map_y, map_s = 1190, hud_y + 12, 132
    draw.rectangle([map_x, map_y, map_x + map_s, map_y + map_s], fill=(24, 30, 42), outline=(82, 100, 135), width=2)
    draw.text((map_x + 8, map_y + 4), "360° RADAR", fill=(190, 208, 235), font=FONTS["tag"])

    def to_map(xz: np.ndarray) -> tuple[int, int]:
        mx = int(map_x + (float(xz[0]) + 10.0) / 20.0 * map_s)
        my = int(map_y + (float(xz[1]) + 10.0) / 20.0 * map_s)
        return mx, my

    for obs_pos, _ in snap.obstacles:
        ox, oy = to_map(obs_pos)
        draw.rectangle([ox - 3, oy - 3, ox + 3, oy + 3], fill=(195, 165, 95))

    lx, ly = to_map(snap.l_pos)
    ex, ey = to_map(snap.e_pos)
    # Draw 4 rotating 360° camera rays around each bot on radar!
    for deg in (0.0, 90.0, 180.0, 270.0):
        th_l = snap.l_yaw + math.radians(deg + orbit_deg)
        draw.line([(lx, ly), (int(lx - math.sin(th_l) * 18), int(ly - math.cos(th_l) * 18))], fill=(45, 125, 185), width=1)
        th_e = snap.e_yaw + math.radians(deg + orbit_deg)
        draw.line([(ex, ey), (int(ex - math.sin(th_e) * 18), int(ey - math.cos(th_e) * 18))], fill=(185, 65, 75), width=1)

    draw.ellipse([lx - 5, ly - 5, lx + 5, ly + 5], fill=(64, 198, 255), outline=(255, 255, 255))
    draw.ellipse([ex - 5, ey - 5, ex + 5, ey + 5], fill=(255, 86, 86), outline=(255, 255, 255))

    # Live Action Log (right)
    log_x = 1345
    draw.text((log_x, hud_y + 10), "LIVE COMBAT & BLOCK FEED", fill=(250, 214, 92), font=FONTS["cam_title"])
    for idx, ev in enumerate(snap.events[-5:]):
        draw.text((log_x, hud_y + 34 + idx * 22), f"• {ev[:52]}", fill=(215, 225, 240), font=FONTS["cam_meta"])

    # Round Result Banner Overlay
    if snap.banner_text:
        bw, bh = 680, 64
        bx0 = (W - bw) // 2
        by0 = 462
        draw.rectangle([bx0, by0, bx0 + bw, by0 + bh], fill=(12, 16, 28), outline=snap.banner_color, width=3)
        draw.text((bx0 + 48, by0 + 12), snap.banner_text, fill=snap.banner_color, font=FONTS["banner"])

    return np.asarray(canvas, dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render 8-Camera 360° PvP Showcase Video")
    parser.add_argument("--model", default="models/pvp_ppo_latest.zip", help="Trained PPO model path")
    parser.add_argument("--output", default="videos/pvp_360_8cam_showcase.mp4", help="Output MP4 path")
    parser.add_argument("--rounds", type=int, default=3, help="Number of PvP rounds to record")
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate (default: 30)")
    parser.add_argument("--substeps", type=int, default=3, help="Interpolated frames per 100ms env step")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"Loading trained PPO model from {args.model}...", flush=True)
    model = PPO.load(args.model, device="cpu")

    print(f"Simulating {args.rounds} asymmetric randomizer PvP showcase rounds...", flush=True)
    snapshots = simulate_showcase_rounds(model, num_rounds=args.rounds)
    total_frames = len(snapshots) * args.substeps
    print(
        f"Captured {len(snapshots)} PvP steps -> Rendering {total_frames} HD frames (1920x1080 @ {args.fps}fps)...",
        flush=True,
    )

    writer = imageio.get_writer(
        str(out_path),
        fps=args.fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=16,
    )
    preview_saved = False
    frame_idx = 0
    try:
        for i in range(len(snapshots)):
            s0 = snapshots[i]
            s1 = snapshots[min(i + 1, len(snapshots) - 1)]
            for sub in range(args.substeps):
                alpha = sub / float(args.substeps)
                interp = interpolate_snapshots(s0, s1, alpha)
                frame = render_composite_frame(interp, frame_idx, args.fps)
                writer.append_data(frame)
                if not preview_saved and i == min(28, len(snapshots) // 2) and sub == 0:
                    preview_path = out_path.with_suffix(".png")
                    Image.fromarray(frame).save(preview_path)
                    preview_saved = True
                frame_idx += 1
                if frame_idx % 90 == 0 or frame_idx == total_frames:
                    print(
                        f"  Rendered {frame_idx}/{total_frames} frames ({frame_idx / args.fps:.1f}s video)...",
                        flush=True,
                    )
    finally:
        writer.close()

    dt = time.time() - t0
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(
        f"Done in {dt:.1f}s! Saved {out_path} ({size_mb:.2f} MB, {total_frames / args.fps:.1f}s @ {args.fps}fps)",
        flush=True,
    )


if __name__ == "__main__":
    main()
