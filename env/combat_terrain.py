"""Compact, vector-friendly terrain layouts for the simulated PvP arena."""
from __future__ import annotations

import numpy as np

TERRAIN_SENSOR_COUNT = 10
TERRAIN_FEATURES = (
    "forward_clearance", "left_clearance", "right_clearance", "back_clearance",
    "forward_support", "left_support", "right_support", "back_support",
    "target_height_delta", "forward_slope",
)
TERRAIN_NAMES = {0: "flat", 1: "walls", 2: "pits", 3: "hills", 4: "mixed"}

_MAX_WALL_POINTS = 12
_MAX_HOLES = 3
_MAX_HILLS = 2
_WALL_RADIUS = 0.72
_SENSOR_DISTANCES = (0.75, 1.5, 2.25, 3.0)


class CombatTerrain:
    """Randomized flat, wall, pit, hill and mixed layouts for each fight.

    Walls are short chains of collision points, pits are unsupported circular patches,
    and hills are smooth height fields. Terrain remains fixed for a whole fight and is
    regenerated only when that vectorized arena row resets.
    """

    def __init__(self, n: int, rng: np.random.Generator, arena_bound: float = 9.2):
        self.n = int(n)
        self.rng = rng
        self.arena_bound = float(arena_bound)
        self.kind = np.zeros(self.n, dtype=np.int8)
        self.wall_points = np.zeros((self.n, _MAX_WALL_POINTS, 2), dtype=np.float32)
        self.wall_count = np.zeros(self.n, dtype=np.int32)
        self.hole_centers = np.zeros((self.n, _MAX_HOLES, 2), dtype=np.float32)
        self.hole_radii = np.zeros((self.n, _MAX_HOLES), dtype=np.float32)
        self.hole_count = np.zeros(self.n, dtype=np.int32)
        self.hill_centers = np.zeros((self.n, _MAX_HILLS, 2), dtype=np.float32)
        self.hill_radii = np.zeros((self.n, _MAX_HILLS), dtype=np.float32)
        self.hill_heights = np.zeros((self.n, _MAX_HILLS), dtype=np.float32)
        self.hill_count = np.zeros(self.n, dtype=np.int32)

    def set_arena_bound(self, value: float) -> None:
        self.arena_bound = float(value)

    def reset(
        self,
        mask: np.ndarray,
        player_pos: np.ndarray,
        enemy_pos: np.ndarray,
        terrain_mix: float,
    ) -> None:
        """Generate a fresh layout for selected rows; `terrain_mix` is the hard-map share."""
        rows = np.flatnonzero(np.asarray(mask, dtype=bool))
        if rows.size == 0:
            return
        self.kind[rows] = 0
        self.wall_points[rows] = 0.0
        self.wall_count[rows] = 0
        self.hole_centers[rows] = 0.0
        self.hole_radii[rows] = 0.0
        self.hole_count[rows] = 0
        self.hill_centers[rows] = 0.0
        self.hill_radii[rows] = 0.0
        self.hill_heights[rows] = 0.0
        self.hill_count[rows] = 0

        mix = float(np.clip(terrain_mix, 0.0, 1.0))
        if mix <= 0.0:
            return
        active = rows[self.rng.random(rows.size) < mix]
        if active.size == 0:
            return
        kinds = self.rng.integers(1, 5, size=active.size, dtype=np.int8)
        self.kind[active] = kinds
        for row, kind in zip(active, kinds):
            start = player_pos[row]
            end = enemy_pos[row]
            delta = end - start
            norm = max(float(np.linalg.norm(delta)), 1e-6)
            forward = delta / norm
            side = np.array([-forward[1], forward[0]], dtype=np.float32)
            mid = (start + end) * 0.5

            if kind in (1, 4):
                shift = forward * float(self.rng.uniform(-0.35, 0.35))
                self._add_wall(int(row), mid + shift, side, float(self.rng.uniform(3.8, 5.8)))
            if kind in (2, 4):
                self._add_hole(int(row), mid + forward * float(self.rng.uniform(-0.25, 0.25))
                               + side * float(self.rng.uniform(-0.9, 0.9)),
                               float(self.rng.uniform(0.78, 1.05)))
                if kind == 4 and self.rng.random() < 0.65:
                    self._add_hole(int(row), mid - forward * 1.0 + side * float(self.rng.uniform(1.2, 2.1)),
                                   float(self.rng.uniform(0.70, 0.92)))
            if kind in (3, 4):
                hill_center = mid + side * float(self.rng.uniform(-1.0, 1.0))
                self._add_hill(int(row), hill_center, float(self.rng.uniform(3.0, 4.6)),
                               float(self.rng.uniform(1.0, 2.4)))
                if kind == 4 and self.rng.random() < 0.4:
                    self._add_hill(int(row), mid - side * float(self.rng.uniform(2.0, 3.0)),
                                   float(self.rng.uniform(2.0, 3.2)), float(self.rng.uniform(0.7, 1.5)))

    def _add_wall(self, row: int, center: np.ndarray, direction: np.ndarray, length: float) -> None:
        count = 9
        offsets = np.linspace(-length * 0.5, length * 0.5, count, dtype=np.float32)
        self.wall_points[row, :count] = center[None, :] + offsets[:, None] * direction[None, :]
        self.wall_count[row] = count

    def _add_hole(self, row: int, center: np.ndarray, radius: float) -> None:
        slot = int(self.hole_count[row])
        if slot >= _MAX_HOLES:
            return
        self.hole_centers[row, slot] = center
        self.hole_radii[row, slot] = radius
        self.hole_count[row] += 1

    def _add_hill(self, row: int, center: np.ndarray, radius: float, height: float) -> None:
        slot = int(self.hill_count[row])
        if slot >= _MAX_HILLS:
            return
        self.hill_centers[row, slot] = center
        self.hill_radii[row, slot] = radius
        self.hill_heights[row, slot] = height
        self.hill_count[row] += 1

    def wall_at(self, positions: np.ndarray) -> np.ndarray:
        p = np.asarray(positions, dtype=np.float32).reshape(self.n, 2)
        delta = p[:, None, :] - self.wall_points
        d2 = np.sum(delta * delta, axis=2)
        active = np.arange(_MAX_WALL_POINTS)[None, :] < self.wall_count[:, None]
        return np.any(active & (d2 < _WALL_RADIUS * _WALL_RADIUS), axis=1)

    def hole_at(self, positions: np.ndarray) -> np.ndarray:
        p = np.asarray(positions, dtype=np.float32).reshape(self.n, 2)
        delta = p[:, None, :] - self.hole_centers
        d2 = np.sum(delta * delta, axis=2)
        active = np.arange(_MAX_HOLES)[None, :] < self.hole_count[:, None]
        return np.any(active & (d2 < self.hole_radii * self.hole_radii), axis=1)

    def has_support(self, positions: np.ndarray) -> np.ndarray:
        p = np.asarray(positions, dtype=np.float32).reshape(self.n, 2)
        inside = np.all(np.abs(p) < self.arena_bound - 0.1, axis=1)
        return inside & ~self.hole_at(p)

    def height(self, positions: np.ndarray) -> np.ndarray:
        p = np.asarray(positions, dtype=np.float32).reshape(self.n, 2)
        delta = p[:, None, :] - self.hill_centers
        distance = np.linalg.norm(delta, axis=2)
        profile = np.clip(1.0 - distance / np.maximum(self.hill_radii, 1e-3), 0.0, 1.0)
        active = np.arange(_MAX_HILLS)[None, :] < self.hill_count[:, None]
        return np.max(np.where(active, profile * self.hill_heights, 0.0), axis=1).astype(np.float32)

    def blocks_line(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        a = np.asarray(start, dtype=np.float32).reshape(self.n, 2)
        b = np.asarray(end, dtype=np.float32).reshape(self.n, 2)
        blocked = np.zeros(self.n, dtype=bool)
        for t in np.linspace(0.08, 0.92, 11, dtype=np.float32):
            blocked |= self.wall_at(a + (b - a) * t)
        return blocked

    def resolve_move(self, old_pos: np.ndarray, proposed_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        old = np.asarray(old_pos, dtype=np.float32).reshape(self.n, 2)
        proposed = np.asarray(proposed_pos, dtype=np.float32).reshape(self.n, 2)
        wall = np.zeros(self.n, dtype=bool)
        hole = np.zeros(self.n, dtype=bool)
        outside = np.zeros(self.n, dtype=bool)
        # Sweep the whole movement segment so a fast step cannot hop across a thin wall
        # or a floor opening without collision.
        for fraction in (0.2, 0.4, 0.6, 0.8, 1.0):
            sample = old + (proposed - old) * fraction
            wall |= self.wall_at(sample)
            hole |= self.hole_at(sample)
            outside |= np.any(np.abs(sample) >= self.arena_bound, axis=1)
        blocked = wall | hole | outside
        return np.where(blocked[:, None], old, proposed), blocked, hole

    def sensor_features(
        self,
        player_pos: np.ndarray,
        enemy_pos: np.ndarray,
        dynamic_obstacles: np.ndarray,
        obstacle_ttl: np.ndarray,
    ) -> np.ndarray:
        """Return four-direction clearance/support plus target-height and slope sensors."""
        player = np.asarray(player_pos, dtype=np.float32).reshape(self.n, 2)
        enemy = np.asarray(enemy_pos, dtype=np.float32).reshape(self.n, 2)
        delta = enemy - player
        forward = delta / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-6)
        right = np.stack([-forward[:, 1], forward[:, 0]], axis=1)
        directions = np.stack([forward, -right, right, -forward], axis=1)  # forward/left/right/back
        clearance = np.ones((self.n, 4), dtype=np.float32)
        blocked_seen = np.zeros((self.n, 4), dtype=bool)
        dyn = np.asarray(dynamic_obstacles, dtype=np.float32)
        ttl = np.asarray(obstacle_ttl, dtype=np.float32)
        for step_distance in _SENSOR_DISTANCES:
            sample = player[:, None, :] + directions * step_distance
            flat = sample.reshape(self.n * 4, 2)
            # Repeat each arena's static layout for its four directional probes.
            wall = np.repeat(self.wall_points, 4, axis=0)
            wall_n = np.repeat(self.wall_count, 4)
            wall_delta = flat[:, None, :] - wall
            wall_d2 = np.sum(wall_delta * wall_delta, axis=2)
            wall_active = np.arange(_MAX_WALL_POINTS)[None, :] < wall_n[:, None]
            blocked = np.any(wall_active & (wall_d2 < _WALL_RADIUS * _WALL_RADIUS), axis=1)
            hole_c = np.repeat(self.hole_centers, 4, axis=0)
            hole_r = np.repeat(self.hole_radii, 4, axis=0)
            hole_n = np.repeat(self.hole_count, 4)
            hole_delta = flat[:, None, :] - hole_c
            hole_d2 = np.sum(hole_delta * hole_delta, axis=2)
            hole_active = np.arange(_MAX_HOLES)[None, :] < hole_n[:, None]
            blocked |= np.any(hole_active & (hole_d2 < hole_r * hole_r), axis=1)
            dyn_delta = flat[:, None, :] - np.repeat(dyn, 4, axis=0)
            dyn_d2 = np.sum(dyn_delta * dyn_delta, axis=2)
            dyn_live = np.repeat(ttl > 0.0, 4, axis=0)
            blocked |= np.any(dyn_live & (dyn_d2 < 0.95 * 0.95), axis=1)
            blocked |= np.any(np.abs(flat) >= self.arena_bound - 0.1, axis=1)
            blocked = blocked.reshape(self.n, 4)
            newly_blocked = blocked & ~blocked_seen
            clearance = np.where(newly_blocked, step_distance / 4.0, clearance)
            blocked_seen |= blocked

        support_samples = player[:, None, :] + directions * 1.5
        support = np.stack([self.has_support(support_samples[:, i, :]) for i in range(4)], axis=1)
        target_height = np.clip((self.height(enemy) - self.height(player)) / 3.0, -1.0, 1.0)
        ahead_height = self.height(player + forward * 1.5)
        current_height = self.height(player)
        slope = np.clip((ahead_height - current_height) / 2.0, -1.0, 1.0)
        return np.concatenate([
            clearance,
            support.astype(np.float32),
            target_height[:, None].astype(np.float32),
            slope[:, None].astype(np.float32),
        ], axis=1)
