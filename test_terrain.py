"""Regression tests for difficult PvP terrain and the terrain-aware observation contract."""
from __future__ import annotations

import numpy as np

from env.combat_terrain import TERRAIN_SENSOR_COUNT
from env.pro_vec_env import MAX_STEPS, OBS_DIM, TERRAIN_OBS_DIM, ProArena, ProPvPEnv, ProPvPVecEnv


def test_flat_legacy_shape_and_hard_terrain_shape():
    flat = ProArena(8, seed=41)
    legacy = ProArena(8, seed=41, terrain_obs=True)
    assert flat._obs().shape == (8, OBS_DIM)
    assert np.all(flat.terrain.kind == 0)
    np.testing.assert_allclose(flat._obs(), legacy._obs()[:, :OBS_DIM])

    hard = ProArena(128, seed=42, terrain_mix=1.0, terrain_obs=True)
    obs = hard._obs()
    assert TERRAIN_OBS_DIM == OBS_DIM + TERRAIN_SENSOR_COUNT == 44
    assert obs.shape == (128, TERRAIN_OBS_DIM)
    assert np.isfinite(obs).all()
    assert np.all(hard.terrain.kind > 0)
    assert len(np.unique(hard.terrain.kind)) >= 3


def test_wall_blocks_movement_and_line_of_sight():
    arena = ProArena(1, seed=5)
    arena.L.pos[0] = (0.0, 0.0)
    arena.E.pos[0] = (4.0, 0.0)
    arena.terrain.wall_count[0] = 9
    arena.terrain.wall_points[0, :9, 0] = 1.0
    arena.terrain.wall_points[0, :9, 1] = np.linspace(-2.0, 2.0, 9)

    assert arena._obstacle_between(arena.L.pos, arena.E.pos)[0]
    resolved, blocked, pit = arena.terrain.resolve_move(
        arena.L.pos.copy(), np.array([[1.5, 0.0]], dtype=np.float32)
    )
    assert blocked[0] and not pit[0]
    np.testing.assert_allclose(resolved, arena.L.pos)


def test_pits_are_unsupported_and_movement_is_rejected():
    arena = ProArena(1, seed=6)
    arena.L.pos[0] = (0.0, 0.0)
    arena.terrain.hole_count[0] = 1
    arena.terrain.hole_centers[0, 0] = (1.5, 0.0)
    arena.terrain.hole_radii[0, 0] = 0.9
    pit = np.array([[1.5, 0.0]], dtype=np.float32)

    assert not arena.terrain.has_support(pit)[0]
    resolved, blocked, pit_blocked = arena.terrain.resolve_move(arena.L.pos.copy(), pit)
    assert blocked[0] and pit_blocked[0]
    np.testing.assert_allclose(resolved, arena.L.pos)


def test_hills_feed_height_and_slope_sensors():
    arena = ProArena(1, seed=7, terrain_obs=True)
    arena.L.pos[0] = (0.0, 0.0)
    arena.E.pos[0] = (3.0, 0.0)
    arena.terrain.hill_count[0] = 1
    arena.terrain.hill_centers[0, 0] = (2.0, 0.0)
    arena.terrain.hill_radii[0, 0] = 4.0
    arena.terrain.hill_heights[0, 0] = 2.0

    features = arena.terrain.sensor_features(
        arena.L.pos, arena.E.pos, arena.obstacles, arena.obstacle_ttl
    )[0]
    assert features[8] > 0.0, "target elevation should be observable"
    assert features[9] > 0.0, "uphill slope should be observable"


def test_max_player_aggression_rewards_closing_distance():
    neutral = ProArena(16, seed=31)
    aggressive = ProArena(16, seed=31)
    aggressive.set_player_aggression(1.0)
    actions = np.ones(16, dtype=np.int32)  # forward pursuit is the positive-control action
    _, neutral_reward, *_ = neutral.step(actions)
    _, aggressive_reward, *_ = aggressive.step(actions)
    assert np.all(aggressive_reward >= neutral_reward)
    assert np.any(aggressive_reward > neutral_reward)

    vec = ProPvPVecEnv(2, seed=32)
    vec.set_player_aggression(1.0)
    assert vec.core.player_aggression == 1.0


def test_enemy_skill_setter_changes_the_skill_used_by_the_brain():
    arena = ProArena(4, seed=13)
    arena.set_enemy_skill(0.27)
    np.testing.assert_allclose(arena.e_skill, 0.27)
    assert np.all(arena.difficulty == np.float32(0.27))

    vec = ProPvPVecEnv(2, seed=14)
    vec.set_enemy_skill(0.63)
    np.testing.assert_allclose(vec.core.e_skill, 0.63)


def test_exact_fight_limit_freezes_the_vector_env_without_counting_extra_fights():
    vec = ProPvPVecEnv(1, seed=21, terrain_mix=1.0, terrain_obs=True)
    vec.set_fight_limit(0)
    before = vec.reset()
    vec.step_async(np.array([0], dtype=np.int32))
    obs, reward, done, infos = vec.step_wait()
    np.testing.assert_array_equal(obs, before)
    np.testing.assert_array_equal(reward, np.zeros(1, dtype=np.float32))
    np.testing.assert_array_equal(done, np.zeros(1, dtype=bool))
    assert vec.episodes == 0 and infos == [{}]


def test_vector_env_propagates_time_limit_truncations_as_done():
    vec = ProPvPVecEnv(1, seed=23)
    vec.core.step_count[0] = MAX_STEPS - 1
    vec.step_async(np.zeros(1, dtype=np.int32))
    _, _, done, infos = vec.step_wait()
    assert done[0]
    assert infos[0]["TimeLimit.truncated"]


def test_exact_quotas_freeze_completed_vector_rows_without_corrupting_them():
    vec = ProPvPVecEnv(4, seed=22, terrain_mix=1.0, terrain_obs=True)
    vec.set_fight_limit(1)
    before = vec.reset()
    frozen_pos = vec.core.L.pos[1:].copy()
    frozen_hp = vec.core.L.hp[1:].copy()
    vec.step_async(np.zeros(4, dtype=np.int32))
    obs, reward, done, _ = vec.step_wait()
    np.testing.assert_array_equal(obs[1:], before[1:])
    np.testing.assert_array_equal(vec.core.L.pos[1:], frozen_pos)
    np.testing.assert_array_equal(vec.core.L.hp[1:], frozen_hp)
    np.testing.assert_array_equal(reward[1:], np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(done[1:], np.zeros(3, dtype=bool))
    assert vec.fight_quota.tolist() == [1, 0, 0, 0]


def test_wrappers_advertise_terrain_shape_when_enabled():
    env = ProPvPEnv(seed=8, terrain_mix=1.0, terrain_obs=True)
    obs, _ = env.reset(seed=8)
    assert env.observation_space.shape == (TERRAIN_OBS_DIM,)
    assert obs.shape == (TERRAIN_OBS_DIM,)

    vec = ProPvPVecEnv(8, seed=8, terrain_mix=1.0, terrain_obs=True)
    assert vec.observation_space.shape == (TERRAIN_OBS_DIM,)
    assert vec.reset().shape == (8, TERRAIN_OBS_DIM)
