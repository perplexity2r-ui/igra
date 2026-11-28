import numpy as np

from env.multi_opponent import (
    FULL_CROWD_WIN_BONUS,
    MAX_OPPONENTS,
    MIN_OPPONENTS,
    MULTI_OBS_DIM,
    MultiOpponentArena,
    MultiOpponentVecEnv,
)


def test_multi_arena_has_three_to_eight_live_opponents_and_crowd_features():
    arena = MultiOpponentArena(48, seed=20261128, terrain_mix=1.0, max_opponents=8)
    obs = arena.reset()
    counts = arena.alive.sum(axis=1)

    assert obs.shape == (48, MULTI_OBS_DIM)
    assert MULTI_OBS_DIM == 52
    shield_up = (arena.core.L.has_shield & arena.core.L.shield_up
                 & (arena.core.L.shield_disabled <= 0.0))
    np.testing.assert_array_equal(obs[:, 19], shield_up.astype(np.float32))
    assert np.all(counts >= MIN_OPPONENTS)
    assert np.all(counts <= MAX_OPPONENTS)
    np.testing.assert_allclose(obs[:, 44], counts / MAX_OPPONENTS)
    assert np.any(counts == MAX_OPPONENTS)
    assert np.count_nonzero(counts == MAX_OPPONENTS) > np.count_nonzero(counts == MIN_OPPONENTS)


def test_shield_blocking_state_is_observed_and_real_melee_blocks_are_counted():
    arena = MultiOpponentArena(1, seed=207, terrain_mix=0.0, max_opponents=3)
    arena.set_enemy_skill(1.0)
    arena.set_aggression(1.0)
    arena.core.L.pos[0] = np.array([0.0, 0.0], dtype=np.float32)
    arena.core.L.has_shield[0] = True
    arena.core.L.crystals[0] = arena.core.L.anchors[0] = arena.core.L.minecarts[0] = 0.0
    arena.enemies.pos[0, :3] = np.array([[2.5, 0.0], [7.5, 0.0], [0.0, 7.5]])
    arena.enemies.weapon[0, :3] = 3  # non-axe, so a raised shield must block the swing
    arena.enemies.cd[0, :3] = 0.0
    arena.enemies.has_shield[0, :3] = False
    arena.enemies.gaps[0, :3] = arena.enemies.splashes[0, :3] = 0.0
    arena.enemies.pearls[0, :3] = arena.enemies.buffs[0, :3] = 0.0
    arena.enemies.crystals[0, :3] = arena.enemies.obsidian[0, :3] = 0.0
    arena.enemies.anchors[0, :3] = arena.enemies.minecarts[0, :3] = 0.0
    arena.enemies.ranged[0, :3] = arena.enemies.cobwebs[0, :3] = 0.0
    arena.enemies.lava_bk[0, :3] = 0.0
    arena.active_slot[0] = 0
    arena._copy_slot_to_core(np.array([0], dtype=np.int64))
    assert arena._obs()[0, 19] == 0.0

    _, _, _, _, _, info = arena.step(np.array([12], dtype=np.int32))

    assert arena._obs()[0, 19] == 1.0
    assert info["tech"][0] == 17
    assert info["blocked_hit"][0]
    assert info["damage_taken"][0] == 0.0


def test_action_nine_queues_and_lands_a_jump_critical_hit():
    arena = MultiOpponentArena(1, seed=208, terrain_mix=0.0, max_opponents=3)
    arena.set_enemy_skill(0.0)
    arena.set_aggression(0.0)
    arena.core.L.pos[0] = np.array([0.0, 0.0], dtype=np.float32)
    arena.core.L.vel[0] = 0.0
    arena.core.L.y[0] = arena.core.L.vy[0] = arena.core.L.cd[0] = 0.0
    arena.core.L.has_shield[0] = False
    arena.core.L.stun[0] = 0.0
    arena.enemies.pos[0, :3] = np.array([[2.8, 0.0], [7.5, 0.0], [0.0, 7.5]])
    arena.enemies.hp[0, :3] = 20.0
    arena.enemies.weapon[0, :3] = 0
    arena.enemies.stun[0, 0] = 5.0
    arena.enemies.has_shield[0, :3] = False
    arena.enemies.gaps[0, :3] = arena.enemies.splashes[0, :3] = 0.0
    arena.enemies.pearls[0, :3] = arena.enemies.buffs[0, :3] = 0.0
    arena.enemies.crystals[0, :3] = arena.enemies.obsidian[0, :3] = 0.0
    arena.enemies.anchors[0, :3] = arena.enemies.minecarts[0, :3] = 0.0
    arena.enemies.ranged[0, :3] = arena.enemies.cobwebs[0, :3] = 0.0
    arena.enemies.lava_bk[0, :3] = 0.0
    arena.active_slot[0] = 0
    arena._copy_slot_to_core(np.array([0], dtype=np.int64))

    landed_crit = False
    for step in range(8):
        action = 9 if step == 0 else 0
        _, _, _, _, _, info = arena.step(np.array([action], dtype=np.int32))
        landed_crit |= bool(info["critical_hit"][0])
        if landed_crit:
            break

    assert landed_crit


def test_observation_includes_nearest_non_target_opponent_offset():
    arena = MultiOpponentArena(1, seed=156, terrain_mix=0.0, max_opponents=3)
    arena.reset()
    arena.core.L.pos[0] = np.array([0.0, 0.0], dtype=np.float32)
    arena.enemies.pos[0, :3] = np.array([[5.0, 0.0], [-2.0, 1.0], [0.0, 4.0]])
    arena.alive[0] = np.arange(MAX_OPPONENTS) < 3
    arena.active_slot[0] = 0
    arena._copy_slot_to_core(np.array([0], dtype=np.int64))

    obs = arena._obs()[0]

    np.testing.assert_allclose(obs[48:50], np.array([-0.1, 0.05], dtype=np.float32))
    other_delta = np.array([[-2.0, 1.0], [0.0, 4.0]], dtype=np.float32)
    other_distance = np.linalg.norm(other_delta, axis=1)
    weights = 1.0 / (1.0 + other_distance)
    expected_direction = np.sum(
        other_delta / other_distance[:, None] * weights[:, None], axis=0
    ) / weights.sum()
    np.testing.assert_allclose(obs[50:52], expected_direction, rtol=1e-6, atol=1e-6)


def test_retarget_prefers_a_wounded_opponent_within_nearby_range():
    arena = MultiOpponentArena(1, seed=155, terrain_mix=0.0, max_opponents=3)
    arena.reset()
    arena.core.L.pos[0] = 0.0
    arena.enemies.pos[0, :3] = np.array([[3.0, 0.0], [4.0, 0.0], [5.0, 0.0]])
    arena.enemies.hp[0, :3] = np.array([20.0, 5.0, 20.0])
    arena.alive[0] = np.arange(MAX_OPPONENTS) < 3

    arena._retarget(np.array([True]))

    assert arena.active_slot[0] == 1
    assert arena.core.E.hp[0] == 5.0


def test_learner_loadout_has_a_small_average_edge_over_opponent_kits():
    arena = MultiOpponentArena(512, seed=141, terrain_mix=0.0, max_opponents=8)
    active = arena.alive
    learner_damage = arena.core.L.dmg().mean()
    opponent_damage = arena.enemies.dmg()[active].mean()
    learner_armor = arena.core.L.armor.mean()
    opponent_armor = arena.enemies.armor[active].mean()

    assert learner_damage > opponent_damage
    assert learner_armor > opponent_armor
    assert learner_armor <= 4.0


def test_multi_arena_steps_keep_observations_finite_and_count_kills():
    arena = MultiOpponentArena(8, seed=44, terrain_mix=1.0, max_opponents=5)
    arena.set_player_aggression(1.0)
    obs = arena.reset()
    rng = np.random.default_rng(45)

    for _ in range(25):
        obs, reward, terminated, truncated, done, info = arena.step(
            rng.integers(0, 30, size=8, dtype=np.int32)
        )
        assert obs.shape == (8, MULTI_OBS_DIM)
        assert np.isfinite(obs).all()
        assert np.isfinite(reward).all()
        assert np.all(info["opponents_alive"] >= 0)
        assert np.all(info["opponents_alive"] <= info["opponent_count"])
        assert np.all(info["enemy_kills"] >= 0)


def test_multi_arena_explosives_damage_the_nearby_opponents_too():
    arena = MultiOpponentArena(1, seed=313, terrain_mix=0.0, max_opponents=3)
    arena.reset()
    arena.core.L.pos[0] = np.array([0.0, 0.0], dtype=np.float32)
    arena.enemies.pos[0, 0] = np.array([4.0, 0.0], dtype=np.float32)
    arena.enemies.pos[0, 1] = np.array([4.5, 0.5], dtype=np.float32)
    arena.enemies.pos[0, 2] = np.array([4.5, -0.5], dtype=np.float32)
    arena.alive[0] = np.arange(MAX_OPPONENTS) < 3
    arena.enemy_count[0] = 3
    arena.active_slot[0] = 0
    arena.core.L.crystals[0] = 1.0
    arena.core.L.obsidian[0] = 4.0
    arena._copy_slot_to_core(np.array([0], dtype=np.int64))
    before = arena.enemies.hp[0, 1:3].copy()

    _, _, _, _, _, info = arena.step(np.array([22], dtype=np.int32))

    assert info["tech"][0] == 11  # end crystal
    assert info["aoe_enemy_hits"][0] == 2
    assert np.all(arena.enemies.hp[0, 1:3] < before)


def test_final_opponent_kill_gets_a_win_shaping_bonus():
    arena = MultiOpponentArena(1, seed=522, terrain_mix=0.0, max_opponents=3)
    arena.reset()
    arena.core.L.pos[0] = np.array([0.0, 0.0], dtype=np.float32)
    arena.enemies.pos[0, 0] = np.array([4.0, 0.0], dtype=np.float32)
    arena.enemies.hp[0, 0] = 5.0
    arena.enemies.has_totem[0, 0] = False
    arena.alive[0] = False
    arena.alive[0, 0] = True
    arena.enemy_count[0] = 1
    arena.active_slot[0] = 0
    arena.core.L.crystals[0] = 1.0
    arena.core.L.obsidian[0] = 4.0
    arena._copy_slot_to_core(np.array([0], dtype=np.int64))

    _, reward, terminated, _, done, info = arena.step(np.array([22], dtype=np.int32))

    assert info["enemy_dead"][0]
    assert terminated[0] and done[0]
    assert reward[0] >= FULL_CROWD_WIN_BONUS + 99.0


def test_multi_vec_env_enforces_exact_fight_quotas():
    env = MultiOpponentVecEnv(4, seed=122, terrain_mix=1.0, max_opponents=3)
    env.set_fight_limit(5)
    obs = env.reset()
    assert obs.shape == (4, MULTI_OBS_DIM)

    rng = np.random.default_rng(123)
    for _ in range(2500):
        obs, rewards, dones, _ = env.step(rng.integers(0, 30, size=4, dtype=np.int32))
        if env.episodes >= 5:
            break

    assert env.episodes == 5
    assert env.completed_by_env.tolist() == [2, 1, 1, 1]
    assert obs.shape == (4, MULTI_OBS_DIM)
