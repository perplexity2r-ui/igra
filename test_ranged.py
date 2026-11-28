"""Regression tests for trained bow/crossbow actions in the PRO simulator."""
from __future__ import annotations

import math

import numpy as np
import pytest

from env.pro_vec_env import DT, ProArena, RANGED_TYPES
from env.sim_env import SimulatedPvPEnv


@pytest.mark.parametrize(
    ("ranged_id", "shot_flag", "charge_time"),
    [
        ("bow", "bow_shot", 1.0),
        ("crossbow", "crossbow_shot", 1.25),
    ],
)
def test_action_17_draws_or_loads_before_releasing_arrows(
    ranged_id: str, shot_flag: str, charge_time: float,
):
    arena = ProArena(1, seed=920, terrain_mix=0.0)
    arena.set_enemy_skill(0.0)
    arena.set_aggression(0.0)
    arena.set_player_aggression(0.0)

    arena.L.pos[0] = np.array([0.0, 0.0], dtype=np.float32)
    arena.E.pos[0] = np.array([8.0, 0.0], dtype=np.float32)
    arena.L.vel[0] = arena.E.vel[0] = 0.0
    arena.L.ranged_type[0] = next(
        i for i, spec in enumerate(RANGED_TYPES) if spec.ranged_id == ranged_id
    )
    arena.L.ranged[0] = 2.0
    arena.L.has_ranged_weapon[0] = True
    arena.L.ranged_charge[0] = 0.0
    arena.L.ranged_loaded[0] = False
    arena.L.use_lock[0] = 0.0
    arena.E.hp[0] = 20.0
    arena.E.armor[0] = 0
    arena.E.abs[0] = 0.0
    arena.E.has_shield[0] = False
    arena.E.has_totem[0] = False
    arena.E.stun[0] = 5.0
    arena.E.gaps[0] = arena.E.splashes[0] = arena.E.pearls[0] = 0.0
    arena.E.buffs[0] = arena.E.ranged[0] = 0.0
    arena.E.blocks[0] = arena.E.cobwebs[0] = arena.E.lava_bk[0] = 0.0
    arena.E.water_bk[0] = arena.E.crystals[0] = arena.E.anchors[0] = 0.0
    arena.E.minecarts[0] = arena.E.obsidian[0] = 0.0

    expected_feature = 0.75 if ranged_id == "bow" else 0.85
    assert np.isclose(arena._obs()[0, 18], expected_feature)
    if ranged_id == "crossbow":
        arena.L.ranged_loaded[0] = True
        assert np.isclose(arena._obs()[0, 18], 1.10)
        arena.L.ranged_loaded[0] = False
    arena.L.ranged_charge[0] = 0.2
    assert np.isclose(arena._obs()[0, 18], 1.25)
    arena.L.ranged_charge[0] = 0.0
    arena.L.ranged[0] = 0.0
    assert arena._obs()[0, 18] <= 0.60, "spent arrows must remove the ready-weapon signal"
    arena.L.ranged[0] = 2.0

    initial_distance = float(np.linalg.norm(arena.E.pos[0] - arena.L.pos[0]))
    action = np.array([17], dtype=np.int32)
    neutral = np.array([0], dtype=np.int32)
    _, _, _, _, _, info = arena.step(action)
    assert float(np.linalg.norm(arena.E.pos[0] - arena.L.pos[0])) < initial_distance
    assert not bool(info[shot_flag][0]), "action 17 starts a draw/load; it does not fire instantly"
    assert arena.L.ranged[0] == 2.0
    assert arena.L.ranged_charge[0] > 0.0
    assert float(info["damage_dealt"][0]) == 0.0

    # The charge is sustained even if the policy selects a different action while the
    # vanilla bow/crossbow use animation is in progress.
    steps_before_release = math.ceil(charge_time / DT) - 2
    for _ in range(steps_before_release):
        _, _, _, _, _, info = arena.step(neutral)
        assert not bool(info[shot_flag][0])
        assert float(info["damage_dealt"][0]) == 0.0

    _, _, _, _, _, info = arena.step(neutral)
    if ranged_id == "bow":
        assert bool(info[shot_flag][0])
        assert bool(info["bow_hit"][0])
        assert float(info["damage_dealt"][0]) > 0.0
        assert arena.L.ranged[0] == 1.0
        assert not arena.L.ranged_loaded[0]
    else:
        assert not bool(info[shot_flag][0]), "crossbow load completion is not the shot"
        assert arena.L.ranged[0] == 1.0, "loading consumes one bolt"
        assert bool(arena.L.ranged_loaded[0])

        # A loaded crossbow releases on the next valid action 17 without a second draw.
        _, _, _, _, _, info = arena.step(action)
        assert bool(info[shot_flag][0])
        assert bool(info["crossbow_hit"][0])
        assert float(info["damage_dealt"][0]) > 0.0
        assert arena.L.ranged[0] == 1.0, "the loaded bolt was already removed from reserve"
        assert not arena.L.ranged_loaded[0]

    # A bow cannot instantly repeat its shot, and a fired crossbow is no longer loaded.
    _, _, _, _, _, next_info = arena.step(action)
    assert not bool(next_info[shot_flag][0])
    assert arena.L.ranged[0] == 1.0


@pytest.mark.parametrize(
    ("ranged_id", "shot_flag", "charge_time"),
    [("bow", "bow_shot", 1.0), ("crossbow", "crossbow_shot", 1.25)],
)
def test_legacy_simulator_matches_arrow_draw_load_and_fire_cycle(
    ranged_id: str, shot_flag: str, charge_time: float,
):
    env = SimulatedPvPEnv(max_steps=100)
    env.reset(seed=921)
    learner, enemy = env.learner, env.enemy
    learner.pos[:] = (-6.5, 0.0)
    enemy.pos[:] = (6.5, 0.0)
    learner.vel[:] = enemy.vel[:] = 0.0
    learner.ranged_spec = next(spec for spec in RANGED_TYPES if spec.ranged_id == ranged_id)
    learner.ranged = 2
    learner.has_ranged_weapon = True
    learner.ranged_charge = 0.0
    learner.ranged_loaded = False
    learner.use_lock = 0.0
    expected_feature = 0.75 if ranged_id == "bow" else 0.85
    assert np.isclose(env._get_obs()[18], expected_feature)
    enemy.ranged = 0
    enemy.ranged_charge = 0.0
    enemy.ranged_loaded = False
    enemy.gaps = enemy.splashes = enemy.pearls = enemy.buffs = 0
    enemy.blocks = enemy.cobwebs = enemy.lava_buckets = enemy.water_buckets = 0
    enemy.has_shield = enemy.has_totem = False

    _, _, _, _, info = env.step(17)
    assert not info[shot_flag]
    assert learner.ranged == 2
    assert learner.ranged_charge > 0.0
    assert np.isclose(env._get_obs()[18], 1.25)
    steps_before_release = math.ceil(charge_time / env.DT) - 2
    for _ in range(steps_before_release):
        _, _, _, _, info = env.step(0)
        assert not info[shot_flag]

    _, _, _, _, info = env.step(0)
    if ranged_id == "bow":
        assert info["bow_shot"] and info["bow_hit"]
        assert info["damage_dealt"] > 0.0
        assert learner.ranged == 1
    else:
        assert not info["crossbow_shot"]
        assert learner.ranged == 1 and learner.ranged_loaded
        assert np.isclose(env._get_obs()[18], 1.10)
        _, _, _, _, info = env.step(17)
        assert info["crossbow_shot"] and info["crossbow_hit"]
        assert info["damage_dealt"] > 0.0
        assert learner.ranged == 1 and not learner.ranged_loaded
