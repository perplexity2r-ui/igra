"""Vectorized one-versus-many PvP fine-tuning environment.

The legacy terrain actor's 44 observations are kept in place and six crowd/threat
features are appended. Fights contain 3..8 simultaneously active opponents; the
nearest active opponent is the current PPO target while the remaining fighters
close in and attack the shared player state.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from .pro_vec_env import (
    ARENA_BOUND,
    BotArrays,
    DEATH,
    DT,
    MAX_STEPS,
    MELEE_POOL,
    REACH,
    RANGED_TYPES,
    TERRAIN_OBS_DIM,
    TECHNIQUES,
    T_ANCHOR,
    T_CRYSTAL,
    T_MINECART,
    T_PEARL_CRYSTAL,
    WALK,
    ProArena,
)

MIN_OPPONENTS = 3
MAX_OPPONENTS = 8
CROWD_FEATURES = 6
MULTI_OBS_DIM = TERRAIN_OBS_DIM + CROWD_FEATURES
DEATH_SHAPING_PENALTY = 400.0
FULL_CROWD_WIN_BONUS = 400.0

_WEAPON_DAMAGE = np.asarray([weapon.damage for weapon in MELEE_POOL], dtype=np.float32)
_WEAPON_UPGRADE = np.arange(len(MELEE_POOL), dtype=np.int32)
for _weapon_index, _weapon_damage in enumerate(_WEAPON_DAMAGE):
    _stronger = np.flatnonzero(_WEAPON_DAMAGE > _weapon_damage + 1e-6)
    if _stronger.size:
        _WEAPON_UPGRADE[_weapon_index] = int(
            _stronger[np.argmin(_WEAPON_DAMAGE[_stronger])]
        )


class MultiBotArrays(BotArrays):
    """BotArrays whose leading axis is arenas and second axis is opponent slots."""

    def __init__(self, n: int, slots: int):
        super().__init__(n * slots)
        self._arena_count = int(n)
        self._slot_count = int(slots)
        for name, value in list(vars(self).items()):
            if isinstance(value, np.ndarray):
                setattr(self, name, value.reshape((n, slots) + value.shape[1:]))


class MultiOpponentArena:
    """A batched one-versus-many combat arena with shared learner state."""

    def __init__(
        self,
        n: int = 128,
        seed: int | None = None,
        terrain_mix: float = 1.0,
        min_opponents: int = MIN_OPPONENTS,
        max_opponents: int = MAX_OPPONENTS,
    ):
        self.n = int(n)
        self.core = ProArena(
            self.n, seed=seed, terrain_mix=terrain_mix, terrain_obs=True
        )
        self.rng = self.core.rng
        self.enemies = MultiBotArrays(self.n, MAX_OPPONENTS)
        self.alive = np.zeros((self.n, MAX_OPPONENTS), dtype=bool)
        self.enemy_count = np.zeros(self.n, dtype=np.int32)
        self.active_slot = np.zeros(self.n, dtype=np.int32)
        self.kills_this_fight = np.zeros(self.n, dtype=np.int32)
        self.opponent_cap = int(np.clip(max_opponents, MIN_OPPONENTS, MAX_OPPONENTS))
        self.min_opponents = int(np.clip(min_opponents, MIN_OPPONENTS, self.opponent_cap))
        self.enemy_skill = 0.3
        self.enemy_aggression = 0.55
        self.player_aggression = 1.0
        self.extra_damage_taken = np.zeros(self.n, dtype=np.float32)
        self.extra_terrain_blocked = np.zeros(self.n, dtype=np.int32)
        self.extra_pit_blocked = np.zeros(self.n, dtype=np.int32)
        self.core.set_enemy_skill(self.enemy_skill)
        self.core.set_aggression(self.enemy_aggression)
        self.core.set_player_aggression(self.player_aggression)
        self.reset(np.ones(self.n, dtype=bool))

    @property
    def observation_space(self):
        return spaces.Box(-1.0, 2.0, (MULTI_OBS_DIM,), np.float32)

    @property
    def action_space(self):
        return spaces.Discrete(30)

    def set_enemy_skill(self, value: float) -> None:
        self.enemy_skill = float(np.clip(value, 0.0, 1.0))
        self.core.set_enemy_skill(self.enemy_skill)

    def set_aggression(self, value: float) -> None:
        self.enemy_aggression = float(np.clip(value, 0.0, 1.0))
        self.core.set_aggression(self.enemy_aggression)

    def set_player_aggression(self, value: float) -> None:
        self.player_aggression = float(np.clip(value, 0.0, 1.0))
        self.core.set_player_aggression(self.player_aggression)

    def set_combo_curriculum(self, share: float) -> None:
        self.core.set_combo_curriculum(share)

    def set_terrain_mix(self, value: float) -> None:
        self.core.set_terrain_mix(value)

    def set_opponent_cap(self, value: int) -> None:
        self.opponent_cap = int(np.clip(value, self.min_opponents, MAX_OPPONENTS))

    def _roll_opponents(self, rows: np.ndarray, counts: np.ndarray) -> None:
        """Roll normal-to-pro equipment; the learner gets a small, bounded gear edge."""
        if rows.size == 0:
            return
        r = self.rng
        k = rows.size
        m = MAX_OPPONENTS
        e = self.enemies
        e.reset_mask(np.isin(np.arange(self.n), rows))

        # Most opponents use ordinary random kits. A minority receive a modest
        # professional upgrade as the scripted skill curriculum gets harder.
        weapon = r.integers(0, len(MELEE_POOL), (k, m), dtype=np.int32)
        strong_weapons = np.asarray([6, 7, 8, 9, 10, 11], dtype=np.int32)
        weapon_favor = r.random((k, m)) < (0.04 + 0.14 * self.enemy_skill)
        weapon = np.where(
            weapon_favor,
            strong_weapons[r.integers(0, len(strong_weapons), (k, m))],
            weapon,
        )
        armor = r.integers(0, 5, (k, m), dtype=np.int32)
        armor_favor = r.random((k, m)) < (0.10 + 0.16 * self.enemy_skill)
        armor = np.where(armor_favor, r.integers(2, 5, (k, m)), armor)
        ranged_type = r.integers(0, len(RANGED_TYPES), (k, m), dtype=np.int32)

        e.weapon[rows] = weapon
        e.armor[rows] = armor
        e.ranged_type[rows] = ranged_type
        e.blocks[rows] = np.where(r.random((k, m)) < 0.85,
                                  r.integers(4, 25, (k, m)), 0).astype(np.float32)
        e.cobwebs[rows] = np.where(r.random((k, m)) < 0.55,
                                   r.integers(1, 5, (k, m)), 0).astype(np.float32)
        e.lava_bk[rows] = (r.random((k, m)) < 0.45).astype(np.float32)
        e.water_bk[rows] = (r.random((k, m)) < 0.50).astype(np.float32)
        e.has_shield[rows] = r.random((k, m)) < 0.55
        e.has_totem[rows] = (~e.has_shield[rows]) & (r.random((k, m)) < 0.20)

        heal_roll = r.integers(0, 5, (k, m))
        primary = r.random((k, m)) < 0.75
        e.gaps[rows] = np.where(
            heal_roll == 3,
            np.where(primary, r.integers(1, 4, (k, m)), 0),
            np.where(r.random((k, m)) < 0.45, r.integers(1, 3, (k, m)), 0),
        ).astype(np.float32)
        e.splashes[rows] = np.where(
            heal_roll != 3,
            np.where(primary, r.integers(1, 4, (k, m)), 0),
            np.where(r.random((k, m)) < 0.45, r.integers(1, 3, (k, m)), 0),
        ).astype(np.float32)
        e.pearls[rows] = np.where(r.random((k, m)) < 0.55,
                                  r.integers(1, 4, (k, m)), 0).astype(np.float32)
        e.buffs[rows] = np.where(r.random((k, m)) < 0.50,
                                 r.integers(1, 3, (k, m)), 0).astype(np.float32)
        ranged_counts = np.zeros((k, m), dtype=np.float32)
        for type_index, spec in enumerate(RANGED_TYPES):
            selected = ranged_type == type_index
            if selected.any():
                ranged_counts[selected] = r.integers(
                    spec.min_count, spec.max_count + 1, int(selected.sum())
                )
        e.ranged[rows] = np.where(r.random((k, m)) < 0.75,
                                  ranged_counts, 0.0).astype(np.float32)
        e.crystals[rows] = np.where(r.random((k, m)) < 0.62,
                                    r.integers(1, 7, (k, m)), 0).astype(np.float32)
        e.obsidian[rows] = np.where(r.random((k, m)) < 0.78,
                                    r.integers(4, 17, (k, m)), 0).astype(np.float32)
        e.anchors[rows] = np.where(r.random((k, m)) < 0.42,
                                   r.integers(1, 4, (k, m)), 0).astype(np.float32)
        e.minecarts[rows] = np.where(r.random((k, m)) < 0.46,
                                     r.integers(1, 5, (k, m)), 0).astype(np.float32)

        # Give the learner a small, consistent edge: one adjacent weapon tier when
        # available and one armor tier, capped at the best tier. Extra consumables
        # remain probabilistic so the loadout advantage does not become overwhelming.
        learner = self.core.L
        old_weapon = learner.weapon[rows].copy()
        learner.weapon[rows] = _WEAPON_UPGRADE[old_weapon]
        learner.armor[rows] = np.minimum(4, learner.armor[rows] + 1)
        learner.gaps[rows] = np.minimum(
            4.0, learner.gaps[rows] + (r.random(k) < 0.45).astype(np.float32)
        )
        learner.splashes[rows] = np.minimum(
            4.0, learner.splashes[rows] + (r.random(k) < 0.45).astype(np.float32)
        )
        learner.pearls[rows] = np.minimum(
            4.0, learner.pearls[rows] + (r.random(k) < 0.35).astype(np.float32)
        )

    def reset(self, mask: np.ndarray | None = None) -> np.ndarray:
        if mask is None:
            mask = np.ones(self.n, dtype=bool)
        mask = np.asarray(mask, dtype=bool)
        rows = np.flatnonzero(mask)
        self.core.reset(mask)
        self.enemies.reset_mask(mask)
        self.alive[mask] = False
        self.enemy_count[mask] = 0
        self.active_slot[mask] = 0
        self.kills_this_fight[mask] = 0
        self.extra_damage_taken[mask] = 0.0
        self.extra_terrain_blocked[mask] = 0
        self.extra_pit_blocked[mask] = 0
        if rows.size == 0:
            return self._obs()

        group_sizes = np.arange(self.min_opponents, self.opponent_cap + 1, dtype=np.int32)
        # Once the curriculum opens larger crowds, deliberately oversample them;
        # uniform sampling left the eight-opponent case underrepresented.
        weights = np.arange(1, group_sizes.size + 1, dtype=np.float64)
        counts = self.rng.choice(
            group_sizes, size=rows.size, p=weights / weights.sum()
        ).astype(np.int32)
        self.enemy_count[rows] = counts
        self.alive[rows] = np.arange(MAX_OPPONENTS)[None, :] < counts[:, None]

        # Start centrally so all members of the ring can engage. There are always at
        # least three live opponents; late curriculum exposes the actor to up to eight.
        learner_pos = self.rng.uniform(-0.75, 0.75, (rows.size, 2)).astype(np.float32)
        self.core.L.pos[rows] = learner_pos
        slot_ids = np.arange(MAX_OPPONENTS, dtype=np.float32)[None, :]
        base_angle = self.rng.uniform(0.0, 2.0 * np.pi, (rows.size, 1))
        angle = (base_angle + (2.0 * np.pi * slot_ids / counts[:, None])
                 + self.rng.uniform(-0.14, 0.14, (rows.size, MAX_OPPONENTS)))
        radius = self.rng.uniform(4.6, 7.0, (rows.size, MAX_OPPONENTS))
        positions = learner_pos[:, None, :] + np.stack(
            [np.cos(angle), np.sin(angle)], axis=2
        ) * radius[:, :, None]
        self.enemies.pos[rows] = np.clip(
            positions, -ARENA_BOUND + 0.5, ARENA_BOUND - 0.5
        ).astype(np.float32)
        self.enemies.vel[rows] = 0.0
        self.enemies.y[rows] = 0.0
        self.enemies.vy[rows] = 0.0
        self._roll_opponents(rows, counts)
        self._retarget(mask)
        self.core.terrain.reset(mask, self.core.L.pos, self.core.E.pos,
                                self.core.terrain_mix)
        if hasattr(self.core, "_prev_chain"):
            self.core._prev_chain[mask] = 0
        return self._obs()

    def _copy_slot_to_core(self, rows: np.ndarray) -> None:
        if rows.size == 0:
            return
        slots = self.active_slot[rows]
        for name, source in vars(self.enemies).items():
            if isinstance(source, np.ndarray):
                getattr(self.core.E, name)[rows] = source[rows, slots]

    def _copy_core_to_slot(self, rows: np.ndarray) -> None:
        if rows.size == 0:
            return
        slots = self.active_slot[rows]
        for name, destination in vars(self.enemies).items():
            if isinstance(destination, np.ndarray):
                destination[rows, slots] = getattr(self.core.E, name)[rows]

    def _retarget(self, mask: np.ndarray) -> None:
        rows = np.flatnonzero(np.asarray(mask, dtype=bool) & self.alive.any(axis=1))
        if rows.size == 0:
            return
        delta = self.enemies.pos[rows] - self.core.L.pos[rows, None, :]
        distance = np.linalg.norm(delta, axis=2)
        # After a kill, focus a weakened nearby foe instead of always starting a
        # fresh full-health duel. The distance term still prevents chasing far targets.
        effective_health = self.enemies.hp[rows] + self.enemies.abs[rows]
        score = distance + 0.15 * effective_health
        score = np.where(self.alive[rows], score, np.inf)
        self.active_slot[rows] = np.argmin(score, axis=1)
        self._copy_slot_to_core(rows)

    def _move_other_opponents(self) -> tuple[np.ndarray, np.ndarray]:
        slot_ids = np.arange(MAX_OPPONENTS)[None, :]
        moving = self.alive & (slot_ids != self.active_slot[:, None])
        delta = self.core.L.pos[:, None, :] - self.enemies.pos
        distance = np.linalg.norm(delta, axis=2)
        toward = delta / np.maximum(distance, 1e-6)[:, :, None]
        tangent = np.stack([-toward[:, :, 1], toward[:, :, 0]], axis=2)
        orbit_sign = np.where((slot_ids % 2) == 0, 1.0, -1.0).astype(np.float32)
        far_move = toward + 0.10 * tangent * orbit_sign[:, :, None]
        orbit_move = 0.55 * tangent * orbit_sign[:, :, None] + 0.08 * toward
        desired = np.where((distance > 2.55)[:, :, None], far_move, orbit_move)
        norm = np.maximum(np.linalg.norm(desired, axis=2, keepdims=True), 1e-6)
        desired = desired / norm
        slow = np.where(self.enemies.slow > 0.0, 0.72, 1.0)
        web = np.where(self.enemies.web > 0.0, 0.22, 1.0)
        speed = WALK * (0.76 + 0.24 * self.enemy_skill) * slow * web
        blocked_all = np.zeros(self.n, dtype=bool)
        pit_all = np.zeros(self.n, dtype=bool)
        for slot in range(MAX_OPPONENTS):
            mask = moving[:, slot]
            if not mask.any():
                continue
            old_pos = self.enemies.pos[:, slot].copy()
            proposed = old_pos + desired[:, slot, :] * (speed[:, slot] * DT)[:, None]
            moved, blocked, pit = self.core._move_with_terrain(old_pos, proposed)
            self.enemies.pos[:, slot] = np.where(mask[:, None], moved, old_pos)
            self.enemies.vel[:, slot] = np.where(
                mask[:, None], (self.enemies.pos[:, slot] - old_pos) / DT,
                self.enemies.vel[:, slot],
            )
            blocked_all |= mask & blocked
            pit_all |= mask & pit
        return blocked_all, pit_all

    def _resolve_collateral_blast(
        self, info: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply the learner's explosive AoE to non-target opponents as well."""
        tech = info["tech"]
        slot_ids = np.arange(MAX_OPPONENTS)[None, :]
        other = self.alive & (slot_ids != self.active_slot[:, None])
        target_pos = self.core.E.pos[:, None, :]
        distance = np.linalg.norm(self.enemies.pos - target_pos, axis=2)
        raw = np.zeros((self.n, MAX_OPPONENTS), dtype=np.float32)
        explosion = np.zeros(self.n, dtype=bool)
        for tech_id, power, falloff in (
            (T_CRYSTAL, 10.5, 3.0),
            (T_ANCHOR, 9.0, 2.5),
            (T_MINECART, 7.5, 2.0),
            (T_PEARL_CRYSTAL, 10.5, 3.0),
        ):
            rows = tech == tech_id
            if rows.any():
                radius = power / falloff
                raw = np.where(
                    rows[:, None] & other & (distance <= radius),
                    np.maximum(power - falloff * np.maximum(distance, 0.6), 0.0),
                    raw,
                )
                explosion |= rows
        if not explosion.any():
            zeros = np.zeros(self.n, dtype=np.float32)
            return zeros, zeros.astype(np.int32), zeros.astype(np.int32), zeros.astype(np.int32)

        enemies = self.enemies
        raw = raw * (enemies.hurt_cd <= 0.0)
        enemies.hurt_cd = np.where(raw > 0.0, 0.5, enemies.hurt_cd)
        reduction = enemies.armor_red() * 0.35
        reduction = np.where(
            enemies.cage_until > 0.0,
            np.minimum(0.92, reduction + 0.45),
            reduction,
        )
        mitigated = np.maximum(raw * (1.0 - reduction), 0.0)
        used_absorption = np.minimum(enemies.abs, mitigated)
        remaining = mitigated - used_absorption
        old_hp = enemies.hp.copy()
        new_hp = np.maximum(0.0, old_hp - remaining)
        popped = (new_hp <= DEATH) & enemies.has_totem & (raw > 0.0)
        enemies.has_totem = np.where(popped, False, enemies.has_totem)
        new_hp = np.where(popped, 11.0, new_hp)
        enemies.abs = np.where(
            popped,
            4.0,
            np.maximum(0.0, enemies.abs - used_absorption),
        )
        enemies.regen = np.where(popped, 4.5, enemies.regen)
        enemies.fire = np.where(popped, 0.0, enemies.fire)
        enemies.lava = np.where(popped, 0.0, enemies.lava)
        dealt = used_absorption + np.where(
            popped,
            old_hp,
            np.maximum(0.0, old_hp - new_hp),
        )
        enemies.hp = new_hp
        hit_count = ((raw > 0.0) & other).sum(axis=1).astype(np.int32)
        pop_count = popped.sum(axis=1).astype(np.int32)
        kill_count = ((old_hp > DEATH) & (new_hp <= DEATH) & other).sum(axis=1).astype(np.int32)
        return dealt.sum(axis=1), pop_count, kill_count, hit_count

    def _extra_attacks(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        slot_ids = np.arange(MAX_OPPONENTS)[None, :]
        others = self.alive & (slot_ids != self.active_slot[:, None])
        delta = self.core.L.pos[:, None, :] - self.enemies.pos
        distance = np.linalg.norm(delta, axis=2)
        self.enemies.cd[others] = np.maximum(0.0, self.enemies.cd[others] - DT)
        self.enemies.hurt_cd[others] = np.maximum(0.0, self.enemies.hurt_cd[others] - DT)
        attack = others & (distance <= REACH) & (self.enemies.cd <= 0.0)
        raw = np.where(attack, self.enemies.dmg() * (0.85 + 0.15 * self.enemy_skill), 0.0)
        # In Minecraft's short invulnerability window, only the strongest simultaneous
        # extra swing lands. This prevents an artificial n-times damage spike per frame.
        incoming = raw.max(axis=1)
        shield_break = (attack & self.enemies.is_axe()
                        & self.core.L.has_shield[:, None]
                        & self.core.L.shield_up[:, None])
        if shield_break.any():
            broke = shield_break.any(axis=1)
            self.core.L.shield_disabled = np.where(
                broke, np.maximum(self.core.L.shield_disabled, 1.6),
                self.core.L.shield_disabled,
            )
            self.core.L.shield_up = np.where(broke, False, self.core.L.shield_up)
        incoming = np.where(
            self.core.L.shield_up,
            incoming * np.where(attack.any(axis=1), 0.42, 1.0),
            incoming,
        )
        dealt, popped = self.core._damage(self.core.L, incoming)
        self.enemies.cd = np.where(attack, self.enemies.cd_time(), self.enemies.cd)
        return dealt, popped, attack.any(axis=1)

    def _obs(self) -> np.ndarray:
        base = self.core._obs()
        if base.shape[1] != TERRAIN_OBS_DIM:
            raise RuntimeError(f"Expected terrain observation ({TERRAIN_OBS_DIM},), got {base.shape}")
        distance = np.linalg.norm(
            self.enemies.pos - self.core.L.pos[:, None, :], axis=2
        )
        count = np.maximum(1, self.alive.sum(axis=1)).astype(np.float32)
        close_count = (self.alive & (distance <= REACH)).sum(axis=1).astype(np.float32)
        mean_distance = (np.where(self.alive, distance, 0.0).sum(axis=1) / count)
        mean_health = (np.where(self.alive, self.enemies.hp + self.enemies.abs, 0.0).sum(axis=1)
                       / count)
        other_mask = self.alive & (
            np.arange(MAX_OPPONENTS)[None, :] != self.active_slot[:, None]
        )
        nearest_other_slot = np.argmin(np.where(other_mask, distance, np.inf), axis=1)
        rows = np.arange(self.n)
        nearest_other_delta = self.enemies.pos[rows, nearest_other_slot] - self.core.L.pos
        nearest_other_delta = np.where(
            other_mask.any(axis=1)[:, None], nearest_other_delta, 0.0
        )
        crowd = np.stack([
            np.clip(self.alive.sum(axis=1) / MAX_OPPONENTS, 0.0, 1.0),
            np.clip(close_count / MAX_OPPONENTS, 0.0, 1.0),
            np.clip(mean_distance / 20.0, 0.0, 1.0),
            np.clip(mean_health / 20.0, 0.0, 1.5),
            np.clip(nearest_other_delta[:, 0] / 20.0, -1.0, 1.0),
            np.clip(nearest_other_delta[:, 1] / 20.0, -1.0, 1.0),
        ], axis=1).astype(np.float32)
        return np.concatenate([base, crowd], axis=1).astype(np.float32)

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.int32).reshape(self.n)
        rows = np.arange(self.n)
        active_alive = self.alive[rows, self.active_slot]
        self._retarget((~active_alive) & self.alive.any(axis=1))
        active_alive = self.alive[rows, self.active_slot]
        if not active_alive.all():
            # Terminal rows are still simulated until the VecEnv auto-resets them;
            # copying the last available slot is harmless for those finished rows.
            self._retarget(~active_alive & self.alive.any(axis=1))

        old_enemy_count = self.alive.sum(axis=1).astype(np.int32)
        other_blocked, other_pits = self._move_other_opponents()
        self._copy_slot_to_core(np.flatnonzero(self.alive.any(axis=1)))
        learner_alive_before = self.core.L.hp > DEATH
        obs, reward, _, _, _, info = self.core.step(actions)
        self._copy_core_to_slot(np.flatnonzero(self.alive.any(axis=1)))

        # Resolve explosive splash on the whole crowd, not just the currently selected
        # target. This is essential for teaching the actor to punish clustered enemies.
        aoe_damage, aoe_totem_pops, aoe_kills, aoe_hits = self._resolve_collateral_blast(info)
        self.alive &= self.enemies.hp > DEATH

        # The remaining opponents keep closing and can land independent melee swings.
        extra_damage, extra_totem, extra_attacked = self._extra_attacks()
        was_dead = info["learner_dead"].copy()
        learner_dead = self.core.L.hp <= DEATH
        has_opponents = self.alive.any(axis=1)
        enemy_dead = ~has_opponents
        final_crowd_kill = enemy_dead & (old_enemy_count > 0)
        reward = reward.copy()
        damage_penalty = 1.4 - 0.25 * self.player_aggression
        reward += aoe_damage * 2.5 + aoe_totem_pops * 8.0 + aoe_kills * 100.0
        reward += final_crowd_kill.astype(np.float32) * FULL_CROWD_WIN_BONUS
        reward -= extra_damage * damage_penalty
        reward -= (learner_alive_before & learner_dead).astype(np.float32) * DEATH_SHAPING_PENALTY
        # The core already prices an in-step death; this adds the matching base penalty
        # when the final blow came from a non-selected opponent after the core step.
        reward -= ((learner_dead & ~was_dead).astype(np.float32) * 100.0)
        reward -= extra_totem.astype(np.float32) * 2.0

        self.kills_this_fight += (old_enemy_count - self.alive.sum(axis=1)).astype(np.int32)
        self.extra_damage_taken += extra_damage
        self.extra_terrain_blocked += other_blocked.astype(np.int32)
        self.extra_pit_blocked += other_pits.astype(np.int32)

        target_lost = has_opponents & ~self.alive[rows, self.active_slot]
        self._retarget(target_lost)
        timeout = self.core.step_count >= MAX_STEPS
        terminated = (learner_dead | enemy_dead) & ~timeout
        truncated = timeout & ~terminated
        done = terminated | truncated

        if isinstance(info, dict):
            info = dict(info)
            info["learner_dead"] = learner_dead.copy()
            info["enemy_dead"] = enemy_dead.copy()
            info["timeout"] = truncated.copy()
            info["learner_hp"] = self.core.L.hp.copy()
            info["enemy_hp"] = np.where(
                has_opponents,
                np.min(np.where(self.alive, self.enemies.hp, np.inf), axis=1),
                0.0,
            ).astype(np.float32)
            info["damage_taken"] = info["damage_taken"] + extra_damage
            info["damage_dealt"] = info["damage_dealt"] + aoe_damage
            info["aoe_enemy_damage"] = aoe_damage.copy()
            info["aoe_enemy_hits"] = aoe_hits.copy()
            info["aoe_enemy_kills"] = aoe_kills.copy()
            info["aoe_enemy_totem_pops"] = aoe_totem_pops.copy()
            info["totem_popped_enemy"] = info["totem_popped_enemy"] | (aoe_totem_pops > 0)
            info["extra_damage_taken"] = extra_damage.copy()
            info["extra_opponent_attacked"] = extra_attacked.copy()
            info["opponent_count"] = self.enemy_count.copy()
            info["opponents_alive"] = self.alive.sum(axis=1).astype(np.int32)
            info["enemy_kills"] = self.kills_this_fight.copy()
            info["enemy_terrain_blocked"] = other_blocked.copy()
            info["enemy_pit_blocked"] = other_pits.copy()
            info["extra_terrain_blocked_total"] = self.extra_terrain_blocked.copy()
            info["extra_pit_blocked_total"] = self.extra_pit_blocked.copy()
            info["distance"] = np.linalg.norm(
                self.core.E.pos - self.core.L.pos, axis=1
            ).astype(np.float32)

        # Rebuild after retargeting so a continuing fight observes a live opponent.
        obs = self._obs()
        return obs, reward.astype(np.float32), terminated, truncated, done, info

    def snapshot_rows(self, mask: np.ndarray) -> dict[str, Any]:
        mask = np.asarray(mask, dtype=bool)
        owners = {
            "core": self.core,
            "learner": self.core.L,
            "enemy": self.core.E,
            "terrain": self.core.terrain,
            "roster": self.enemies,
            "arena": self,
        }
        snapshot: dict[str, Any] = {}
        for owner_name, owner in owners.items():
            arrays = {}
            for name, value in vars(owner).items():
                if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == self.n:
                    arrays[name] = value[mask].copy()
            snapshot[owner_name] = arrays
        snapshot["events"] = [copy.deepcopy(self.core.events[i]) for i in np.flatnonzero(mask)]
        snapshot["event_indices"] = np.flatnonzero(mask)
        return snapshot

    def restore_rows(self, mask: np.ndarray, snapshot: dict[str, Any]) -> None:
        mask = np.asarray(mask, dtype=bool)
        owners = {
            "core": self.core,
            "learner": self.core.L,
            "enemy": self.core.E,
            "terrain": self.core.terrain,
            "roster": self.enemies,
            "arena": self,
        }
        for owner_name, owner in owners.items():
            for name, value in snapshot[owner_name].items():
                getattr(owner, name)[mask] = value
        for index, event in zip(snapshot["event_indices"], snapshot["events"]):
            self.core.events[int(index)] = event


class MultiOpponentVecEnv(VecEnv):
    """Stable-Baselines-compatible vector wrapper with exact fight quotas."""

    render_mode = None

    def __init__(
        self,
        n_envs: int = 128,
        seed: int = 0,
        terrain_mix: float = 1.0,
        min_opponents: int = MIN_OPPONENTS,
        max_opponents: int = MAX_OPPONENTS,
    ):
        self.core = MultiOpponentArena(
            n_envs, seed=seed, terrain_mix=terrain_mix,
            min_opponents=min_opponents, max_opponents=max_opponents,
        )
        self._obs = self.core._obs()
        self._actions = np.zeros(n_envs, dtype=np.int32)
        self.episodes = 0
        self.fight_limit: int | None = None
        self.fight_quota: np.ndarray | None = None
        self.completed_by_env = np.zeros(n_envs, dtype=np.int32)
        self.wins = self.losses = self.draws = 0
        self.last_results: list[str] = []
        self.tech_counts = {name: 0 for name in TECHNIQUES if name != "none"}
        self.combo_breaks = 0
        self.combo_counters = 0
        self.combo_hits = 0
        self.enemy_kills = 0
        self.aoe_hits = 0
        self.aoe_kills = 0
        self.aoe_damage = 0.0
        self.opponents_total = 0
        self.opponents_histogram = np.zeros(MAX_OPPONENTS + 1, dtype=np.int64)
        self.terrain_blocked = 0
        self.pit_blocked = 0
        self.extra_terrain_blocked = 0
        self.extra_pit_blocked = 0
        super().__init__(n_envs, self.core.observation_space, self.core.action_space)

    def set_enemy_skill(self, value: float) -> None:
        self.core.set_enemy_skill(value)

    def set_aggression(self, value: float) -> None:
        self.core.set_aggression(value)

    def set_player_aggression(self, value: float) -> None:
        self.core.set_player_aggression(value)

    def set_combo_curriculum(self, share: float) -> None:
        self.core.set_combo_curriculum(share)

    def set_terrain_mix(self, value: float) -> None:
        self.core.set_terrain_mix(value)

    def set_opponent_cap(self, value: int) -> None:
        self.core.set_opponent_cap(value)

    def set_fight_limit(self, value: int | None) -> None:
        if value is None:
            self.fight_limit = None
            self.fight_quota = None
            self.completed_by_env.fill(0)
            return
        self.fight_limit = max(0, int(value))
        base, remainder = divmod(self.fight_limit, self.num_envs)
        self.fight_quota = np.full(self.num_envs, base, dtype=np.int32)
        self.fight_quota[:remainder] += 1
        self.completed_by_env.fill(0)
        self.episodes = 0

    def reset(self):
        if self.fight_quota is not None:
            self.completed_by_env.fill(0)
            self.episodes = 0
        self._obs = self.core.reset()
        return self._obs.copy()

    def step_async(self, actions):
        self._actions = np.asarray(actions, dtype=np.int32).reshape(self.num_envs)

    def step_wait(self):
        if self.fight_limit is not None and self.episodes >= self.fight_limit:
            return (self._obs.copy(), np.zeros(self.num_envs, dtype=np.float32),
                    np.zeros(self.num_envs, dtype=bool), [{} for _ in range(self.num_envs)])
        frozen = (self.completed_by_env >= self.fight_quota
                  if self.fight_quota is not None else np.zeros(self.num_envs, dtype=bool))
        snapshot = self.core.snapshot_rows(frozen) if bool(frozen.any()) else None
        obs, rewards, terminated, truncated, done, info = self.core.step(self._actions)
        if snapshot is not None:
            self.core.restore_rows(frozen, snapshot)
            obs[frozen] = self._obs[frozen]
            rewards[frozen] = 0.0
            terminated[frozen] = False
            truncated[frozen] = False
            done[frozen] = False

        active = ~frozen
        self.combo_breaks += int(np.count_nonzero(info["combo_broken"] & active))
        self.combo_counters += int(np.count_nonzero(info["combo_countered"] & active))
        self.combo_hits += int(np.count_nonzero(info["combo_hit"] & active))
        self.terrain_blocked += int(np.count_nonzero(info["terrain_blocked"] & active))
        self.pit_blocked += int(np.count_nonzero(info["pit_blocked"] & active))
        self.extra_terrain_blocked += int(np.count_nonzero(info["enemy_terrain_blocked"] & active))
        self.extra_pit_blocked += int(np.count_nonzero(info["enemy_pit_blocked"] & active))
        self.aoe_hits += int(info["aoe_enemy_hits"][active].sum())
        self.aoe_kills += int(info["aoe_enemy_kills"][active].sum())
        self.aoe_damage += float(info["aoe_enemy_damage"][active].sum())

        infos: list[dict] = [{} for _ in range(self.num_envs)]
        done_idx = np.flatnonzero(done & active)
        if len(done_idx):
            row_mask = np.zeros(self.num_envs, dtype=bool)
            for i in done_idx:
                won = bool(info["enemy_dead"][i]) and not bool(info["learner_dead"][i])
                lost = bool(info["learner_dead"][i]) and not bool(info["enemy_dead"][i])
                self.episodes += 1
                if self.fight_quota is not None:
                    self.completed_by_env[i] += 1
                if won:
                    self.wins += 1
                    self.last_results.append("W")
                elif lost:
                    self.losses += 1
                    self.last_results.append("L")
                else:
                    self.draws += 1
                    self.last_results.append("T")
                if len(self.last_results) > 200:
                    self.last_results.pop(0)
                opponents = int(info["opponent_count"][i])
                self.opponents_total += opponents
                if 0 <= opponents < len(self.opponents_histogram):
                    self.opponents_histogram[opponents] += 1
                self.enemy_kills += int(info["enemy_kills"][i])
                for t in range(1, len(TECHNIQUES)):
                    count = int(self.core.core.consumed[i, t])
                    if count:
                        self.tech_counts[TECHNIQUES[t]] += count
                        self.core.core.consumed[i, t] = 0
                infos[i] = {
                    "TimeLimit.truncated": bool(truncated[i]),
                    "terminal_observation": obs[i].copy(),
                    "learner_dead": bool(info["learner_dead"][i]),
                    "enemy_dead": bool(info["enemy_dead"][i]),
                    "timeout": bool(truncated[i]),
                    "learner_hp": float(info["learner_hp"][i]),
                    "enemy_hp": float(info["enemy_hp"][i]),
                    "opponent_count": opponents,
                    "opponents_alive": int(info["opponents_alive"][i]),
                    "enemy_kills": int(info["enemy_kills"][i]),
                    "extra_damage_taken": float(info["extra_damage_taken"][i]),
                    "terrain_kind": int(info["terrain_kind"][i]),
                    "combo_broken": bool(info["combo_broken"][i]),
                    "combo_countered": bool(info["combo_countered"][i]),
                    "combo_hit": bool(info["combo_hit"][i]),
                }
                row_mask[i] = True
            obs[row_mask] = self.core.reset(row_mask)[row_mask]
        self._obs = obs
        return self._obs.copy(), rewards.copy(), done.copy(), infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self.core, attr_name, None) for _ in range(self.num_envs)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.core, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        method = getattr(self.core, method_name)
        return [method(*method_args, **method_kwargs) for _ in range(self.num_envs)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in range(self.num_envs)]
