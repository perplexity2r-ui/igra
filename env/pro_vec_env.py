"""PRO Minecraft 1.16.5 PvP arena — full technique set, vectorised for 500k-fight training.

New meta techniques on top of the randomizer arena (`env/sim_env.py`):
  * **End Crystal** detonation (obsidian base + crystal) — enormous AoE, self-damage if too close.
  * **Respawn Anchor** charged blast (glowstone) — fast, short-radius, huge damage.
  * **TNT Minecart** (rail + minecart + TNT) — long-range explosive with heavy knockback.
  * **Obsidian blast cage** — walls up to absorb explosion damage & grant resist.
  * **W-tap sprint-reset combo** — bonus damage, knockback cancel, faster follow-up.
  * **Pearl + crystal combo** — blink into the enemy and detonate instantly.

The opponent is a **professional scripted pro player** (skill-scaled): it kites at crystal
distance, shield-times, axe-breaks shields, cages against blasts, drinks, traps, pearls in
and executes full explosive combos — not a random walker anymore.

Observation `Box(30,)`, action `Discrete(28)`. Same protocol as `bot/bot.js`.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from .sim_env import ARMOR_TIERS, BLOCK_TYPES, MELEE_POOL, RANGED_TYPES

N_ACTIONS = 28
OBS_DIM = 30
DT = 0.10
WALK = 4.317
SPRINT = 5.612
REACH = 3.25
ARENA_BOUND = 9.2
DEATH = 6.0
MAX_OBSTACLES = 6
MAX_STEPS = 600

WEAPON_DMG = np.array([w.damage for w in MELEE_POOL], dtype=np.float32)
WEAPON_CD = np.array([w.cooldown for w in MELEE_POOL], dtype=np.float32)
WEAPON_KB = np.array([w.kb_mult for w in MELEE_POOL], dtype=np.float32)
WEAPON_AXE = np.array([1.0 if w.shield_break else 0.0 for w in MELEE_POOL], dtype=np.float32)
WEAPON_FIRE = np.array([w.fire_seconds for w in MELEE_POOL], dtype=np.float32)
ARMOR_RED = np.array([a.reduction for a in ARMOR_TIERS], dtype=np.float32)
RANGED_DMG = np.array([r.damage for r in RANGED_TYPES], dtype=np.float32)
RANGED_KB = np.array([r.knockback for r in RANGED_TYPES], dtype=np.float32)
RANGED_SLOW = np.array([r.slow_sec for r in RANGED_TYPES], dtype=np.float32)
RANGED_PIERCE = np.array([1.0 if r.pierces_armor else 0.0 for r in RANGED_TYPES], dtype=np.float32)

# technique ids recorded for logging
TECHNIQUES = [
    "none", "golden_apple", "instant_heal", "ender_pearl", "buff_potion", "ranged",
    "block_barricade", "cobweb_trap", "lava_trap", "water_bucket", "pillar_jump",
    "end_crystal", "respawn_anchor", "tnt_minecart", "obsidian_cage", "wtap_combo",
    "pearl_crystal", "shield_block",
]
T_GAP, T_HEAL, T_PEARL, T_BUFF, T_RANGED = 1, 2, 3, 4, 5
T_BARRICADE, T_COBWEB, T_LAVA, T_WATER, T_PILLAR = 6, 7, 8, 9, 10
T_CRYSTAL, T_ANCHOR, T_MINECART, T_CAGE, T_WTAP, T_PEARL_CRYSTAL = 11, 12, 13, 14, 15, 16


class BotArrays:
    """All per-bot state as numpy arrays of shape (n,)."""

    def __init__(self, n: int):
        f = np.float32
        self.hp = np.full(n, 20.0, f)
        self.abs = np.zeros(n, f)
        self.cd = np.zeros(n, f)
        self.hurt_cd = np.zeros(n, f)
        self.regen = np.zeros(n, f)
        self.buff = np.zeros(n, f)
        self.fire = np.zeros(n, f)
        self.slow = np.zeros(n, f)
        self.web = np.zeros(n, f)
        self.lava = np.zeros(n, f)
        self.pillar = np.zeros(n, f)
        self.pearl_cd = np.zeros(n, f)
        self.use_lock = np.zeros(n, f)
        self.shield_disabled = np.zeros(n, f)
        self.shield_up = np.zeros(n, bool)
        self.shield_held = np.zeros(n, bool)  # current offhand block while guarding
        self.has_shield = np.zeros(n, bool)
        self.has_totem = np.zeros(n, bool)
        self.cage_until = np.zeros(n, f)

        self.weapon = np.zeros(n, np.int32)
        self.armor = np.zeros(n, np.int32)
        self.ranged_type = np.zeros(n, np.int32)

        self.blocks = np.zeros(n, f)
        self.cobwebs = np.zeros(n, f)
        self.lava_bk = np.zeros(n, f)
        self.water_bk = np.zeros(n, f)
        self.gaps = np.zeros(n, f)
        self.splashes = np.zeros(n, f)
        self.pearls = np.zeros(n, f)
        self.buffs = np.zeros(n, f)
        self.ranged = np.zeros(n, f)
        self.crystals = np.zeros(n, f)
        self.anchors = np.zeros(n, f)
        self.minecarts = np.zeros(n, f)
        self.obsidian = np.zeros(n, f)

        self.pos = np.zeros((n, 2), np.float32)
        self.vel = np.zeros((n, 2), np.float32)
        self.y = np.zeros(n, f)
        self.vy = np.zeros(n, f)

    # convenience derived values -------------------------------------------------
    def dmg(self) -> np.ndarray:
        return WEAPON_DMG[self.weapon]

    def cd_time(self) -> np.ndarray:
        return WEAPON_CD[self.weapon]

    def kb_mult(self) -> np.ndarray:
        return WEAPON_KB[self.weapon]

    def is_axe(self) -> np.ndarray:
        return WEAPON_AXE[self.weapon] > 0.5

    def fire_sec(self) -> np.ndarray:
        return WEAPON_FIRE[self.weapon]

    def armor_red(self) -> np.ndarray:
        return ARMOR_RED[self.armor]

    def ranged_spec(self):
        return (
            RANGED_DMG[self.ranged_type],
            RANGED_KB[self.ranged_type],
            RANGED_SLOW[self.ranged_type],
            RANGED_PIERCE[self.ranged_type] > 0.5,
        )

    def reset_mask(self, mask: np.ndarray) -> None:
        for arr in (
            self.cd, self.hurt_cd, self.regen, self.buff, self.fire, self.slow, self.web,
            self.lava, self.pillar, self.pearl_cd, self.use_lock, self.shield_disabled,
            self.y, self.vy, self.blocks, self.cobwebs, self.lava_bk, self.water_bk,
            self.gaps, self.splashes, self.pearls, self.buffs, self.ranged,
            self.crystals, self.anchors, self.minecarts, self.obsidian, self.cage_until,
        ):
            arr[mask] = 0.0
        self.hp[mask] = 20.0
        self.abs[mask] = 0.0
        self.shield_up[mask] = False
        self.shield_held[mask] = False
        self.has_shield[mask] = False
        self.has_totem[mask] = False
        self.vel[mask] = 0.0


class ProArena:
    """Vectorised PRO PvP arena. `n` independent fights simulated in lock-step."""

    def __init__(self, n: int = 1, seed: int | None = None, difficulty: float = 1.0):
        self.n = int(n)
        self.rng = np.random.default_rng(seed)
        self.difficulty = np.full(self.n, float(difficulty), np.float32)
        # training wheels for the *opponent*: ramp from beatable to full pro
        self.e_skill = np.full(self.n, float(difficulty), np.float32)
        self.L = BotArrays(self.n)
        self.E = BotArrays(self.n)
        self.step_count = np.zeros(self.n, np.int32)
        self.t = np.zeros(self.n, np.float32)
        self.obstacles = np.zeros((self.n, MAX_OBSTACLES, 2), np.float32)
        self.obstacle_ttl = np.zeros((self.n, MAX_OBSTACLES), np.float32)
        self.obstacle_n = np.zeros(self.n, np.int32)
        self.consumed = np.zeros((self.n, len(TECHNIQUES)), np.int32)
        self.last_tech = np.zeros(self.n, np.int32)
        self.events = [dict() for _ in range(self.n)]
        self.reset(np.ones(self.n, bool))

    # ------------------------------------------------------------------ loadouts
    def _pick_diff(self, n: int, hi: int, avoid: np.ndarray) -> np.ndarray:
        """n random ints in [0, hi) that never equal the matching `avoid` entry."""
        out = self.rng.integers(0, hi, n)
        bad = out == avoid
        tries = 0
        while bad.any() and tries < 8:
            out[bad] = self.rng.integers(0, hi, int(bad.sum()))
            bad = out == avoid
            tries += 1
        return out

    def _roll(self, B: BotArrays, mask: np.ndarray, avoid_w: np.ndarray, avoid_a: np.ndarray,
              avoid_r: np.ndarray, pro: bool) -> None:
        n = int(mask.sum())
        if n == 0:
            return
        r = self.rng
        diff = self.difficulty[mask]
        w = self._pick_diff(n, len(MELEE_POOL), avoid_w[mask])
        a = self._pick_diff(n, len(ARMOR_TIERS), avoid_a[mask])
        rr = self._pick_diff(n, len(RANGED_TYPES), avoid_r[mask])
        if pro:
            # Professionals favour the strong half of the arsenal & heavy armour.
            strong = np.array([6, 7, 8, 9, 10, 11], np.int32)
            fav = r.random(n) < (0.15 + 0.25 * diff)
            w = np.where(fav, strong[r.integers(0, len(strong), n)], w)
            bad = w == avoid_w[mask]
            if bad.any():
                w[bad] = self._pick_diff(int(bad.sum()), len(MELEE_POOL), avoid_w[mask][bad])
            afav = r.random(n) < (0.35 + 0.25 * diff)
            a = np.where(afav, r.integers(2, 5, n), a)
            bad = a == avoid_a[mask]
            if bad.any():
                a[bad] = self._pick_diff(int(bad.sum()), len(ARMOR_TIERS), avoid_a[mask][bad])

        blocks = np.where(r.random(n) < 0.85, r.integers(4, 25, n), 0).astype(np.float32)
        cobwebs = np.where(r.random(n) < 0.55, r.integers(1, 5, n), 0).astype(np.float32)
        lava_bk = (r.random(n) < 0.45).astype(np.float32)
        water_bk = (r.random(n) < 0.50).astype(np.float32)
        has_shield = r.random(n) < 0.55
        has_totem = (~has_shield) & (r.random(n) < 0.20)

        heal_roll = r.integers(0, 5, n)
        primary = r.random(n) < 0.75
        gaps = np.where(heal_roll == 3,
                        np.where(primary, r.integers(1, 4, n), 0),
                        np.where(r.random(n) < 0.45, r.integers(1, 3, n), 0)).astype(np.float32)
        splashes = np.where(heal_roll != 3,
                            np.where(primary, r.integers(1, 4, n), 0),
                            np.where(r.random(n) < 0.45, r.integers(1, 3, n), 0)).astype(np.float32)
        pearls = np.where(r.random(n) < 0.55, r.integers(1, 4, n), 0).astype(np.float32)
        buffs = np.where(r.random(n) < 0.50, r.integers(1, 3, n), 0).astype(np.float32)
        rcnt = np.zeros(n, np.float32)
        for i in range(len(RANGED_TYPES)):
            sel = rr == i
            if sel.any():
                rcnt[sel] = r.integers(RANGED_TYPES[i].min_count,
                                       RANGED_TYPES[i].max_count + 1, int(sel.sum()))
        ranged = np.where(r.random(n) < 0.75, rcnt, 0).astype(np.float32)

        crystals = np.where(r.random(n) < 0.62, r.integers(1, 7, n), 0).astype(np.float32)
        obsidian = np.where(r.random(n) < 0.78, r.integers(4, 17, n), 0).astype(np.float32)
        anchors = np.where(r.random(n) < 0.42, r.integers(1, 4, n), 0).astype(np.float32)
        minecarts = np.where(r.random(n) < 0.46, r.integers(1, 5, n), 0).astype(np.float32)

        B.weapon[mask] = w
        B.armor[mask] = a
        B.ranged_type[mask] = rr
        B.blocks[mask] = blocks
        B.cobwebs[mask] = cobwebs
        B.lava_bk[mask] = lava_bk
        B.water_bk[mask] = water_bk
        B.has_shield[mask] = has_shield
        B.has_totem[mask] = has_totem
        B.gaps[mask] = gaps
        B.splashes[mask] = splashes
        B.pearls[mask] = pearls
        B.buffs[mask] = buffs
        B.ranged[mask] = ranged
        B.crystals[mask] = crystals
        B.obsidian[mask] = obsidian
        B.anchors[mask] = anchors
        B.minecarts[mask] = minecarts

    def reset(self, mask: np.ndarray | None = None) -> np.ndarray:
        if mask is None:
            mask = np.ones(self.n, bool)
        n = self.n
        r = self.rng
        self.L.reset_mask(mask)
        self.E.reset_mask(mask)
        self.step_count[mask] = 0
        self.t[mask] = 0.0
        self.obstacle_ttl[mask] = 0.0
        self.obstacle_n[mask] = 0
        self.consumed[mask] = 0
        self.last_tech[mask] = 0
        for i in np.flatnonzero(mask):
            self.events[i] = {}

        idx = np.flatnonzero(mask)
        k = len(idx)
        if k == 0:
            return self._obs()
        ang = r.uniform(0.0, 2.0 * np.pi, k)
        sep = r.uniform(4.0, 10.5, k)
        centre = r.uniform(-2.5, 2.5, (k, 2))
        off = np.stack([np.cos(ang), np.sin(ang)], 1) * (sep * 0.5)[:, None]
        lp = np.clip(centre - off, -ARENA_BOUND, ARENA_BOUND).astype(np.float32)
        ep = np.clip(centre + off, -ARENA_BOUND, ARENA_BOUND).astype(np.float32)
        self.L.pos[mask] = lp
        self.E.pos[mask] = ep

        sub = np.zeros(self.n, bool)
        sub[idx] = True
        self._roll(self.L, sub, np.full(self.n, -1, np.int32), np.full(self.n, -1, np.int32),
                   np.full(self.n, -1, np.int32), pro=False)
        # Enemy is a professional: harder loadouts & mirrored asymmetry
        self._roll(self.E, sub, self.L.weapon.copy(), self.L.armor.copy(),
                   self.L.ranged_type.copy(), pro=True)
        return self._obs()

    def set_difficulty(self, value: float) -> None:
        self.difficulty[:] = float(value)
        self.e_skill[:] = float(value)

    def set_enemy_skill(self, value: float) -> None:
        self._skill = float(value)
        self.difficulty[:] = float(value)

    # ---------------------------------------------------------------- helpers
    def _speed_mult(self, B: BotArrays) -> np.ndarray:
        m = np.ones(self.n, np.float32)
        m = np.where(B.buff > 0, m * 1.35, m)
        m = np.where(B.slow > 0, m * 0.72, m)
        m = np.where(B.web > 0, m * 0.22, m)
        m = np.where(B.lava > 0, m * 0.50, m)
        return m

    def _damage(self, B: BotArrays, raw: np.ndarray, bypass=False,
                blast=False, iframes: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Apply raw damage with vanilla 10-tick invulnerability frames; returns (dealt, pop)."""
        raw = np.maximum(np.asarray(raw, np.float32), 0.0)
        if iframes:
            raw = raw * (B.hurt_cd <= 0.0)
            B.hurt_cd = np.where(raw > 0.0, 0.5, B.hurt_cd)
        bypass = np.asarray(bypass, np.float32) * np.ones(self.n, np.float32)
        blast = np.asarray(blast, np.float32) * np.ones(self.n, np.float32)
        factor = np.where(bypass > 0.5, 0.0, np.where(blast > 0.5, 0.35, 0.78))
        red = B.armor_red() * factor
        red = np.where(B.cage_until > 0.0, np.minimum(0.92, red + 0.45), red)
        red = np.where((blast > 0.5) & B.shield_up & (B.shield_disabled <= 0.0),
                       np.minimum(0.95, red + 0.25), red)
        mit = np.maximum(raw * (1.0 - red), 0.0)
        used_abs = np.minimum(B.abs, mit)
        rem = mit - used_abs
        old_hp = B.hp
        new_hp = np.maximum(0.0, old_hp - rem)
        pop = (new_hp <= DEATH) & B.has_totem
        B.has_totem = np.where(pop, False, B.has_totem)
        new_hp = np.where(pop, 11.0, new_hp)
        B.abs = np.where(pop, 4.0, np.maximum(0.0, B.abs - used_abs))
        B.regen = np.where(pop, 4.5, B.regen)
        B.fire = np.where(pop, 0.0, B.fire)
        B.lava = np.where(pop, 0.0, B.lava)
        dealt = used_abs + np.where(pop, old_hp, np.maximum(0.0, old_hp - new_hp))
        B.hp = new_hp
        return dealt, pop

    def _explode(self, target: BotArrays, caster: BotArrays, power: float, falloff: float,
                 self_power: float, self_falloff: float, armor_kind: str = "blast"):
        """AoE explosion at the target's feet. Returns (dmg_to_target, dmg_to_caster)."""
        d_t = np.linalg.norm(target.pos - caster.pos, axis=1)
        # crystal / anchor sits ~0.6 blocks from the target
        dmg_t = np.maximum(power - falloff * 0.6, 0.0)
        dmg_c = np.maximum(self_power - self_falloff * d_t, 0.0)
        cage_t = target.cage_until > 0.0
        dmg_t = np.where(cage_t, dmg_t * 0.35, dmg_t)
        cage_c = caster.cage_until > 0.0
        dmg_c = np.where(cage_c, dmg_c * 0.35, dmg_c)
        d_tgt, _ = self._damage(target, dmg_t, blast=True)
        d_self, _ = self._damage(caster, dmg_c, blast=True)
        hit_dir = target.pos - caster.pos
        norm = np.maximum(np.linalg.norm(hit_dir, axis=1, keepdims=True), 1e-6)
        target.vel += (hit_dir / norm) * 5.5
        return d_tgt, d_self, dmg_t, dmg_c, dmg_c.sum()

    def _obstacle_between(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        seg = p2 - p1
        seg_len = np.maximum(np.linalg.norm(seg, axis=1, keepdims=True), 1e-6)
        res = np.zeros(self.n, bool)
        for s in range(MAX_OBSTACLES):
            live = self.obstacle_ttl[:, s] > 0
            ob = self.obstacles[:, s, :]
            t = np.clip(((ob - p1) * seg).sum(1, keepdims=True) / (seg_len ** 2), 0.0, 1.0)
            proj = p1 + seg * t
            res |= live & (np.linalg.norm(ob - proj, axis=1) <= 1.25)
        return res

    def _add_obstacle(self, mid: np.ndarray, ttl: float = 12.0) -> None:
        free = self.obstacle_n < MAX_OBSTACLES
        if not free.any():
            return
        slot = self.obstacle_n.copy()
        rows = np.flatnonzero(free)
        self.obstacles[rows, slot[rows]] = mid[rows]
        self.obstacle_ttl[rows, slot[rows]] = ttl
        self.obstacle_n[rows] += 1

    def _push_out(self, pos: np.ndarray) -> np.ndarray:
        out = pos.copy()
        for s in range(MAX_OBSTACLES):
            live = self.obstacle_ttl[:, s] > 0
            ob = self.obstacles[:, s, :]
            diff = out - ob
            d = np.linalg.norm(diff, axis=1)
            push = live & (d < 0.95)
            if push.any():
                pdir = diff / np.maximum(d, 1e-5)[:, None]
                pdir[push & (d <= 1e-5)] = np.array([1.0, 0.0], np.float32)
                out[push] = ob[push] + pdir[push] * 0.95
        return np.clip(out, -ARENA_BOUND, ARENA_BOUND)

    # ---------------------------------------------------------------- observation
    def _obs(self) -> np.ndarray:
        L, E = self.L, self.E
        d = E.pos - L.pos
        dist = np.linalg.norm(d, axis=1)
        obs = np.stack([
            np.clip((L.hp + L.abs) / 20.0, 0, 1.5),
            np.clip((E.hp + E.abs) / 20.0, 0, 1.5),
            np.clip(dist / 20.0, 0, 1),
            np.clip(np.linalg.norm(L.vel, axis=1) * 0.05, 0, 2),
            np.clip(np.linalg.norm(E.vel, axis=1) * 0.05, 0, 2),
            np.clip(d[:, 0] / 20.0, -1, 1),
            np.clip(d[:, 1] / 20.0, -1, 1),
            (L.y <= 1e-4).astype(np.float32),
            (L.cd <= 0.0).astype(np.float32),
            np.clip(self.step_count / MAX_STEPS, 0, 1),
            np.clip(L.dmg() / 14.0, 0, 1),
            np.clip(L.cd_time() / 1.2, 0, 1),
            np.clip(E.dmg() / 14.0, 0, 1),
            np.clip(L.armor_red(), 0, 1),
            np.clip(E.armor_red(), 0, 1),
            np.clip(L.gaps / 5.0, 0, 1),
            np.clip(L.splashes / 5.0, 0, 1),
            np.clip(L.pearls / 4.0, 0, 1),
            np.clip((L.buffs + L.ranged) / 12.0, 0, 1),
            ((L.has_shield & L.shield_up & (L.shield_disabled <= 0)).astype(np.float32)),
            ((E.has_shield & E.shield_up & (E.shield_disabled <= 0)).astype(np.float32)),
            np.clip(L.blocks / 24.0, 0, 1),
            np.clip((L.cobwebs + L.lava_bk + L.water_bk) / 6.0, 0, 1),
            np.clip(self.obstacle_n / 6.0, 0, 1),
            np.clip(L.crystals / 6.0, 0, 1),
            np.clip(L.anchors / 4.0, 0, 1),
            np.clip(L.minecarts / 4.0, 0, 1),
            np.clip((E.crystals + E.anchors + E.minecarts) / 10.0, 0, 1),
            np.clip(L.obsidian / 16.0, 0, 1),
            (L.cage_until > 0.0).astype(np.float32),
        ], axis=1).astype(np.float32)
        return obs

    # ---------------------------------------------------------------- enemy brain
    def _enemy_brain(self, act: np.ndarray, dmg_dealt: np.ndarray, dmg_taken: np.ndarray,
                     healed: np.ndarray, pos_delta_enemy: np.ndarray) -> dict:
        """Professional scripted opponent. Returns per-env event flags."""
        L, E = self.L, self.E
        r = self.rng
        diff = self.e_skill
        vec = L.pos - E.pos
        dist = np.linalg.norm(vec, axis=1)
        fwd = vec / np.maximum(dist, 1e-6)[:, None]
        e_move = np.zeros((self.n, 2), np.float32)
        acted = np.zeros(self.n, bool)
        info = {}
        # pro action cooldown: explosives/utilities can't fire every single tick
        utility_ready = E.pearl_cd <= 0.0

        def chance(p):
            return r.random(self.n) < (p * (0.45 + 0.55 * diff))

        # 1. emergency heal
        m = (~acted) & (E.hp <= 13.0) & (E.splashes > 0) & chance(0.5)
        if m.any():
            E.splashes = np.where(m, E.splashes - 1, E.splashes)
            before = E.hp
            E.hp = np.where(m, np.minimum(20.0, E.hp + 7.5), E.hp)
            healed += np.where(m, E.hp - before, 0.0)
            acted |= m
            e_move = np.where(m[:, None], -fwd * 0.6, e_move)
        m = (~acted) & (E.hp <= 15.0) & (E.gaps > 0) & (E.regen <= 0) & chance(0.45)
        if m.any():
            E.gaps = np.where(m, E.gaps - 1, E.gaps)
            E.regen = np.where(m, 5.0, E.regen)
            E.abs = np.where(m, np.maximum(E.abs, 4.0), E.abs)
            acted |= m
        # 2. pro: pull out axe and crack the learner's shield
        m = ((~acted) & (L.has_shield & L.shield_up & (L.shield_disabled <= 0))
             & (dist <= 3.45) & (E.cd <= 0.0) & E.is_axe() & chance(0.85))
        if m.any():
            L.shield_up = np.where(m, False, L.shield_up)
            L.shield_disabled = np.where(m, 5.0, L.shield_disabled)
            L.vel -= fwd * 2.0
            E.cd = np.where(m, E.cd_time(), E.cd)
            acted |= m
            info["e_shield_break"] = m
        # 3. crystal trap (signature pro move): detonate at learner, stay outside blast
        m = ((~acted) & utility_ready & (E.crystals > 0) & (E.obsidian > 0) & (dist >= 2.55) & (dist <= 5.2)
             & (E.hp > 9.5) & (L.hp + L.abs <= 17.0) & chance(0.18 + 0.3 * diff))
        if m.any():
            E.crystals = np.where(m, E.crystals - 1, E.crystals)
            E.obsidian = np.where(m, E.obsidian - 1, E.obsidian)
            dtg, dsf, raw_t, raw_c, _ = self._explode(L, E, 7.0, 2.2, 24.0, 7.0)
            dmg_taken += np.where(m, dtg, 0.0)
            E.pearl_cd = np.where(m, 2.2, E.pearl_cd)
            acted |= m
            e_move = np.where(m[:, None], -fwd * 0.5, e_move)
            info["e_crystal"] = m
        # 4. respawn anchor
        m = ((~acted) & utility_ready & (E.anchors > 0) & (dist >= 1.6) & (dist <= 4.6)
             & (L.hp + L.abs <= 18.0) & chance(0.20 + 0.3 * diff))
        if m.any():
            E.anchors = np.where(m, E.anchors - 1, E.anchors)
            dtg, dsf, *_ = self._explode(L, E, 6.2, 1.8, 20.0, 6.0)
            dmg_taken += np.where(m, dtg, 0.0)
            E.pearl_cd = np.where(m, 2.0, E.pearl_cd)
            acted |= m
            info["e_anchor"] = m
        # 5. TNT minecart artillery
        m = ((~acted) & utility_ready & (E.minecarts > 0) & (dist >= 3.0) & (dist <= 8.5)
             & (L.hp + L.abs <= 19.0) & chance(0.16 + 0.3 * diff))
        if m.any():
            E.minecarts = np.where(m, E.minecarts - 1, E.minecarts)
            dtg, dsf, *_ = self._explode(L, E, 5.4, 1.6, 17.0, 5.0)
            dmg_taken += np.where(m, dtg, 0.0)
            E.pearl_cd = np.where(m, 2.0, E.pearl_cd)
            acted |= m
            L.slow = np.where(m, np.maximum(L.slow, 1.2), L.slow)
            info["e_minecart"] = m
        # 6. ranged poke
        m = (~acted) & (E.ranged > 0) & (dist >= 4.5) & (dist <= 13.5) & (E.cd <= 0.0) & chance(0.16 + 0.25 * diff)
        if m.any():
            rdm, rkb, rsl, rpierce = E.ranged_spec()
            blocked = L.has_shield & L.shield_up & (L.shield_disabled <= 0) & (~rpierce)
            dtg, _ = self._damage(L, np.where(m, rdm, 0.0), bypass=rpierce)
            dtg = np.where(m & ~blocked, dtg, 0.0)
            dmg_taken += dtg
            L.slow = np.where(m & ~blocked, np.maximum(L.slow, rsl), L.slow)
            L.vel -= fwd * np.where(m & ~blocked, rkb, 0.0)[:, None]
            E.ranged = np.where(m, E.ranged - 1, E.ranged)
            E.cd = np.where(m, 0.9, E.cd)
            acted |= m
            info["e_ranged"] = m
        # 7. pearl gap-close
        m = ((~acted) & (E.pearls > 0) & (E.pearl_cd <= 0) & (dist > 7.0) & (E.hp > 12.0)
             & chance(0.35))
        if m.any():
            E.pearls = np.where(m, E.pearls - 1, E.pearls)
            E.pearl_cd = np.where(m, 2.0, E.pearl_cd)
            E.pos = np.where(m[:, None], np.clip(L.pos + fwd * 2.2, -ARENA_BOUND, ARENA_BOUND),
                             E.pos)
            _, _ = self._damage(E, np.where(m, 1.5, 0.0), bypass=True)
            E.web = np.where(m, 0.0, E.web)
            acted |= m
        # 8. buff
        m = (~acted) & (E.buffs > 0) & (E.buff <= 0.0) & (dist > 4.0) & chance(0.3)
        if m.any():
            E.buffs = np.where(m, E.buffs - 1, E.buffs)
            E.buff = np.where(m, 15.0, E.buff)
            acted |= m
        # 8b. pro defence: if learner is loaded with explosives, keep a shield up or cage
        m = ((~acted) & (E.has_shield) & (E.shield_disabled <= 0.0)
             & (L.crystals + L.anchors + L.minecarts > 0) & (dist <= 5.0)
             & (r.random(self.n) < 0.45 * diff))
        if m.any():
            E.shield_up = np.where(m, True, E.shield_up)
            acted |= m
            e_move = np.where(m[:, None], -fwd * 0.5, e_move)
        # 9. panic cage when learner is holding explosives
        m = ((~acted) & (E.hp <= 14.0) & (E.obsidian >= 4)
             & (L.crystals + L.anchors + L.minecarts > 0)
             & (L.cd <= 0.2) & chance(0.25 + 0.35 * diff))
        if m.any():
            E.obsidian = np.where(m, E.obsidian - 4, E.obsidian)
            E.cage_until = np.where(m, 4.0, E.cage_until)
            acted |= m
            info["e_cage"] = m
        # 10. cobweb / lava trap
        m = (~acted) & (E.cobwebs > 0) & (dist <= 3.8) & (L.web <= 0) & chance(0.25)
        if m.any():
            E.cobwebs = np.where(m, E.cobwebs - 1, E.cobwebs)
            L.web = np.where(m, 2.6, L.web)
            acted |= m
        m = (~acted) & (E.lava_bk > 0) & (dist >= 2.1) & (dist <= 4.2) & chance(0.25)
        if m.any():
            E.lava_bk = np.where(m, E.lava_bk - 1, E.lava_bk)
            L.lava = np.where(m, 1.6, L.lava)
            L.fire = np.where(m, np.maximum(L.fire, 3.5), L.fire)
            acted |= m
        # 11. defensive barricade
        m = ((~acted) & (E.hp <= 14.0) & (E.blocks >= 2) & (dist < 5.0)
             & (self.obstacle_n < MAX_OBSTACLES) & chance(0.35))
        if m.any():
            E.blocks = np.where(m, E.blocks - 2, E.blocks)
            mid = np.clip((L.pos + E.pos) * 0.5, -8.0, 8.0)
            self._add_obstacle(mid, 14.0)
            e_move = np.where(m[:, None], -fwd * 0.7, e_move)
            acted |= m

        # ---- movement: pro keeps crystal distance, strafes, kites, jumps for crits
        want_close = (dist > 3.05) & (~E.shield_up)
        e_move += fwd * want_close[:, None]
        back = (dist < 2.15) | ((E.hp <= 8.0) & (dist < 3.5))
        e_move -= fwd * back[:, None] * 0.85
        strafe = np.where(r.random(self.n) < 0.5, 1.0, -1.0).astype(np.float32)
        right = np.stack([-fwd[:, 1], fwd[:, 0]], 1)
        active = np.linalg.norm(e_move, axis=1) > 0.05
        e_move += right * (strafe * active * 0.55)[:, None]
        speed = WALK * self._speed_mult(E)
        speed = np.where(dist > 4.5, SPRINT * self._speed_mult(E), speed)
        norm = np.maximum(np.linalg.norm(e_move, axis=1, keepdims=True), 1.0)
        e_move = e_move / norm

        # shield timing: pro raises shield when the learner's swing is ready and close
        can_shield = E.has_shield & (E.shield_disabled <= 0.0) & (~acted) & (dist < 3.5)
        raise_shield = can_shield & (L.cd <= 0.25) & (r.random(self.n) < 0.55 * diff)
        E.shield_up = raise_shield
        e_move = np.where(raise_shield[:, None], -fwd * 0.4, e_move)

        # jump-crit attempts
        jump = ((E.y <= 1e-4) & (E.web <= 0) & (dist < 4.2) & (r.random(self.n) < 0.10)
                & (~E.cage_until.astype(bool)))
        E.vy = np.where(jump, 8.4, E.vy)
        E.web = np.where(raise_shield, E.web, E.web)

        # ---- enemy melee swing
        re_dist = np.linalg.norm(E.pos - L.pos, axis=1)
        want_hit = ((~E.shield_up) & (re_dist <= REACH) & (E.cd <= 0.0)
                    & (r.random(self.n) < (0.55 + 0.4 * diff)) & (~acted))
        if want_hit.any():
            blocked = L.has_shield & L.shield_up & (L.shield_disabled <= 0)
            axe_break = blocked & E.is_axe()
            L.shield_up = np.where(axe_break, False, L.shield_up)
            L.shield_disabled = np.where(axe_break, 5.0, L.shield_disabled)
            real_hit = want_hit & (~blocked | axe_break)
            crit = (E.y > 0.05) & (E.vy < 0.0)
            bonus = np.where(E.buff > 0, 3.0, 0.0)
            raw = np.where(real_hit, (E.dmg() + bonus) * np.where(crit, 1.5, 1.0), 0.0)
            dtg, pop = self._damage(L, raw, blast=False)
            dmg_taken += np.where(want_hit & ~blocked, dtg, 0.0)
            L.fire = np.where((dtg > 0) & (E.fire_sec() > 0),
                              np.maximum(L.fire, E.fire_sec()), L.fire)
            kb = np.where(real_hit & (dtg > 0), 4.4 * E.kb_mult(), 0.0)
            hit_dir = (L.pos - E.pos) / np.maximum(re_dist, 1e-6)[:, None]
            L.vel += hit_dir * kb[:, None]
            E.cd = np.where(want_hit, np.maximum(E.cd, E.cd_time()), E.cd)
            if pop.any():
                info["totem_popped_learner"] = (info.get("totem_popped_learner", False) | pop)

        E.vel = E.vel * 0.45 + e_move * speed[:, None] * 0.55
        E.pos = self._push_out(np.clip(E.pos + E.vel * DT, -ARENA_BOUND, ARENA_BOUND))
        return {
            "move": e_move, "dist": dist, "fwd": fwd, "acted": acted,
            "dmg_dealt": dmg_dealt, "dmg_taken": dmg_taken, "healed": healed,
        }

    # ---------------------------------------------------------------- main step
    def step(self, actions: np.ndarray):
        act = np.asarray(actions, np.int32)
        n = self.n
        L, E = self.L, self.E
        self.step_count += 1
        self.t += DT

        dmg_dealt = np.zeros(n, np.float32)
        dmg_taken = np.zeros(n, np.float32)
        healed = np.zeros(n, np.float32)
        tech = np.zeros(n, np.int32)
        waste = np.zeros(n, bool)
        flags = {
            "shield_broken": np.zeros(n, bool),
            "blocked_hit": np.zeros(n, bool),
            "trapped_enemy": np.zeros(n, bool),
            "tactical_cover": np.zeros(n, bool),
            "crystal_hit": np.zeros(n, bool),
            "anchor_hit": np.zeros(n, bool),
            "minecart_hit": np.zeros(n, bool),
            "cage_blocked": np.zeros(n, bool),
            "wtap_combo": np.zeros(n, bool),
            "self_blast": np.zeros(n, bool),
            "totem_popped_learner": np.zeros(n, bool),
            "totem_popped_enemy": np.zeros(n, bool),
        }
        L.shield_up[:] = False

        vec = E.pos - L.pos
        dist = np.linalg.norm(vec, axis=1)
        fwd = vec / np.maximum(dist, 1e-6)[:, None]
        right = np.stack([-fwd[:, 1], fwd[:, 0]], 1)

        move = np.zeros((n, 2), np.float32)
        speed_mult = self._speed_mult(L)
        speed = np.full(n, WALK, np.float32) * speed_mult
        want_jump = np.zeros(n, bool)
        want_attack = np.zeros(n, bool)
        sprinting = np.zeros(n, bool)

        def a(*ids):
            m = np.zeros(n, bool)
            for i in ids:
                m |= act == i
            return m

        m = a(1)
        move = np.where(m[:, None], fwd, move)
        m = a(2)
        move = np.where(m[:, None], -fwd, move)
        m = a(3)
        move = np.where(m[:, None], -right, move)
        m = a(4)
        move = np.where(m[:, None], right, move)
        m = a(5)
        move = np.where(m[:, None], fwd, move)
        want_attack |= m
        m = a(6)
        move = np.where(m[:, None], -right, move)
        want_attack |= m
        m = a(7)
        move = np.where(m[:, None], right, move)
        want_attack |= m
        want_jump |= a(8)
        m = a(9)
        move = np.where(m[:, None], fwd, move)
        want_jump |= m
        sprinting |= m
        want_attack |= m
        speed = np.where(m, SPRINT * speed_mult, speed)
        m = a(10)
        move = np.where(m[:, None], fwd, move)
        sprinting |= m
        speed = np.where(m, SPRINT * speed_mult, speed)
        m = a(11)
        move = np.where(m[:, None], -fwd, move)
        want_attack |= m

        # --- 12 shield
        m = a(12)
        ok = m & L.has_shield & (L.shield_disabled <= 0.0)
        L.shield_up = np.where(ok, True, L.shield_up)
        L.shield_held = np.where(ok, True, L.shield_held)
        move = np.where(ok[:, None], -fwd * 0.45, move)
        waste |= m & ~ok

        # --- 13 golden apple
        m = a(13) & (L.gaps > 0) & (L.use_lock <= 0.0)
        waste |= a(13) & (L.hp >= 19.5) & (L.abs >= 2.0)
        if m.any():
            L.gaps = np.where(m, L.gaps - 1, L.gaps)
            L.regen = np.where(m, 5.0, L.regen)
            before = L.hp + L.abs
            L.abs = np.where(m, np.maximum(L.abs, 4.0), L.abs)
            healed += np.where(m, np.maximum(0.0, (L.hp + L.abs) - before), 0.0)
            L.use_lock = np.where(m, 0.35, L.use_lock)
            move = np.where(m[:, None], -fwd * 0.35, move)
            tech = np.where(m, T_GAP, tech)
        waste |= a(13) & ~(m | (a(13) & (L.hp >= 19.5) & (L.abs >= 2.0)))

        # --- 14 instant heal splash
        m = a(14) & (L.splashes > 0) & (L.use_lock <= 0.0)
        waste |= a(14) & (L.hp >= 18.5)
        if m.any():
            L.splashes = np.where(m, L.splashes - 1, L.splashes)
            before = L.hp
            L.hp = np.where(m, np.minimum(20.0, L.hp + 7.5), L.hp)
            healed += np.where(m, L.hp - before, 0.0)
            L.use_lock = np.where(m, 0.2, L.use_lock)
            tech = np.where(m, T_HEAL, tech)
        waste |= a(14) & ((L.splashes <= 0) | (L.use_lock > 0.0))

        # --- 15 ender pearl
        m = a(15) & (L.pearls > 0) & (L.pearl_cd <= 0.0)
        if m.any():
            L.pearls = np.where(m, L.pearls - 1, L.pearls)
            L.pearl_cd = np.where(m, 1.5, L.pearl_cd)
            L.web = np.where(m, 0.0, L.web)
            L.lava = np.where(m, 0.0, L.lava)
            low_hp = L.hp <= 12.0
            tgt = np.where(low_hp[:, None], E.pos - fwd * 8.0, E.pos - fwd * 2.1)
            far = dist > 3.8
            tgt = np.where(far[:, None], tgt, L.pos - fwd * 4.0)
            L.pos = np.where(m[:, None], np.clip(tgt, -ARENA_BOUND, ARENA_BOUND), L.pos)
            pearl_dmg, _ = self._damage(L, np.where(m, 1.5, 0.0), bypass=True)
            dmg_taken += pearl_dmg
            L.hp = np.where(m & low_hp, L.hp, L.hp)
            tech = np.where(m, T_PEARL, tech)
            waste |= m & ~far & ~low_hp
        waste |= a(15) & ((L.pearls <= 0) | (L.pearl_cd > 0.0))

        # --- 16 buff potion
        m = a(16) & (L.buffs > 0) & (L.buff <= 1.0) & (L.use_lock <= 0.0)
        if m.any():
            L.buffs = np.where(m, L.buffs - 1, L.buffs)
            L.buff = np.where(m, 15.0, L.buff)
            L.use_lock = np.where(m, 0.25, L.use_lock)
            move = np.where(m[:, None], -fwd * 0.4, move)
            tech = np.where(m, T_BUFF, tech)
        waste |= a(16) & ~(m)

        # --- 17 ranged
        m = a(17) & (L.ranged > 0) & (L.use_lock <= 0.0)
        if m.any():
            rdm, rkb, rsl, rpierce = L.ranged_spec()
            in_range = (dist >= 1.6) & (dist <= 16.0)
            cover = self._obstacle_between(L.pos, E.pos)
            blocked = E.has_shield & E.shield_up & (E.shield_disabled <= 0) & (~rpierce)
            hit = m & in_range & (~cover) & (~blocked)
            raw = np.where(hit, rdm, 0.0)
            dtg, pop = self._damage(E, raw, bypass=rpierce)
            dmg_dealt += dtg
            flags["totem_popped_enemy"] |= pop
            E.slow = np.where(hit, np.maximum(E.slow, rsl), E.slow)
            E.vel += fwd * np.where(hit, rkb, 0.0)[:, None]
            flags["blocked_hit"] |= (m & blocked).astype(bool)
            L.ranged = np.where(m, L.ranged - 1, L.ranged)
            L.use_lock = np.where(m, 0.15, L.use_lock)
            tech = np.where(m, T_RANGED, tech)
            waste |= m & ~in_range
        waste |= a(17) & (L.ranged <= 0)

        # --- 18 barricade
        m = (a(18) & (L.blocks >= 2) & (L.use_lock <= 0.0) & (dist >= 1.6) & (dist <= 7.5)
             & (self.obstacle_n < MAX_OBSTACLES))
        if m.any():
            L.blocks = np.where(m, L.blocks - 2, L.blocks)
            mid = np.clip((L.pos + E.pos) * 0.5, -8.0, 8.0)
            self._add_obstacle(mid, 12.0)
            L.use_lock = np.where(m, 0.25, L.use_lock)
            move = np.where(m[:, None], -fwd * 0.65, move)
            tech = np.where(m, T_BARRICADE, tech)
            flags["tactical_cover"] |= m & (L.hp <= 15.5)
        waste |= a(18) & ~m

        # --- 19 cobweb / lava trap
        m_web = (a(19) & (L.cobwebs > 0) & (L.use_lock <= 0.0) & (dist <= 4.5)
                 & (L.lava_bk <= 0))
        m_lava = (a(19) & ~m_web & (L.lava_bk > 0) & (L.use_lock <= 0.0)
                  & (dist >= 1.9) & (dist <= 4.2))
        if m_web.any():
            L.cobwebs = np.where(m_web, L.cobwebs - 1, L.cobwebs)
            L.use_lock = np.where(m_web, 0.25, L.use_lock)
            E.web = np.where(m_web, 2.6, E.web)
            tech = np.where(m_web, T_COBWEB, tech)
            flags["trapped_enemy"] |= m_web
        if m_lava.any():
            L.lava_bk = np.where(m_lava, L.lava_bk - 1, L.lava_bk)
            L.use_lock = np.where(m_lava, 0.3, L.use_lock)
            E.lava = np.where(m_lava, 1.6, E.lava)
            E.fire = np.where(m_lava, np.maximum(E.fire, 3.5), E.fire)
            close = m_lava & (dist < 2.2)
            L.fire = np.where(close, np.maximum(L.fire, 2.5), L.fire)
            tech = np.where(m_lava, T_LAVA, tech)
            flags["trapped_enemy"] |= m_lava
            move = np.where(m_lava[:, None], -fwd * 0.7, move)
        waste |= a(19) & ~(m_web | m_lava)

        # --- 20 water bucket
        m = a(20) & (L.water_bk > 0) & (L.use_lock <= 0.0)
        hazard = (L.fire > 0) | (L.lava > 0) | (L.web > 0)
        close_push = dist <= 3.4
        use = m & (hazard | close_push)
        if use.any():
            L.water_bk = np.where(use, L.water_bk - 1, L.water_bk)
            L.use_lock = np.where(use, 0.25, L.use_lock)
            L.fire = np.where(use, 0.0, L.fire)
            L.lava = np.where(use, 0.0, L.lava)
            L.web = np.where(use, 0.0, L.web)
            push = use & close_push
            E.vel += fwd * np.where(push, 4.5, 0.0)[:, None]
            E.slow = np.where(push, np.maximum(E.slow, 1.2), E.slow)
            tech = np.where(use, T_WATER, tech)
        waste |= (a(20) & ~use) | (m & ~use)

        # --- 21 pillar jump
        m = (a(21) & (L.blocks > 0) & (L.use_lock <= 0.0) & (L.y <= 1e-4)
             & (L.pillar <= 0.0))
        if m.any():
            L.blocks = np.where(m, L.blocks - 1, L.blocks)
            L.use_lock = np.where(m, 0.2, L.use_lock)
            L.pillar = np.where(m, 1.8, L.pillar)
            want_jump |= m
            tech = np.where(m, T_PILLAR, tech)
            waste |= m & (dist > 4.5) & (L.ranged <= 0)
        waste |= a(21) & ~m

        # --- 22 END CRYSTAL (the pro finisher)
        m = (a(22) & (L.crystals > 0) & (L.obsidian > 0) & (L.use_lock <= 0.0)
             & (dist >= 2.4) & (dist <= 5.5))
        if m.any():
            L.crystals = np.where(m, L.crystals - 1, L.crystals)
            L.obsidian = np.where(m, L.obsidian - 1, L.obsidian)
            L.use_lock = np.where(m, 0.3, L.use_lock)
            dtg, dsf, raw_t, raw_c, _ = self._explode(E, L, 10.5, 3.0, 24.0, 7.0)
            dmg_dealt += dtg
            dmg_taken += dsf
            flags["crystal_hit"] |= dtg >= 6.0
            flags["self_blast"] |= dsf >= 4.0
            flags["cage_blocked"] |= (E.cage_until > 0.0) & (raw_t >= 8.0)
            tech = np.where(m, T_CRYSTAL, tech)
            move = np.where(m[:, None], -fwd * 0.6, move)
        waste |= (a(22) & ((L.crystals <= 0) | (L.obsidian <= 0) | (dist < 2.4) | (dist > 5.5))
                  & ~m)

        # --- 23 respawn anchor
        m = (a(23) & (L.anchors > 0) & (L.use_lock <= 0.0) & (dist >= 1.5) & (dist <= 4.8))
        if m.any():
            L.anchors = np.where(m, L.anchors - 1, L.anchors)
            L.use_lock = np.where(m, 0.3, L.use_lock)
            dtg, dsf, raw_t, _, _ = self._explode(E, L, 9.0, 2.5, 20.0, 6.0)
            dmg_dealt += dtg
            dmg_taken += dsf
            flags["anchor_hit"] |= dtg >= 5.0
            flags["self_blast"] |= dsf >= 4.0
            flags["cage_blocked"] |= (E.cage_until > 0.0) & (raw_t >= 8.0)
            tech = np.where(m, T_ANCHOR, tech)
        waste |= a(23) & ((L.anchors <= 0) | (dist < 1.5) | (dist > 4.8))

        # --- 24 TNT minecart
        m = (a(24) & (L.minecarts > 0) & (L.use_lock <= 0.0) & (dist >= 3.0) & (dist <= 8.5))
        if m.any():
            L.minecarts = np.where(m, L.minecarts - 1, L.minecarts)
            L.use_lock = np.where(m, 0.35, L.use_lock)
            dtg, dsf, raw_t, _, _ = self._explode(E, L, 7.5, 2.0, 17.0, 5.0)
            dmg_dealt += dtg
            dmg_taken += dsf
            flags["minecart_hit"] |= dtg >= 5.0
            flags["self_blast"] |= dsf >= 4.0
            flags["cage_blocked"] |= (E.cage_until > 0.0) & (raw_t >= 8.0)
            E.slow = np.where(m, np.maximum(E.slow, 1.2), E.slow)
            tech = np.where(m, T_MINECART, tech)
        waste |= a(24) & ((L.minecarts <= 0) | (dist < 3.0) | (dist > 8.5))

        # --- 25 obsidian blast cage
        m = a(25) & (L.obsidian >= 4) & (L.use_lock <= 0.0)
        if m.any():
            L.obsidian = np.where(m, L.obsidian - 4, L.obsidian)
            L.cage_until = np.where(m, 4.0, L.cage_until)
            L.use_lock = np.where(m, 0.3, L.use_lock)
            tech = np.where(m, T_CAGE, tech)
        waste |= a(25) & (L.obsidian < 4)

        # --- 26 W-tap sprint-reset combo (attack resolved later, bonus & no knockback)
        m = a(26) & (dist <= REACH + 0.2) & (L.cd <= 0.03)
        wtap_attack = m.copy()
        want_attack |= m
        if m.any():
            L.cd = np.where(m, L.cd_time() * 0.7, L.cd)
            tech = np.where(m, T_WTAP, tech)
        waste |= a(26) & ~m

        # --- 27 pearl + crystal combo
        m = (a(27) & (L.pearls > 0) & (L.crystals > 0) & (L.obsidian > 0)
             & (L.pearl_cd <= 0.0) & (L.use_lock <= 0.0) & (dist >= 3.6) & (dist <= 12.0))
        if m.any():
            L.pearls = np.where(m, L.pearls - 1, L.pearls)
            L.crystals = np.where(m, L.crystals - 1, L.crystals)
            L.obsidian = np.where(m, L.obsidian - 1, L.obsidian)
            L.pearl_cd = np.where(m, 1.5, L.pearl_cd)
            L.use_lock = np.where(m, 0.35, L.use_lock)
            L.pos = np.where(m[:, None],
                             np.clip(E.pos - fwd * 1.9, -ARENA_BOUND, ARENA_BOUND), L.pos)
            dtg, dsf, raw_t, _, _ = self._explode(E, L, 10.5, 3.0, 24.0, 7.0)
            dmg_dealt += dtg
            dmg_taken += dsf
            flags["crystal_hit"] |= m & (dtg >= 6.0)
            flags["self_blast"] |= m & (dsf >= 4.0)
            flags["cage_blocked"] |= m & (E.cage_until > 0.0) & (raw_t >= 8.0)
            tech = np.where(m, T_PEARL_CRYSTAL, tech)
            move = np.zeros((n, 2), np.float32)
        waste |= a(27) & ~m

        # ---- learner vertical physics
        jump_ok = want_jump & (L.y <= 1e-4) & (L.web <= 0)
        L.vy = np.where(jump_ok, 8.4, L.vy)
        prev_vy = L.vy.copy()
        L.y = L.y + L.vy * DT
        L.vy = np.where(L.y > 0, L.vy - 28.0 * DT, 0.0)
        L.y = np.maximum(L.y, 0.0)

        # ---- player horizontal movement
        L.vel = L.vel * 0.45 + move * speed[:, None] * 0.55
        L.pos = self._push_out(np.clip(L.pos + L.vel * DT, -ARENA_BOUND, ARENA_BOUND))

        # ---- enemy decides & moves
        brain = self._enemy_brain(act, dmg_dealt, dmg_taken, healed, None)
        dmg_dealt, dmg_taken, healed = brain["dmg_dealt"], brain["dmg_taken"], brain["healed"]

        # ---- obstacles & timers
        self.obstacle_ttl = np.maximum(0.0, self.obstacle_ttl - DT)
        for s in range(MAX_OBSTACLES):
            gone = (self.obstacle_ttl[:, s] <= 0) & (self.obstacle_n > s)
            self.obstacle_n = np.where(gone, np.maximum(0, self.obstacle_n - 1), self.obstacle_n)
        for B in (L, E):
            B.cd = np.maximum(0.0, B.cd - DT)
            B.hurt_cd = np.maximum(0.0, B.hurt_cd - DT)
            B.shield_disabled = np.maximum(0.0, B.shield_disabled - DT)
            B.buff = np.maximum(0.0, B.buff - DT)
            B.slow = np.maximum(0.0, B.slow - DT)
            B.web = np.maximum(0.0, B.web - DT)
            B.lava = np.maximum(0.0, B.lava - DT)
            B.fire = np.maximum(0.0, B.fire - DT)
            B.pillar = np.maximum(0.0, B.pillar - DT)
            B.pearl_cd = np.maximum(0.0, B.pearl_cd - DT)
            B.use_lock = np.maximum(0.0, B.use_lock - DT)
            B.regen = np.maximum(0.0, B.regen - DT)
            B.cage_until = np.maximum(0.0, B.cage_until - DT)

        # regen
        before = L.hp
        L.hp = np.where(L.regen > 0, np.minimum(20.0, L.hp + 0.35), L.hp)
        healed += np.maximum(0.0, L.hp - before)
        E.hp = np.where(E.regen > 0, np.minimum(20.0, E.hp + 0.35), E.hp)

        # lava / fire ticks
        ld, lp = self._damage(L, np.where(L.lava > 0, 0.45, 0.0), iframes=False)
        dmg_taken += ld
        flags["totem_popped_learner"] |= lp
        ed, ep = self._damage(E, np.where(E.lava > 0, 0.45, 0.0), iframes=False)
        dmg_dealt += ed
        flags["totem_popped_enemy"] |= ep
        ld, lp = self._damage(L, np.where(L.fire > 0, 0.12, 0.0), bypass=True, iframes=False)
        dmg_taken += ld
        flags["totem_popped_learner"] |= lp
        ed, ep = self._damage(E, np.where(E.fire > 0, 0.12, 0.0), bypass=True, iframes=False)
        dmg_dealt += ed
        flags["totem_popped_enemy"] |= ep

        # ---- learner melee resolution
        re_dist = np.linalg.norm(E.pos - L.pos, axis=1)
        hit_dir = (E.pos - L.pos) / np.maximum(re_dist, 1e-6)[:, None]
        cover = self._obstacle_between(L.pos, E.pos)
        eff_reach = REACH + np.where(L.pillar > 0, 0.30, 0.0)
        swing = (want_attack & (L.use_lock <= 0.0) & (L.cd <= 0.04) & (re_dist <= eff_reach)
                 & ((~cover) | (L.pillar > 0)))
        e_shield = E.has_shield & E.shield_up & (E.shield_disabled <= 0)
        break_shield = swing & e_shield & L.is_axe()
        blocked = swing & e_shield & (~L.is_axe())
        real = swing & (~e_shield | break_shield)
        crit = ((L.y > 0.05) & (prev_vy < 0.0) & (~sprinting)) | ((L.pillar > 0) & (~sprinting))
        mult = np.where(crit, 1.5, 1.0)
        mult = np.where(wtap_attack & real, mult * 1.15, mult)
        bonus = np.where(L.buff > 0, 3.0, 0.0)
        raw = np.where(real, (L.dmg() + bonus) * mult, 0.0)
        dealt, pop = self._damage(E, raw, blast=False)
        dmg_dealt += dealt
        flags["totem_popped_enemy"] |= pop
        E.fire = np.where((dealt > 0) & (L.fire_sec() > 0), np.maximum(E.fire, L.fire_sec()), E.fire)
        base_kb = 4.4 * L.kb_mult() * np.where(sprinting, 1.35, 1.0)
        kb = np.where(real & (dealt > 0), base_kb, 0.0)
        kb = np.where(wtap_attack, 0.0, kb)  # W-tap cancels knockback
        E.vel += hit_dir * kb[:, None]
        flags["wtap_combo"] |= wtap_attack & real
        flags["shield_broken"] |= break_shield
        flags["blocked_hit"] |= blocked
        L.vel -= hit_dir * np.where(blocked, 2.0, 0.0)[:, None]
        L.cd = np.where(swing, np.maximum(L.cd, L.cd_time() * np.where(wtap_attack, 0.7, 1.0)),
                        L.cd)

        L.cobwebs = np.maximum(L.cobwebs, 0.0)

        # ---- termination
        learner_dead = L.hp <= DEATH
        enemy_dead = E.hp <= DEATH
        timeout = self.step_count >= MAX_STEPS
        terminated = (learner_dead | enemy_dead) & ~timeout
        truncated = timeout & ~terminated

        # ---- rewards
        reward = np.full(n, -0.01, np.float32)
        reward += dmg_dealt * 2.5 - dmg_taken * 1.4 + healed * 0.4
        reward += np.where(flags["shield_broken"], 4.0, 0.0)
        reward += np.where(flags["blocked_hit"], 2.5, 0.0)
        reward += np.where(flags["trapped_enemy"], 3.0, 0.0)
        reward += np.where(flags["tactical_cover"], 2.0, 0.0)
        reward += np.where(flags["crystal_hit"], 5.0, 0.0)
        reward += np.where(flags["anchor_hit"], 4.0, 0.0)
        reward += np.where(flags["minecart_hit"], 4.0, 0.0)
        reward += np.where(flags["cage_blocked"], 4.0, 0.0)
        reward += np.where(flags["wtap_combo"], 2.0, 0.0)
        reward += np.where(flags["totem_popped_enemy"], 8.0, 0.0)
        reward -= np.where(flags["self_blast"], 3.0, 0.0)
        reward -= np.where(waste, 1.5, 0.0)

        prev_dist = brain["dist"]
        new_dist = re_dist
        reward += np.where(new_dist > 3.0, np.clip((prev_dist - new_dist) * 0.05, -0.05, 0.05), 0.0)
        reward += np.where(enemy_dead, 100.0, 0.0)
        reward -= np.where(learner_dead, 100.0, 0.0)
        reward = reward.astype(np.float32)

        run = np.ones(n, bool)
        cnt = np.minimum(len(TECHNIQUES) - 1, tech).astype(np.int32)
        self.consumed[np.arange(n), cnt] += (tech > 0).astype(np.int32)
        self.last_tech = tech

        done = terminated | truncated
        info = {
            "learner_dead": learner_dead, "enemy_dead": enemy_dead, "timeout": truncated,
            "learner_hp": L.hp.copy(), "enemy_hp": E.hp.copy(), "distance": re_dist.copy(),
            "damage_dealt": dmg_dealt, "damage_taken": dmg_taken, "healed": healed,
            "tech": tech, "waste": waste, **{k: v for k, v in flags.items()},
        }
        obs = self._obs()
        return obs, reward, terminated, truncated, done, info

    def kits(self) -> list[tuple[str, str, str, str]]:
        out = []
        for i in range(self.n):
            lw = MELEE_POOL[int(self.L.weapon[i])].weapon_id
            la = ARMOR_TIERS[int(self.L.armor[i])].name
            ew = MELEE_POOL[int(self.E.weapon[i])].weapon_id
            ea = ARMOR_TIERS[int(self.E.armor[i])].name
            out.append((f"{lw}+{la}+blk({int(self.L.blocks[i])})",
                        f"{ew}+{ea}+blk({int(self.E.blocks[i])})", lw, ew))
        return out


class ProPvPEnv(gym.Env):
    """Single-arena gym wrapper (used for evaluation / recording / play.py)."""

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = MAX_STEPS, seed: int | None = None):
        super().__init__()
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(-1.0, 2.0, (OBS_DIM,), np.float32)
        self.max_steps = max_steps
        self.core = ProArena(1, seed=seed)
        self._prev_dist = 0.0
        self.consumables_used: dict[str, int] = {}
        self._kits: tuple[str, str, str, str] = ("", "", "", "")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.core.rng = np.random.default_rng(seed)
        self.core.reset()
        self.consumables_used = {}
        self._kits = self.core.kits()[0]
        obs = self.core._obs()[0]
        self._prev_dist = float(obs[2]) * 20.0
        return obs, {
            "ready": True, "learner_kit": self._kits[0], "enemy_kit": self._kits[1],
            "learner_weapon": self._kits[2], "enemy_weapon": self._kits[3],
        }

    def step(self, action):
        act = np.array([int(action)], np.int32)
        obs, rew, term, trunc, done, info = self.core.step(act)
        tech = int(info["tech"][0])
        used = TECHNIQUES[tech] if tech > 0 else None
        if used:
            self.consumables_used[used] = self.consumables_used.get(used, 0) + 1
        out = {
            "ready": True,
            "learner_kit": self._kits[0], "enemy_kit": self._kits[1],
            "learner_weapon": self._kits[2], "enemy_weapon": self._kits[3],
            "damage_dealt": float(info["damage_dealt"][0]),
            "damage_taken": float(info["damage_taken"][0]),
            "healed": float(info["healed"][0]),
            "used_consumable": used,
            "placed_block": used if used in ("block_barricade", "pillar_jump") else None,
            "wasted_consumable": bool(info["waste"][0]),
            "shield_broken": bool(info["shield_broken"][0]),
            "blocked_hit": bool(info["blocked_hit"][0]),
            "trapped_enemy": bool(info["trapped_enemy"][0]),
            "tactical_cover": bool(info["tactical_cover"][0]),
            "crystal_hit": bool(info["crystal_hit"][0]),
            "anchor_hit": bool(info["anchor_hit"][0]),
            "minecart_hit": bool(info["minecart_hit"][0]),
            "cage_blocked": bool(info["cage_blocked"][0]),
            "wtap_combo": bool(info["wtap_combo"][0]),
            "self_blast": bool(info["self_blast"][0]),
            "totem_popped_learner": bool(info["totem_popped_learner"][0]),
            "totem_popped_enemy": bool(info["totem_popped_enemy"][0]),
            "consumables_summary": dict(self.consumables_used),
            "learner_hp": float(info["learner_hp"][0]),
            "enemy_hp": float(info["enemy_hp"][0]),
            "distance": float(info["distance"][0]),
            "learner_dead": bool(info["learner_dead"][0]),
            "enemy_dead": bool(info["enemy_dead"][0]),
            "timeout": bool(info["timeout"][0]),
        }
        return obs[0], float(rew[0]), bool(term[0]), bool(trunc[0]), out

    def close(self):
        pass


class ProPvPVecEnv(VecEnv):
    """SB3 VecEnv over the vectorised PRO arena."""

    def __init__(self, n_envs: int = 128, seed: int = 0, difficulty: float = 1.0):
        super().__init__(n_envs, spaces.Box(-1.0, 2.0, (OBS_DIM,), np.float32),
                         spaces.Discrete(N_ACTIONS))
        self.core = ProArena(n_envs, seed=seed, difficulty=difficulty)
        self._obs = self.core.reset()
        self._actions = np.zeros(n_envs, np.int32)
        self.episodes = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.tech_counts = {name: 0 for name in TECHNIQUES if name != "none"}
        self.last_results: list[str] = []

    def set_difficulty(self, value: float) -> None:
        self.core.set_difficulty(value)

    def set_enemy_skill(self, value: float) -> None:
        self.core.set_enemy_skill(value)

    def reset(self):
        self._obs = self.core.reset()
        return self._obs.copy()

    def step_async(self, actions):
        self._actions = np.asarray(actions, np.int32)

    def step_wait(self):
        obs, rew, term, trunc, done, info = self.core.step(self._actions)
        infos: list[dict] = [{} for _ in range(self.num_envs)]
        done_idx = np.flatnonzero(done)
        if len(done_idx):
            kits = self.core.kits()
            row_mask = np.zeros(self.num_envs, bool)
            for i in done_idx:
                won = bool(info["enemy_dead"][i]) and not bool(info["learner_dead"][i])
                lost = bool(info["learner_dead"][i]) and not bool(info["enemy_dead"][i])
                self.episodes += 1
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
                cons = {}
                for t in range(1, len(TECHNIQUES)):
                    c = int(self.core.consumed[i, t])
                    if c > 0:
                        self.tech_counts[TECHNIQUES[t]] += c
                        cons[TECHNIQUES[t]] = c
                        self.core.consumed[i, t] = 0
                infos[i] = {
                    "TimeLimit.truncated": bool(trunc[i]),
                    "terminal_observation": obs[i].copy(),
                    "learner_kit": kits[i][0], "enemy_kit": kits[i][1],
                    "learner_weapon": kits[i][2], "enemy_weapon": kits[i][3],
                    "learner_dead": bool(info["learner_dead"][i]),
                    "enemy_dead": bool(info["enemy_dead"][i]),
                    "timeout": bool(trunc[i]),
                    "learner_hp": float(info["learner_hp"][i]),
                    "enemy_hp": float(info["enemy_hp"][i]),
                    "crystal_hit": bool(info["crystal_hit"][i]),
                    "anchor_hit": bool(info["anchor_hit"][i]),
                    "minecart_hit": bool(info["minecart_hit"][i]),
                    "wtap_combo": bool(info["wtap_combo"][i]),
                    "cage_blocked": bool(info["cage_blocked"][i]),
                    "shield_broken": bool(info["shield_broken"][i]),
                    "totems_popped": int(info["totem_popped_enemy"][i]) + int(info["totem_popped_learner"][i]),
                    "consumables_summary": cons,
                }
                row_mask[i] = True
            self.core.reset(row_mask)
            obs = self.core._obs()
        self._obs = obs
        return obs.copy(), rew.copy(), term.copy(), infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self.core, attr_name, None) for _ in range(self.num_envs)]

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [None] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs
