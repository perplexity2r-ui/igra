"""Fast local 1.16.5 PvP arena simulator — Absolute Asymmetric Item Randomizer & Blocks Mode.

Matches `MinecraftPvPEnv` (`Box(24,)` observations, `Discrete(22)` actions) 1-to-1 so PPO
models trained here transfer directly to `bot/bot.js` over WebSocket.

Every episode, `LearnerBot` and `EnemyBot` receive COMPLETELY DIFFERENT, randomly rolled
loadouts (`avoid_weapon_idx`, `avoid_armor_idx`, `avoid_ranged_idx`):
  - 12 random melee items/tools/weapons (Knockback II Stick, Wooden Shovel, Fire Golden Hoe,
    Stone Sword, Iron Pickaxe, Iron Sword, Iron Axe, Diamond Sword, Impaling II Trident,
    Sharpness II Diamond Axe, Fire II Netherite Sword, Sharpness III Netherite Axe).
  - 5 distinct armor tiers (`none` 0%, `leather` 16%, `chain_gold` 36%, `iron` 56%, `diamond_mix` 72%
    damage mitigation — scaled for 1.16.5 armor + toughness).
  - Building blocks (`cobblestone`, `oak_planks`, `obsidian`, `hay_block`: 0..24) for placing
    tactical cover barricades (blocks ranged shots & slows direct rushes) and pillar high-ground jumps.
  - Traps & utility buckets: `cobweb` (1..4, 80% slowness trap), `lava_bucket` (0..1, heavy burn & slow),
    `water_bucket` (0..1, extinguishes fire, clears cobwebs/lava, and pushes enemy back).
  - Off-hand `shield` OR `totem_of_undying` (auto-pops on lethal hit: restores HP + Absorption + Regen!).
  - Random consumables: healing items (`golden_apple` + `splash_heal` / stews / foods), `ender_pearl`,
    `buff_potion` (Speed II + Strength), and asymmetric projectiles (`snowball`, `egg`, `crossbow`,
    `splash_harming`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class WeaponSpec:
    idx: int
    weapon_id: str
    damage: float
    cooldown: float
    kb_mult: float
    shield_break: bool
    fire_seconds: float


MELEE_POOL: tuple[WeaponSpec, ...] = (
    WeaponSpec(0, "stick_kb", 1.0, 0.25, 2.2, False, 0.0),
    WeaponSpec(1, "wooden_shovel", 2.5, 1.00, 1.0, False, 0.0),
    WeaponSpec(2, "golden_hoe", 1.0, 0.25, 1.0, False, 3.0),
    WeaponSpec(3, "stone_sword", 5.0, 0.625, 1.0, False, 0.0),
    WeaponSpec(4, "iron_pickaxe", 4.0, 0.83, 1.0, False, 0.0),
    WeaponSpec(5, "iron_sword", 6.0, 0.625, 1.0, False, 0.0),
    WeaponSpec(6, "iron_axe", 9.0, 1.10, 1.1, True, 0.0),
    WeaponSpec(7, "diamond_sword", 7.0, 0.625, 1.0, False, 0.0),
    WeaponSpec(8, "trident", 9.0, 0.91, 1.1, False, 0.0),
    WeaponSpec(9, "diamond_axe", 10.5, 1.00, 1.1, True, 0.0),
    WeaponSpec(10, "netherite_sword_fire", 8.0, 0.625, 1.2, False, 4.0),
    WeaponSpec(11, "netherite_axe_sharp", 12.0, 1.00, 1.2, True, 0.0),
)


@dataclass(frozen=True)
class ArmorSpec:
    idx: int
    name: str
    reduction: float


ARMOR_TIERS: tuple[ArmorSpec, ...] = (
    ArmorSpec(0, "none", 0.00),
    ArmorSpec(1, "leather", 0.16),
    ArmorSpec(2, "chain_gold", 0.36),
    ArmorSpec(3, "iron", 0.56),
    ArmorSpec(4, "diamond_mix", 0.72),
)


BLOCK_TYPES: tuple[str, ...] = (
    "minecraft:cobblestone",
    "minecraft:oak_planks",
    "minecraft:obsidian",
    "minecraft:hay_block",
)


@dataclass(frozen=True)
class RangedSpec:
    idx: int
    ranged_id: str
    min_count: int
    max_count: int
    damage: float
    knockback: float
    slow_sec: float
    pierces_armor: bool


RANGED_TYPES: tuple[RangedSpec, ...] = (
    RangedSpec(0, "snowball", 4, 12, 0.5, 5.2, 0.8, False),
    RangedSpec(1, "egg", 4, 12, 0.5, 5.2, 0.8, False),
    RangedSpec(2, "crossbow", 1, 3, 7.0, 3.8, 3.5, False),
    RangedSpec(3, "splash_harming", 1, 2, 6.0, 1.5, 1.5, True),
)


@dataclass
class BotState:
    weapon: WeaponSpec
    armor: ArmorSpec
    block_item: str
    ranged_spec: RangedSpec
    blocks: int
    cobwebs: int
    lava_buckets: int
    water_buckets: int
    has_shield: bool
    has_totem: bool
    gaps: int
    splashes: int
    pearls: int
    buffs: int
    ranged: int

    pos: np.ndarray
    vel: np.ndarray
    y: float = 0.0
    vy: float = 0.0
    hp: float = 20.0
    absorption: float = 0.0
    cd: float = 0.0
    hurt_cd: float = 0.0

    shield_up: bool = False
    shield_disabled: float = 0.0
    regen_timer: float = 0.0
    buff_timer: float = 0.0
    fire_timer: float = 0.0
    slow_timer: float = 0.0
    cobweb_timer: float = 0.0
    lava_timer: float = 0.0
    pillar_timer: float = 0.0
    pearl_cd: float = 0.0
    use_lock: float = 0.0

    @property
    def summary(self) -> str:
        return f"{self.weapon.weapon_id}+{self.armor.name}+blk({self.blocks})"


class SimulatedPvPEnv(gym.Env):
    metadata = {"render_modes": []}

    ARENA_BOUND = 9.2
    DT = 0.10  # 100ms per step
    REACH = 3.25
    DEATH_HP_THRESHOLD = 6.0

    def __init__(self, max_steps: int = 600):
        super().__init__()
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(22)
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(24,), dtype=np.float32
        )

        self.learner: BotState = self._dummy_bot()
        self.enemy: BotState = self._dummy_bot()
        self.obstacles: list[tuple[np.ndarray, float]] = []  # (pos_xz, ttl_sec)
        self.step_count = 0
        self._prev_dist = 0.0
        self.consumables_used: dict[str, int] = {}

    @staticmethod
    def _dummy_bot() -> BotState:
        return BotState(
            weapon=MELEE_POOL[0],
            armor=ARMOR_TIERS[0],
            block_item=BLOCK_TYPES[0],
            ranged_spec=RANGED_TYPES[0],
            blocks=0,
            cobwebs=0,
            lava_buckets=0,
            water_buckets=0,
            has_shield=False,
            has_totem=False,
            gaps=0,
            splashes=0,
            pearls=0,
            buffs=0,
            ranged=0,
            pos=np.zeros(2, dtype=np.float64),
            vel=np.zeros(2, dtype=np.float64),
        )

    def _pick_different(self, n: int, avoid: int) -> int:
        if n <= 1 or avoid < 0:
            return int(self.np_random.integers(0, n))
        idx = int(self.np_random.integers(0, n - 1))
        if idx >= avoid:
            idx += 1
        return idx

    def _roll_loadout(
        self,
        pos: np.ndarray,
        avoid_weapon: int = -1,
        avoid_armor: int = -1,
        avoid_ranged: int = -1,
        forced_weapon: int | None = None,
    ) -> BotState:
        if forced_weapon is not None:
            w_idx = int(forced_weapon) % len(MELEE_POOL)
        else:
            w_idx = self._pick_different(len(MELEE_POOL), avoid_weapon)
        a_idx = self._pick_different(len(ARMOR_TIERS), avoid_armor)
        r_idx = self._pick_different(len(RANGED_TYPES), avoid_ranged)

        weapon = MELEE_POOL[w_idx]
        armor = ARMOR_TIERS[a_idx]
        block_item = BLOCK_TYPES[int(self.np_random.integers(0, len(BLOCK_TYPES)))]
        ranged_spec = RANGED_TYPES[r_idx]

        blocks = int(self.np_random.integers(4, 25)) if float(self.np_random.random()) < 0.85 else 0
        cobwebs = int(self.np_random.integers(1, 5)) if float(self.np_random.random()) < 0.55 else 0
        lava_buckets = 1 if float(self.np_random.random()) < 0.45 else 0
        water_buckets = 1 if float(self.np_random.random()) < 0.50 else 0

        has_shield = bool(float(self.np_random.random()) < 0.55)
        has_totem = bool((not has_shield) and (float(self.np_random.random()) < 0.20))

        heal_roll = int(self.np_random.integers(0, 5))
        has_primary_heals = float(self.np_random.random()) < 0.75
        if heal_roll == 3:  # golden_apple primary
            gaps = int(self.np_random.integers(1, 4)) if has_primary_heals else 0
            splashes = int(self.np_random.integers(1, 3)) if float(self.np_random.random()) < 0.45 else 0
        else:
            gaps = int(self.np_random.integers(1, 3)) if float(self.np_random.random()) < 0.45 else 0
            splashes = int(self.np_random.integers(1, 4)) if has_primary_heals else 0

        pearls = int(self.np_random.integers(1, 4)) if float(self.np_random.random()) < 0.55 else 0
        buffs = int(self.np_random.integers(1, 3)) if float(self.np_random.random()) < 0.50 else 0
        ranged = (
            int(self.np_random.integers(ranged_spec.min_count, ranged_spec.max_count + 1))
            if float(self.np_random.random()) < 0.75
            else 0
        )

        return BotState(
            weapon=weapon,
            armor=armor,
            block_item=block_item,
            ranged_spec=ranged_spec,
            blocks=blocks,
            cobwebs=cobwebs,
            lava_buckets=lava_buckets,
            water_buckets=water_buckets,
            has_shield=has_shield,
            has_totem=has_totem,
            gaps=gaps,
            splashes=splashes,
            pearls=pearls,
            buffs=buffs,
            ranged=ranged,
            pos=pos.copy(),
            vel=np.zeros(2, dtype=np.float64),
        )

    def _has_cover_between(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        if not self.obstacles:
            return False
        seg = p2 - p1
        seg_len_sq = float(np.dot(seg, seg))
        if seg_len_sq < 1e-6:
            return False
        for obs_pos, _ in self.obstacles:
            t = float(np.clip(np.dot(obs_pos - p1, seg) / seg_len_sq, 0.0, 1.0))
            proj = p1 + t * seg
            if float(np.linalg.norm(obs_pos - proj)) <= 1.25:
                return True
        return False

    def _push_out_of_obstacles(self, pos: np.ndarray) -> np.ndarray:
        out = pos
        for obs_pos, _ in self.obstacles:
            diff = out - obs_pos
            d = float(np.linalg.norm(diff))
            if d < 0.95:
                push_dir = diff / d if d > 1e-5 else np.array([1.0, 0.0], dtype=np.float64)
                out = obs_pos + push_dir * 0.95
        return np.clip(out, -self.ARENA_BOUND, self.ARENA_BOUND)

    def _get_obs(self) -> np.ndarray:
        L = self.learner
        E = self.enemy
        dx = float(E.pos[0] - L.pos[0])
        dz = float(E.pos[1] - L.pos[1])
        dist = math.hypot(dx, dz)
        l_speed = float(np.linalg.norm(L.vel)) * 0.05
        e_speed = float(np.linalg.norm(E.vel)) * 0.05

        obs = np.array(
            [
                np.clip((L.hp + L.absorption) / 20.0, 0.0, 1.5),
                np.clip((E.hp + E.absorption) / 20.0, 0.0, 1.5),
                np.clip(dist / 20.0, 0.0, 1.0),
                np.clip(l_speed, 0.0, 2.0),
                np.clip(e_speed, 0.0, 2.0),
                np.clip(dx / 20.0, -1.0, 1.0),
                np.clip(dz / 20.0, -1.0, 1.0),
                1.0 if L.y <= 1e-4 else 0.0,
                1.0 if L.cd <= 0.0 else 0.0,
                np.clip(self.step_count / self.max_steps, 0.0, 1.0),
                np.clip(L.weapon.damage / 14.0, 0.0, 1.0),
                np.clip((L.weapon.cooldown * 1000.0) / 1200.0, 0.0, 1.0),
                np.clip(E.weapon.damage / 14.0, 0.0, 1.0),
                np.clip(L.armor.reduction, 0.0, 1.0),
                np.clip(E.armor.reduction, 0.0, 1.0),
                np.clip(L.gaps / 5.0, 0.0, 1.0),
                np.clip(L.splashes / 5.0, 0.0, 1.0),
                np.clip(L.pearls / 4.0, 0.0, 1.0),
                np.clip((L.buffs + L.ranged) / 12.0, 0.0, 1.0),
                1.0 if (L.has_shield and L.shield_up and L.shield_disabled <= 0.0) else 0.0,
                1.0 if (E.has_shield and E.shield_up and E.shield_disabled <= 0.0) else 0.0,
                np.clip(L.blocks / 24.0, 0.0, 1.0),
                np.clip((L.cobwebs + L.lava_buckets + L.water_buckets) / 6.0, 0.0, 1.0),
                np.clip(len(self.obstacles) / 6.0, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        return obs

    def _apply_bot_damage(
        self, target: BotState, raw_dmg: float, bypass_armor: bool = False
    ) -> tuple[float, bool]:
        """Applies armor mitigation, absorption, HP loss, and Totem of Undying pop."""
        if raw_dmg <= 0.0:
            return 0.0, False
        mitigated = raw_dmg if bypass_armor else raw_dmg * (1.0 - target.armor.reduction * 0.78)
        used_abs = min(target.absorption, mitigated)
        rem = mitigated - used_abs
        old_hp = target.hp
        new_hp = max(0.0, old_hp - rem)
        totem_popped = False
        if new_hp <= self.DEATH_HP_THRESHOLD and target.has_totem:
            # Totem of Undying pops on fatal blow!
            target.has_totem = False
            new_hp = 11.0
            target.absorption = 4.0
            target.regen_timer = 4.5
            target.fire_timer = 0.0
            target.lava_timer = 0.0
            totem_popped = True
        target.hp = new_hp
        target.absorption = max(0.0, target.absorption - used_abs) if not totem_popped else target.absorption
        total_lost = max(0.0, used_abs + max(0.0, old_hp - new_hp))
        return total_lost, totem_popped

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        angle = float(self.np_random.uniform(0.0, 2.0 * math.pi))
        sep = float(self.np_random.uniform(5.0, 11.0))
        center = self.np_random.uniform(-2.5, 2.5, size=2)
        offset = np.array([math.cos(angle), math.sin(angle)]) * (sep * 0.5)

        l_pos = np.clip(center - offset, -self.ARENA_BOUND, self.ARENA_BOUND)
        e_pos = np.clip(center + offset, -self.ARENA_BOUND, self.ARENA_BOUND)

        forced_w = int(options["kit"]) if (options and "kit" in options) else None
        self.learner = self._roll_loadout(l_pos, forced_weapon=forced_w)
        # Enemy ALWAYS receives different items/weapon/armor/projectiles from Learner!
        self.enemy = self._roll_loadout(
            e_pos,
            avoid_weapon=self.learner.weapon.idx,
            avoid_armor=self.learner.armor.idx,
            avoid_ranged=self.learner.ranged_spec.idx,
        )

        self.obstacles = []
        self.step_count = 0
        self.consumables_used = {}

        obs = self._get_obs()
        self._prev_dist = float(obs[2]) * 20.0
        return obs, {
            "ready": True,
            "learner_kit": self.learner.summary,
            "enemy_kit": self.enemy.summary,
            "learner_weapon": self.learner.weapon.weapon_id,
            "enemy_weapon": self.enemy.weapon.weapon_id,
            "learner_armor": self.learner.armor.name,
            "enemy_armor": self.enemy.armor.name,
        }

    def _speed_multiplier(self, bot: BotState) -> float:
        mult = 1.35 if bot.buff_timer > 0.0 else 1.0
        if bot.slow_timer > 0.0:
            mult *= 0.72
        if bot.cobweb_timer > 0.0:
            mult *= 0.22  # Stuck in cobweb!
        if bot.lava_timer > 0.0:
            mult *= 0.50
        return mult

    def step(self, action: int | np.ndarray):
        act = int(np.asarray(action).item())
        self.step_count += 1

        L = self.learner
        E = self.enemy

        vec = E.pos - L.pos
        dist = float(np.linalg.norm(vec))
        fwd = vec / dist if dist > 1e-6 else np.array([0.0, 1.0], dtype=np.float64)
        right = np.array([-fwd[1], fwd[0]], dtype=np.float64)

        walk_speed = 4.317
        sprint_speed = 5.612
        l_speed_mult = self._speed_multiplier(L)
        e_speed_mult = self._speed_multiplier(E)

        move_dir = np.zeros(2, dtype=np.float64)
        speed = walk_speed * l_speed_mult
        want_jump = False
        want_attack = False
        sprinting = False

        used_consumable: str | None = None
        placed_block: str | None = None
        wasted_consumable = False
        shield_broken = False
        blocked_hit = False
        trapped_enemy = False
        tactical_cover = False
        totem_popped_learner = False
        totem_popped_enemy = False
        healed = 0.0
        damage_dealt = 0.0
        damage_taken = 0.0

        L.shield_up = False

        if act == 1:
            move_dir = fwd
        elif act == 2:
            move_dir = -fwd
        elif act == 3:
            move_dir = -right
        elif act == 4:
            move_dir = right
        elif act == 5:
            move_dir = fwd
            want_attack = True
        elif act == 6:
            move_dir = -right
            want_attack = True
        elif act == 7:
            move_dir = right
            want_attack = True
        elif act == 8:
            want_jump = True
        elif act == 9:
            move_dir = fwd
            speed = sprint_speed * l_speed_mult
            sprinting = True
            want_jump = True
            want_attack = True
        elif act == 10:
            move_dir = fwd
            speed = sprint_speed * l_speed_mult
            sprinting = True
        elif act == 11:
            move_dir = -fwd
            want_attack = True
        elif act == 12:
            # Raise Shield (requires having a shield in random loadout!)
            if L.has_shield and L.shield_disabled <= 0.0:
                L.shield_up = True
                move_dir = -fwd * 0.45
            else:
                wasted_consumable = True
        elif act == 13:
            # Eat Golden Apple
            if L.gaps > 0 and L.use_lock <= 0.0:
                if L.hp >= 19.5 and L.absorption >= 2.0:
                    wasted_consumable = True
                L.gaps -= 1
                L.regen_timer = 5.0
                prev_eff = L.hp + L.absorption
                L.absorption = 4.0
                healed += max(0.0, (L.hp + L.absorption) - prev_eff)
                L.use_lock = 0.35
                move_dir = -fwd * 0.35
                used_consumable = "golden_apple"
            else:
                wasted_consumable = True
        elif act == 14:
            # Instant Heal / Splash / Food
            if L.splashes > 0 and L.use_lock <= 0.0:
                if L.hp >= 18.5:
                    wasted_consumable = True
                L.splashes -= 1
                before_hp = L.hp
                L.hp = min(20.0, L.hp + 7.5)
                healed += L.hp - before_hp
                if dist < 2.1 and not self._has_cover_between(L.pos, E.pos):
                    E.hp = min(20.0, E.hp + 2.5)
                L.use_lock = 0.2
                used_consumable = "instant_heal"
            else:
                wasted_consumable = True
        elif act == 15:
            # Ender Pearl blink
            if L.pearls > 0 and L.pearl_cd <= 0.0:
                L.pearls -= 1
                L.pearl_cd = 1.5
                L.cobweb_timer = 0.0
                L.lava_timer = 0.0
                if L.hp <= 12.0 and (L.splashes > 0 or L.gaps > 0 or L.blocks >= 2):
                    target_blink = E.pos - fwd * 8.0
                elif dist > 3.8:
                    target_blink = E.pos - fwd * 2.1
                else:
                    wasted_consumable = True
                    target_blink = L.pos - fwd * 4.0
                L.pos = np.clip(target_blink, -self.ARENA_BOUND, self.ARENA_BOUND)
                pearl_dmg, t_pop = self._apply_bot_damage(L, 1.5, bypass_armor=True)
                damage_taken += pearl_dmg
                totem_popped_learner = totem_popped_learner or t_pop
                used_consumable = "ender_pearl"
            else:
                wasted_consumable = True
        elif act == 16:
            # Buff Potion (Speed II + Strength)
            if L.buffs > 0 and L.buff_timer <= 1.0 and L.use_lock <= 0.0:
                L.buffs -= 1
                L.buff_timer = 15.0
                L.use_lock = 0.25
                move_dir = -fwd * 0.4
                used_consumable = "buff_potion"
            else:
                wasted_consumable = True
        elif act == 17:
            # Asymmetric Ranged Item (snowball / egg / crossbow / splash_harming)
            if L.ranged > 0 and L.use_lock <= 0.0:
                spec = L.ranged_spec
                if dist < 1.8 or dist > 16.0:
                    wasted_consumable = True
                L.ranged -= 1
                L.use_lock = 0.15
                used_consumable = spec.ranged_id
                if 1.6 <= dist <= 16.0 and not self._has_cover_between(L.pos, E.pos):
                    if E.has_shield and E.shield_up and E.shield_disabled <= 0.0 and not spec.pierces_armor:
                        pass
                    else:
                        dealt, e_pop = self._apply_bot_damage(
                            E, spec.damage, bypass_armor=spec.pierces_armor
                        )
                        damage_dealt += dealt
                        totem_popped_enemy = totem_popped_enemy or e_pop
                        E.slow_timer = max(E.slow_timer, spec.slow_sec)
                        E.vel = E.vel + fwd * spec.knockback
            else:
                wasted_consumable = True
        elif act == 18:
            # Place Tactical Block Barricade / Pillar cover between Learner and Enemy
            if L.blocks >= 2 and L.use_lock <= 0.0 and 1.6 <= dist <= 7.5 and len(self.obstacles) < 6:
                L.blocks -= 2
                L.use_lock = 0.25
                mid = np.clip((L.pos + E.pos) * 0.5, -8.0, 8.0)
                self.obstacles.append((mid, 12.0))
                placed_block = L.block_item
                used_consumable = "block_barricade"
                move_dir = -fwd * 0.65
                if L.hp <= 15.5 or E.weapon.damage > L.weapon.damage + 2.0 or E.ranged > 0:
                    tactical_cover = True
            else:
                wasted_consumable = True
        elif act == 19:
            # Place Cobweb or Lava Trap at Enemy position (must aim close!)
            if L.cobwebs > 0 and L.use_lock <= 0.0 and dist <= 3.6:
                L.cobwebs -= 1
                L.use_lock = 0.25
                E.cobweb_timer = 2.6
                placed_block = "minecraft:cobweb"
                used_consumable = "cobweb_trap"
                trapped_enemy = True
            elif L.lava_buckets > 0 and L.use_lock <= 0.0 and 1.9 <= dist <= 3.6:
                L.lava_buckets -= 1
                L.use_lock = 0.30
                E.lava_timer = 1.6
                E.fire_timer = max(E.fire_timer, 3.5)
                if dist < 2.2:
                    # Placing lava too close splashes burn onto self!
                    L.fire_timer = max(L.fire_timer, 2.5)
                placed_block = "minecraft:lava"
                used_consumable = "lava_trap"
                trapped_enemy = True
                move_dir = -fwd * 0.7
            else:
                wasted_consumable = True
        elif act == 20:
            # Water Bucket (extinguish fire/lava, clear cobweb, push close enemy back!)
            if L.water_buckets > 0 and L.use_lock <= 0.0:
                had_hazard = (L.fire_timer > 0.0 or L.lava_timer > 0.0 or L.cobweb_timer > 0.0)
                close_push = dist <= 3.4
                if not had_hazard and not close_push:
                    wasted_consumable = True
                L.water_buckets -= 1
                L.use_lock = 0.25
                L.fire_timer = 0.0
                L.lava_timer = 0.0
                L.cobweb_timer = 0.0
                if close_push:
                    E.vel = E.vel + fwd * 4.5
                    E.slow_timer = max(E.slow_timer, 1.2)
                used_consumable = "water_bucket"
            else:
                wasted_consumable = True
        elif act == 21:
            # Pillar-up / High-ground block jump under feet
            if L.blocks > 0 and L.use_lock <= 0.0 and L.y <= 1e-4 and L.pillar_timer <= 0.0:
                if dist > 4.5 and L.ranged <= 0:
                    wasted_consumable = True
                L.blocks -= 1
                L.use_lock = 0.20
                want_jump = True
                L.pillar_timer = 1.8  # Temporary +1 block elevation advantage
                placed_block = L.block_item
                used_consumable = "pillar_jump"
            else:
                wasted_consumable = True

        if used_consumable is not None:
            self.consumables_used[used_consumable] = (
                self.consumables_used.get(used_consumable, 0) + 1
            )

        # Learner vertical physics
        if want_jump and L.y <= 1e-4 and L.cobweb_timer <= 0.0:
            L.vy = 8.4
        prev_l_vy = L.vy
        L.y += L.vy * self.DT
        if L.y > 0.0:
            L.vy -= 28.0 * self.DT
        if L.y <= 0.0:
            L.y = 0.0
            L.vy = 0.0

        # Asymmetric Tactical Enemy AI (uses its own distinct random items, blocks, traps, shield & heals)
        e_fwd = -fwd
        e_right = -right
        e_move = np.zeros(2, dtype=np.float64)
        e_speed = walk_speed * e_speed_mult
        E.shield_up = False
        enemy_acted_utility = False

        if (E.fire_timer > 1.0 or E.cobweb_timer > 0.5) and E.water_buckets > 0 and float(self.np_random.random()) < 0.45:
            E.water_buckets -= 1
            E.fire_timer = 0.0
            E.lava_timer = 0.0
            E.cobweb_timer = 0.0
            enemy_acted_utility = True
        elif E.hp <= 12.0 and E.splashes > 0 and float(self.np_random.random()) < 0.35:
            E.splashes -= 1
            E.hp = min(20.0, E.hp + 7.5)
            enemy_acted_utility = True
        elif (
            E.hp <= 14.0
            and E.blocks >= 2
            and dist < 4.5
            and len(self.obstacles) < 6
            and float(self.np_random.random()) < 0.25
        ):
            E.blocks -= 2
            mid = np.clip((L.pos + E.pos) * 0.5, -8.0, 8.0)
            self.obstacles.append((mid, 14.0))
            e_move = -e_fwd * 0.8
            enemy_acted_utility = True
        elif (
            E.hp <= 15.0
            and E.gaps > 0
            and E.regen_timer <= 0.0
            and (dist > 2.8 or self._has_cover_between(L.pos, E.pos))
            and float(self.np_random.random()) < 0.28
        ):
            E.gaps -= 1
            E.regen_timer = 5.0
            E.absorption = 4.0
            enemy_acted_utility = True
        elif E.cobwebs > 0 and dist <= 3.8 and L.cobweb_timer <= 0.0 and float(self.np_random.random()) < 0.24:
            E.cobwebs -= 1
            L.cobweb_timer = 2.6
            enemy_acted_utility = True
        elif E.lava_buckets > 0 and 2.1 <= dist <= 4.0 and float(self.np_random.random()) < 0.20:
            E.lava_buckets -= 1
            L.lava_timer = 1.6
            L.fire_timer = max(L.fire_timer, 3.5)
            enemy_acted_utility = True
        elif E.buffs > 0 and E.buff_timer <= 0.0 and dist > 4.2 and float(self.np_random.random()) < 0.24:
            E.buffs -= 1
            E.buff_timer = 15.0
            enemy_acted_utility = True
        elif (
            E.pearls > 0
            and E.pearl_cd <= 0.0
            and dist > 7.0
            and E.hp > 12.0
            and float(self.np_random.random()) < 0.20
        ):
            E.pearls -= 1
            E.pearl_cd = 2.0
            E.cobweb_timer = 0.0
            E.pos = np.clip(L.pos + fwd * 2.2, -self.ARENA_BOUND, self.ARENA_BOUND)
            self._apply_bot_damage(E, 1.5, bypass_armor=True)
            enemy_acted_utility = True

        # Enemy shield & movement
        if (
            not enemy_acted_utility
            and E.has_shield
            and E.shield_disabled <= 0.0
            and dist < 3.6
            and L.cd <= 0.15
            and float(self.np_random.random()) < 0.22
        ):
            E.shield_up = True
            e_move = -e_fwd * 0.4
        else:
            if dist > 2.5:
                e_move += e_fwd
                if self._has_cover_between(E.pos, L.pos):
                    # Enemy strafes around placed block barricades!
                    e_move += e_right * 0.85
                elif dist > 4.5:
                    e_speed = sprint_speed * e_speed_mult
            if float(self.np_random.random()) < 0.16:
                e_move -= e_right
            if float(self.np_random.random()) < 0.16:
                e_move += e_right
            if float(self.np_random.random()) < 0.11 and E.y <= 1e-4 and E.cobweb_timer <= 0.0:
                E.vy = 8.4

        e_norm = float(np.linalg.norm(e_move))
        if e_norm > 1.0:
            e_move /= e_norm

        E.y += E.vy * self.DT
        if E.y > 0.0:
            E.vy -= 28.0 * self.DT
        if E.y <= 0.0:
            E.y = 0.0
            E.vy = 0.0

        # Horizontal velocity, block obstacle collision & arena bounds
        L.vel = L.vel * 0.45 + move_dir * speed * 0.55
        E.vel = E.vel * 0.45 + e_move * e_speed * 0.55
        L.pos = self._push_out_of_obstacles(
            np.clip(L.pos + L.vel * self.DT, -self.ARENA_BOUND, self.ARENA_BOUND)
        )
        E.pos = self._push_out_of_obstacles(
            np.clip(E.pos + E.vel * self.DT, -self.ARENA_BOUND, self.ARENA_BOUND)
        )

        # Tick obstacles & bot timers
        self.obstacles = [
            (op, ttl - self.DT) for (op, ttl) in self.obstacles if ttl - self.DT > 0.0
        ]
        for b in (L, E):
            b.cd = max(0.0, b.cd - self.DT)
            b.hurt_cd = max(0.0, b.hurt_cd - self.DT)
            b.shield_disabled = max(0.0, b.shield_disabled - self.DT)
            b.buff_timer = max(0.0, b.buff_timer - self.DT)
            b.slow_timer = max(0.0, b.slow_timer - self.DT)
            b.cobweb_timer = max(0.0, b.cobweb_timer - self.DT)
            b.pillar_timer = max(0.0, b.pillar_timer - self.DT)
            b.pearl_cd = max(0.0, b.pearl_cd - self.DT)
            b.use_lock = max(0.0, b.use_lock - self.DT)

        # Regeneration ticks
        if L.regen_timer > 0.0:
            L.regen_timer = max(0.0, L.regen_timer - self.DT)
            before = L.hp
            L.hp = min(20.0, L.hp + 0.35)
            healed += L.hp - before
        if E.regen_timer > 0.0:
            E.regen_timer = max(0.0, E.regen_timer - self.DT)
            E.hp = min(20.0, E.hp + 0.35)

        # Lava & Fire damage ticks
        if L.lava_timer > 0.0:
            L.lava_timer = max(0.0, L.lava_timer - self.DT)
            taken, t_pop = self._apply_bot_damage(L, 0.45, bypass_armor=False)
            damage_taken += taken
            totem_popped_learner = totem_popped_learner or t_pop
        if E.lava_timer > 0.0:
            E.lava_timer = max(0.0, E.lava_timer - self.DT)
            dealt, e_pop = self._apply_bot_damage(E, 0.45, bypass_armor=False)
            damage_dealt += dealt
            totem_popped_enemy = totem_popped_enemy or e_pop

        if L.fire_timer > 0.0:
            L.fire_timer = max(0.0, L.fire_timer - self.DT)
            taken, t_pop = self._apply_bot_damage(L, 0.12, bypass_armor=True)
            damage_taken += taken
            totem_popped_learner = totem_popped_learner or t_pop
        if E.fire_timer > 0.0:
            E.fire_timer = max(0.0, E.fire_timer - self.DT)
            dealt, e_pop = self._apply_bot_damage(E, 0.12, bypass_armor=True)
            damage_dealt += dealt
            totem_popped_enemy = totem_popped_enemy or e_pop

        new_vec = E.pos - L.pos
        new_dist = float(np.linalg.norm(new_vec))
        hit_dir = new_vec / new_dist if new_dist > 1e-6 else fwd
        cover_blocking = self._has_cover_between(L.pos, E.pos)
        eff_reach = self.REACH + (0.30 if L.pillar_timer > 0.0 else 0.0)

        # Learner melee attack
        if (
            want_attack
            and L.use_lock <= 0.0
            and L.cd <= 0.03
            and new_dist <= eff_reach
            and (not cover_blocking or L.pillar_timer > 0.0)
        ):
            if E.has_shield and E.shield_up and E.shield_disabled <= 0.0:
                if L.weapon.shield_break:
                    E.shield_up = False
                    E.shield_disabled = 5.0
                    shield_broken = True
                    E.vel = E.vel + hit_dir * 3.0
                else:
                    L.vel = L.vel - hit_dir * 2.0
            elif E.hurt_cd <= 0.0:
                is_crit = (L.y > 0.05 and prev_l_vy < 0.0 and not sprinting) or (
                    L.pillar_timer > 0.0 and not sprinting
                )
                mult = 1.5 if is_crit else 1.0
                str_bonus = 3.0 if L.buff_timer > 0.0 else 0.0
                raw_dmg = (L.weapon.damage + str_bonus) * mult
                dealt, e_pop = self._apply_bot_damage(E, raw_dmg, bypass_armor=False)
                damage_dealt += dealt
                totem_popped_enemy = totem_popped_enemy or e_pop
                E.hurt_cd = 0.5
                if L.weapon.fire_seconds > 0.0:
                    E.fire_timer = max(E.fire_timer, L.weapon.fire_seconds)
                base_kb = 4.4 * L.weapon.kb_mult * (1.35 if sprinting else 1.0)
                E.vel = E.vel + hit_dir * base_kb
            L.cd = L.weapon.cooldown

        # Enemy melee / ranged attack
        re_dist = float(np.linalg.norm(E.pos - L.pos))
        cover_blocking_after = self._has_cover_between(E.pos, L.pos)
        if (
            not enemy_acted_utility
            and not E.shield_up
            and re_dist < 3.25
            and not cover_blocking_after
            and E.cd <= 0.0
            and float(self.np_random.random()) < 0.72
        ):
            if L.has_shield and L.shield_up and L.shield_disabled <= 0.0:
                if E.weapon.shield_break:
                    L.shield_up = False
                    L.shield_disabled = 5.0
                    blocked_hit = True
                else:
                    blocked_hit = True
                    E.vel = E.vel + hit_dir * 2.5
            elif L.hurt_cd <= 0.0:
                e_bonus = 3.0 if E.buff_timer > 0.0 else 0.0
                raw_dmg = E.weapon.damage + e_bonus
                taken, l_pop = self._apply_bot_damage(L, raw_dmg, bypass_armor=False)
                damage_taken += taken
                totem_popped_learner = totem_popped_learner or l_pop
                L.hurt_cd = 0.5
                if E.weapon.fire_seconds > 0.0:
                    L.fire_timer = max(L.fire_timer, E.weapon.fire_seconds)
                L.vel = L.vel - hit_dir * (4.4 * E.weapon.kb_mult)
            E.cd = E.weapon.cooldown
        elif (
            not enemy_acted_utility
            and E.ranged > 0
            and 4.5 <= re_dist <= 13.5
            and E.cd <= 0.0
            and float(self.np_random.random()) < 0.22
        ):
            E.ranged -= 1
            E.cd = 0.9
            if cover_blocking_after:
                # Learner's placed block barricade absorbed the enemy's projectile!
                blocked_hit = True
            elif L.has_shield and L.shield_up and L.shield_disabled <= 0.0 and not E.ranged_spec.pierces_armor:
                blocked_hit = True
            else:
                taken, l_pop = self._apply_bot_damage(
                    L, E.ranged_spec.damage, bypass_armor=E.ranged_spec.pierces_armor
                )
                damage_taken += taken
                totem_popped_learner = totem_popped_learner or l_pop
                L.slow_timer = max(L.slow_timer, E.ranged_spec.slow_sec)
                L.vel = L.vel - hit_dir * E.ranged_spec.knockback

        learner_dead = L.hp <= self.DEATH_HP_THRESHOLD
        enemy_dead = E.hp <= self.DEATH_HP_THRESHOLD
        timeout = self.step_count >= self.max_steps

        terminated = bool((learner_dead or enemy_dead) and not timeout)
        truncated = bool(timeout and not terminated)

        obs = self._get_obs()

        reward = -0.01
        reward += damage_dealt * 2.5
        reward -= damage_taken * 1.3
        reward += healed * 0.4

        if shield_broken:
            reward += 4.0
        if blocked_hit:
            reward += 2.0
        if trapped_enemy:
            reward += 3.0
        if tactical_cover:
            reward += 2.0
        if wasted_consumable:
            reward -= 1.5

        curr_dist = float(obs[2]) * 20.0
        if curr_dist > 3.0:
            reward += float(np.clip((self._prev_dist - curr_dist) * 0.05, -0.05, 0.05))
        self._prev_dist = curr_dist

        if enemy_dead:
            reward += 100.0
        if learner_dead:
            reward -= 100.0

        info = {
            "ready": True,
            "learner_kit": L.summary,
            "enemy_kit": E.summary,
            "learner_weapon": L.weapon.weapon_id,
            "enemy_weapon": E.weapon.weapon_id,
            "learner_armor": L.armor.name,
            "enemy_armor": E.armor.name,
            "damage_dealt": damage_dealt,
            "damage_taken": damage_taken,
            "healed": healed,
            "used_consumable": used_consumable,
            "placed_block": placed_block,
            "wasted_consumable": wasted_consumable,
            "shield_broken": shield_broken,
            "blocked_hit": blocked_hit,
            "trapped_enemy": trapped_enemy,
            "tactical_cover": tactical_cover,
            "totem_popped_learner": totem_popped_learner,
            "totem_popped_enemy": totem_popped_enemy,
            "consumables_summary": dict(self.consumables_used),
            "learner_hp": L.hp,
            "enemy_hp": E.hp,
            "distance": curr_dist,
            "learner_dead": learner_dead,
            "enemy_dead": enemy_dead,
            "timeout": timeout,
        }
        return obs, float(reward), terminated, truncated, info
