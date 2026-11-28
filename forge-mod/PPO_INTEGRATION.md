# PPO actor integration (Minecraft Forge 1.16.5)

## Model and runtime

The exported actor has 52 features: the terrain actor's 44 inputs, four crowd
summaries, the nearest non-target opponent's relative X/Z offsets, and a weighted
crowd-bearing vector. The four added input columns were zero-initialized in the
warm-start seed, preserving the best 48-feature actor's logits at initialization; the
final 50,000-fight actor has since learned their values. The architecture is
`52 -> Linear(192) -> Tanh -> Linear(192) -> Tanh -> Linear(30)`. Forge CI must rebuild
the JAR after the final weight export.

The 52-feature run slightly improved wins against groups of three through five, but
lost ground against six and seven and tied at eight versus the AoE baseline. No stable
multi-opponent win rate has been demonstrated, and live Minecraft group combat remains
unverified.

`PpoPolicy` reads `src/main/resources/com/arena/aipilot/ppo_policy.bin` and selects
`argmax(logits)` in Java. Python, PyTorch, Stable-Baselines3, and network access are
not needed in the game. The client runs one inference every two Minecraft ticks
(10 decisions per second, matching the arena's `DT = 0.10`).

To regenerate and verify the embedded weights after updating the checkpoint:

```bash
.venv/bin/python forge-mod/tools/export_ppo_weights.py \\
  --model models/terrain_finetune/pvp_ppo_terrain_multi_aggressive_pressure_50k.zip
.venv/bin/python forge-mod/tools/verify_ppo_weights.py \\
  --model models/terrain_finetune/pvp_ppo_terrain_multi_aggressive_pressure_50k.zip
```

The verifier compares all 30 logits against Stable-Baselines3 on seeded
observations and prints the resource SHA-256. If the resource is missing or has an
incompatible header/size, the mod fails closed:
AI combat is disabled; it does not fall back to the old heuristic.

Compatibility note: the packaged 50,000-fight actor predates the current action-9
jump-crit mapping; it was trained when action 9 was sprint-hit. The simulator and
Java execution now agree on jump-crit, but these packaged weights have not been
fine-tuned or evaluated after that action remap. Treat action-9 policy quality as
unverified until a compatible training run and benchmark are completed.

### Bow and crossbow training/runtime

Action 17 remains one shared ranged action, so the 52-feature actor shape and action
IDs do not change. `RANGED_TYPES` now includes bows. In both simulators a bow consumes
an arrow and releases after a 1.0-second draw; a crossbow consumes a bolt when its
1.25-second load completes, remains loaded, and fires on a later action 17. These
weapons reach 24 blocks, honor line-of-sight/cover and shields, and use separate
shot/hit flags. Action 17 pursues outside melee reach, matching Forge's navigation assist.
The ranged observation uses explicit availability codes (`0.75` bow, `0.85` crossbow,
`1.10` loaded crossbow, `1.25` active draw/load). Starting a draw/load and releasing a
projectile each add `+4` training reward; a clear, unblocked projectile contact (even
inside vanilla hurt cooldown) adds `+8` on top of the normal damage reward. The
multi-opponent training log/status reports `bow_hits/bow_shots` and
`crossbow_hits/crossbow_shots`.

Forge action 17 selects a loaded crossbow first, otherwise loads one or draws a bow
only when arrow ammo is present. The client releases after the vanilla 25/20-tick use
window, tracks the target while charging, and leads its horizontal/vertical motion to
compensate for projectile travel/drop.

A ranged-focused actor was warm-started from the prior 50k checkpoint, continued for
about 25.4k PPO fights, then trained for 30 behavior-cloning epochs on valid bow and
crossbow states while cloning the PPO actor on other states. The candidate checkpoint is
`models/terrain_finetune/pvp_ppo_terrain_multi_aggressive_bow_crossbow_25k_bc30.zip`.
In a deterministic 1,000-fight simulator evaluation (3–8 opponents, terrain mix 0.5,
seed 6069), it fired 531 bow shots (323 unblocked contacts) and 358 crossbow shots (356
contacts); the old actor fired neither. The candidate won 69 fights versus 62 for the old
actor in this single seed, which is not enough evidence to claim a win-rate improvement.
The actor was exported to `ppo_policy.bin`; the exporter matched SB3 on 8,192 observations
(max logit delta `5.72e-06`, zero action mismatches). This verifies the binary export, not
a live Forge match. The installed JAR has not been rebuilt, and the Forge runtime has not
yet been tested in-game.

## Observation compatibility

The legacy source contract is `env/pro_vec_env.py::_obs()` (`Box(-1, 2, (34,), float32)`).
Terrain fine-tuning appends ten sensors without changing those first 34 indices, for
`Box(44,)`. `env/multi_opponent.py` appends eight crowd/threat sensors for `Box(52,)`;
the original 44 positions and action IDs remain unchanged. Minecraft has no direct
counterpart for some simulator-only state, so the bridge uses the following measurements/estimates:

| Index | Training feature | Forge input |
|---:|---|---|
| 0 | learner `(hp + absorption) / 20` | local player's health plus absorption |
| 1 | enemy `(hp + absorption) / 20` | selected target health plus absorption |
| 2 | distance `/ 20` | horizontal X/Z distance to target `/ 20` |
| 3 | learner speed `norm(vel) * .05` | horizontal player velocity magnitude per tick |
| 4 | enemy speed `norm(vel) * .05` | horizontal target velocity magnitude per tick |
| 5 | relative X `/ 20` | `(target.x - player.x) / 20` |
| 6 | relative second arena axis `/ 20` | `(target.z - player.z) / 20` |
| 7 | learner on ground | `player.isOnGround()` |
| 8 | melee cooldown ready | attack-strength scale at least `0.95` |
| 9 | arena step `/ 600` | PPO decisions since target acquisition `/ 600` |
| 10 | learner weapon damage `/ 14` | selected combat weapon's attack-damage modifiers |
| 11 | learner weapon cooldown `/ 1.2` | inverse attack speed for selected combat weapon |
| 12 | enemy weapon damage `/ 14` | target's visible main-hand attack damage |
| 13 | learner armor tier reduction | armor points mapped to the simulator's five reduction tiers |
| 14 | enemy armor tier reduction | same mapping for target armor points |
| 15 | golden apples `/ 5` | golden and enchanted golden apples in inventory |
| 16 | instant-heal potions `/ 5` | splash potions carrying `HEAL` |
| 17 | pearls `/ 4` | ender pearls in inventory |
| 18 | ranged availability/status | generic ranged-resource count capped at `0.60`; `0.75` bow + arrows, `0.85` crossbow + arrows, `1.10` loaded crossbow, `1.25` active draw/load; simulator and Forge use matching codes |
| 19 | learner shield up | local player actively blocking with a shield |
| 20 | enemy shield up | target actively blocking with a shield |
| 21 | generic blocks `/ 24` | block items, excluding obsidian, cobweb, and respawn anchor |
| 22 | cobweb + lava + water `/ 6` | corresponding inventory items |
| 23 | obstacle count `/ 6` | visibility-based `1/6` approximation; the new terrain sensors provide explicit local obstacles |
| 24 | end crystals `/ 6` | end crystals in inventory |
| 25 | anchors `/ 4` | respawn anchors in inventory; action 23 also requires glowstone and an explosive dimension |
| 26 | TNT minecarts `/ 4` | minecarts in inventory; action 24 also requires an activator rail and redstone source |
| 27 | enemy explosives `/ 10` | those three item types in a player target's replicated inventory |
| 28 | obsidian `/ 16` | obsidian in inventory |
| 29 | cage active | detected when at least four nearby cage-wall blocks are obsidian |
| 30 | learner combo taken `/ 5` | consecutive local health/absorption losses within 1.6 seconds |
| 31 | learner stun `/ 1s` | estimated remaining 0.45-second combo stun |
| 32 | enemy combo taken `/ 5` | target health/absorption losses near the local player |
| 33 | hurt cooldown `/ .5`, only in combo | Minecraft `hurtTime * .05`, gated by the estimated combo counter |
| 34 | forward clearance | nearest of four probes (0.75–3 blocks) that meets a wall, obstruction, or pit |
| 35 | left clearance | same probe, left of the target-facing direction |
| 36 | right clearance | same probe, right of the target-facing direction |
| 37 | rear clearance | same probe, behind the player |
| 38–41 | floor support, forward/left/right/back | solid-ground checks at 1.5 blocks; unsupported holes score zero |
| 42 | target height delta `/ 3` | target feet height relative to the player's feet |
| 43 | forward slope `/ 2` | sampled ground-height change 1.5 blocks ahead |
| 44 | alive opponents `/ 8` | count of valid nearby living targets, clipped to eight |
| 45 | opponents within melee reach `/ 8` | count within 3.25 horizontal blocks, clipped to eight |
| 46 | mean opponent distance `/ 20` | mean horizontal distance of the nearest eight valid targets |
| 47 | mean opponent health `/ 20` | mean health plus absorption of those targets |
| 48 | nearest non-target opponent X offset `/ 20` | `(nearestOther.x - player.x) / 20` |
| 49 | nearest non-target opponent second arena-axis offset `/ 20` | `(nearestOther.z - player.z) / 20` |
| 50 | weighted mean direction of non-target opponents, X | mean unit X direction with weight `1 / (1 + distance)` |
| 51 | weighted mean direction of non-target opponents, Z | mean unit Z direction with weight `1 / (1 + distance)` |

On target acquisition, the engine slightly favors weakened nearby targets after
accounting for distance and visibility. It also runs a short safe-direction search
when the forward probe is blocked, adds forward pressure outside melee range, and only
retreats automatically below the configured low-health threshold.

Observation counts include the full player inventory; utility items outside the
hotbar are swapped into a free or non-combat hotbar slot before use. The target's
private inventory may not be synchronized by every server. Armor tiers, simulator
stun/combos, obstacle count, and the custom cage flag are therefore approximations
rather than exact simulator state. This does not change the network's tensor shape or
feature order.

## Combat style controls

Press **K** in-game to open draggable sliders; values persist to
`config/aipilot-style.properties`.

- **Aggression** biases movement toward the target and discourages passive kiting.
- **Pursuit** controls how strongly the safe-route assist steers around terrain.
- **Strafing** adjusts lateral movement in close-range exchanges.
- **Caution** sets how readily the bot retreats at critically low health.

Press **Esc** to close the panel. Existing keys remain **O** (auto-combat) and **J** (HUD).

## Multi-opponent fine-tuning

`train_multi.py` uses completed fights (not simulation steps) for its exact quota.
Every fight has at least three simultaneous opponents and the curriculum reaches
eight. Larger groups are oversampled once available. The learner receives one
adjacent weapon tier and one armor tier when available, plus occasional extra
consumables; player aggression is `1.0`. The simulator applies explosive splash to
nearby opponents, focuses weakened nearby targets after a kill, and gives a stronger
terminal penalty for the learner's death than for ordinary damage.

```bash
.venv/bin/python train_multi.py \\
  --fights 50000 --exact-fights --n-envs 128 \\
  --min-opponents 3 --max-opponents 8 --player-aggression 1.0 \\
  --load models/terrain_finetune/pvp_ppo_terrain_multi_aggressive_pressure_seed.zip \\
  --save models/terrain_finetune/pvp_ppo_terrain_multi_aggressive_pressure_50k
.venv/bin/python forge-mod/tools/export_ppo_weights.py \\
  --model models/terrain_finetune/pvp_ppo_terrain_multi_aggressive_pressure_50k.zip
.venv/bin/python forge-mod/tools/verify_ppo_weights.py \\
  --model models/terrain_finetune/pvp_ppo_terrain_multi_aggressive_pressure_50k.zip
for opponents in 3 4 5 6 7 8; do
  .venv/bin/python play_multi.py --model models/terrain_finetune/pvp_ppo_terrain_multi_aggressive_pressure_50k.zip \\
    --episodes 200 --min-opponents "$opponents" --opponent-cap "$opponents" \\
    --skill 1.0 --terrain-mix 1.0 --seed 6061
done
```

The corrected 50,000-fight run recorded **8,395 wins, 41,447 losses, and 158 draws**.
For the best-tested checkpoint, deterministic evaluation at skill 1.0 used 200 fights
per group:

| Opponents | Wins | Losses | Draws | Win rate |
|---:|---:|---:|---:|---:|
| 3 | 58 | 140 | 2 | 29.0% |
| 4 | 30 | 170 | 0 | 15.0% |
| 5 | 22 | 178 | 0 | 11.0% |
| 6 | 12 | 188 | 0 | 6.0% |
| 7 | 4 | 196 | 0 | 2.0% |
| 8 | 1 | 199 | 0 | 0.5% |

This is not stable winning and does not meet the no-loss goal. Six later exact
50,000-fight refinements (hard-crowd sampling, target focus, win/death shaping,
weighted groups, full-difficulty training, and 50-feature threat positions) failed to
improve all sizes. The latest 52-feature crowd-bearing model won 63/200, 34/200,
23/200, 9/200, 2/200, and 1/200 against groups of three through eight. It raised the
aggregate by five wins over the AoE baseline but regressed against six and seven. The
full iteration table and machine-readable summary are in
`models/terrain_finetune/multi_opponent_iterations.md` and
`pvp_ppo_terrain_multi_aggressive_pressure_50k_summary.json`. Forge runtime group
combat remains unverified; Python and packaged-actor checks are not an in-game
multiplayer test.
## Action compatibility

The policy still chooses an integer in `Discrete(30)`; Java maps that ID to the
matching control/item intent. Standard Minecraft client interactions are used—no
server commands or custom arena protocol:

| ID | Training action | Forge execution |
|---:|---|---|
| 0 | no-op | no direct input; the navigation assist may pursue when the area is clear |
| 1–4 | forward, back, left, right | movement relative to the target-facing camera; back/strafe intents are preserved |
| 5–7 | forward/left/right + attack | corresponding movement plus charged melee attack |
| 8 | jump | jump input without forced forward movement |
| 9 | jump-crit | jump without sprint, then send one charged attack during the falling window |
| 10 | sprint + forward | same movement |
| 11 | back + attack | retreating melee swing |
| 12 | shield | hold an available shield and retreat; inventory shields can be swapped to the hotbar |
| 13 | golden apple | eat a golden/enchanted golden apple from inventory or hotbar |
| 14 | instant heal | throw an available splash `HEAL` potion toward the player's feet |
| 15 | ender pearl | throw toward the target normally; on an escape, aim away and retreat |
| 16 | buff potion | drink a strength or speed potion from inventory or hotbar |
| 17 | ranged | use a harming potion, snowball, egg, bow, or crossbow; bow draws for 20 ticks, an uncharged crossbow loads for 25 ticks and then fires on a later action 17; arrows are checked, projectiles are aimed ahead of target velocity, and inventory items can be retrieved |
| 18 | barricade | place an available solid block between player and target, then withdraw |
| 19 | cobweb/lava trap | place a cobweb, otherwise pour a lava bucket near target |
| 20 | water bucket | place water near a close target or at the player's feet to clear a hazard |
| 21 | pillar jump | jump and try to place an available block adjacent to the player |
| 22 | end crystal | place obsidian/crystal, then retreat while vanilla combat attacks the crystal |
| 23 | respawn anchor | only in a dimension where anchors explode; charge with glowstone, activate, then retreat |
| 24 | TNT minecart | place an activator rail and cart, power it with redstone, then retreat from the fuse |
| 25 | obsidian cage | try to place four obsidian walls around player |
| 26 | W-tap combo | reset sprint before the charged vanilla attack |
| 27 | pearl + crystal | throw pearl, then attempt obsidian/crystal placement and withdraw |
| 28 | combo break | reset the local chain only after a shield, water bucket, or pearl is actually used |
| 29 | combo counter | attack during the locally estimated counter/stun window |

Utility items and weapons outside the hotbar are moved with a standard vanilla inventory
`SWAP` click while the inventory screen is closed; the selected combat weapon is kept
out of the swap slot. Action 23 is suppressed unless glowstone is available in a
dimension where respawn anchors explode (`respawnAnchorWorks() == false`). Action 24 requires an activator rail and a redstone
source in addition to the TNT minecart. Crystal/anchor/minecart blasts retain real
Minecraft self-damage and placement constraints; W-tap has no simulator-only damage
or knockback bonus, and combo stun/counter windows are estimated from observed health
changes. These client-side checks and CI compilation do not substitute for a live
multiplayer combat test; server-specific effects may differ from `ProArena`.
