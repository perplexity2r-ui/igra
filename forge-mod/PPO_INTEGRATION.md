# PPO actor integration (Minecraft Forge 1.16.5)

## Model and runtime

The current Forge build embeds the deterministic 48-feature actor fine-tuned for
one-versus-many fights from `models/terrain_finetune/pvp_ppo_terrain_multi_aggressive_50k.zip`.
It retains the terrain actor's 44 inputs and appends four crowd-context sensors; the
architecture is `48 -> Linear(192) -> Tanh -> Linear(192) -> Tanh -> Linear(30)`.
`PpoPolicy` reads `src/main/resources/com/arena/aipilot/ppo_policy.bin` and selects
`argmax(logits)` in Java. Python, PyTorch, Stable-Baselines3, and network access are
not needed in the game. The client runs one inference every two Minecraft ticks
(10 decisions per second, matching the arena's `DT = 0.10`).

To regenerate and verify the embedded weights after updating the checkpoint:

```bash
.venv/bin/python forge-mod/tools/export_ppo_weights.py
.venv/bin/python forge-mod/tools/verify_ppo_weights.py
```

The verifier compares all 30 logits against Stable-Baselines3 on seeded
observations and prints the resource SHA-256. If the resource is missing or has an
incompatible header/size, the mod fails closed:
AI combat is disabled; it does not fall back to the old heuristic.

## Observation compatibility

The legacy source contract is `env/pro_vec_env.py::_obs()` (`Box(-1, 2, (34,), float32)`).
Terrain fine-tuning appends ten sensors without changing those first 34 indices, for
`Box(44,)`. `env/multi_opponent.py` appends four crowd sensors for `Box(48,)`; the
original 44 positions and action IDs remain unchanged. Minecraft has no direct
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
| 18 | `(buffs + ranged) / 12` | speed/strength potions plus arrows, snowballs, eggs, harming splash potions, and crossbows |
| 19 | learner shield up | local player actively blocking with a shield |
| 20 | enemy shield up | target actively blocking with a shield |
| 21 | generic blocks `/ 24` | block items, excluding obsidian, cobweb, and respawn anchor |
| 22 | cobweb + lava + water `/ 6` | corresponding inventory items |
| 23 | obstacle count `/ 6` | visibility-based `1/6` approximation; the new terrain sensors provide explicit local obstacles |
| 24 | end crystals `/ 6` | end crystals in inventory |
| 25 | anchors `/ 4` | respawn anchors in inventory |
| 26 | TNT minecarts `/ 4` | TNT minecarts in inventory |
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

The engine also runs a short safe-direction search when the forward probe is blocked,
adds forward pressure outside melee range, and only retreats automatically below the
configured low-health threshold.

Inventory counts include the full player inventory, while actions can only select
hotbar items; the target's private inventory may not be synchronized by every server.
Armor tiers, simulator stun/combos, obstacle count, and the custom cage flag are
therefore approximations rather than exact simulator state. This does not change the
network's tensor shape or feature order.

## Combat style controls

Press **K** in-game to open draggable sliders; values persist to
`config/aipilot-style.properties`.

- **Aggression** biases movement toward the target and discourages passive kiting.
- **Pursuit** controls how strongly the safe-route assist steers around terrain.
- **Strafing** adjusts lateral movement in close-range exchanges.
- **Caution** sets how readily the bot retreats at critically low health.

Press **Esc** to close the panel. Existing keys remain **O** (auto-combat) and **J** (HUD).

## Multi-opponent fine-tuning

`train_multi.py` warm-starts the 48-feature actor from the aggressive terrain model.
Every fight has at least three simultaneously active opponents; the curriculum ramps
up to eight. The learner gets a small gear/resource edge on average, the opponents
use skill-scaled kits, and player aggression remains at `1.0`. Exact quotas
freeze each vector row after its allocated number of completed fights.

```bash
.venv/bin/python train_multi.py \\
  --fights 50000 --exact-fights --n-envs 128 \\
  --min-opponents 3 --max-opponents 8 --player-aggression 1.0 \\
  --load models/terrain_finetune/pvp_ppo_terrain_aggressive_50k.zip \\
  --save models/terrain_finetune/pvp_ppo_terrain_multi_aggressive_50k
.venv/bin/python forge-mod/tools/export_ppo_weights.py
.venv/bin/python forge-mod/tools/verify_ppo_weights.py
.venv/bin/python play_multi.py --episodes 100 --opponent-cap 8
```

The completed 50,000-fight run recorded **5,918 wins, 43,976 losses, and 106 draws**.
At max opponent skill, the initial deterministic 100-fight checks returned 6W/94L
against three opponents and 0W/100L against eight. This does not meet the no-loss
goal. A later simulator fix added collateral explosive damage to other opponents;
a separate 10,000-fight diagnostic model improved the three-opponent check to 38W/160L/2D
in 200 fights, but still managed only 1W/199L against eight. That diagnostic is not
counted as part of the exact 50,000-fight run. See the training summary JSON for
histograms and the repeat 200-fight checks of the original model.

## Action compatibility

The policy still chooses an integer in `Discrete(30)`; Java maps that ID to the
matching control/item intent. Standard Minecraft client interactions are used—no
server commands or custom arena protocol:

| ID | Training action | Forge execution |
|---:|---|---|
| 0 | no-op | release movement |
| 1–4 | forward, back, left, right | movement relative to the target-facing camera |
| 5–7 | forward/left/right + attack | corresponding movement plus charged melee attack |
| 8 | jump | jump input |
| 9 | sprint + jump + forward + attack | same inputs and attack |
| 10 | sprint + forward | same movement |
| 11 | back + attack | same movement and attack |
| 12 | shield | hold an available shield and retreat |
| 13 | golden apple | use golden/enchanted golden apple |
| 14 | instant heal | use a hotbar splash potion with `HEAL` |
| 15 | ender pearl | throw an ender pearl |
| 16 | buff potion | drink a strength or speed potion |
| 17 | ranged | throw harming potion, snowball, or egg; charge/release a bow or crossbow when present |
| 18 | barricade | place a generic block between player and target |
| 19 | cobweb/lava trap | place a cobweb, otherwise pour a lava bucket near target |
| 20 | water bucket | pour water at the player's feet |
| 21 | pillar jump | jump and try to place a block adjacent to the player |
| 22 | end crystal | place obsidian/crystal and attack the crystal using vanilla combat |
| 23 | respawn anchor | place anchor; charge and activate it if glowstone is available |
| 24 | TNT minecart | place a rail and TNT minecart near target |
| 25 | obsidian cage | try to place four obsidian walls around player |
| 26 | W-tap combo | vanilla charged attack with a sprint reset |
| 27 | pearl + crystal | throw pearl, then attempt obsidian/crystal placement |
| 28 | combo break | shield, otherwise water/pearl if available, and retreat |
| 29 | combo counter | attack during the locally estimated counter/stun window |

Some arena actions depend on custom server commands or simulator-only damage rules.
In ordinary Minecraft, action 23 needs glowstone and detonates only where respawn
anchors explode; action 24 places a cart but vanilla ignition is still required; W-tap
has no simulator-only damage/knockback bonus; and the simulator's combo stun/counter
are estimated from observed health changes. They remain the model's chosen action
IDs, but their real-server effect cannot be identical to `ProArena` without a matching
server plugin. All action IDs are still mapped, and unavailable items/invalid
placements are safely ignored by the client.
