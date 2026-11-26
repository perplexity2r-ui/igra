# PPO actor integration (Minecraft Forge 1.16.5)

## Model and runtime

The Forge client embeds only the deterministic actor exported from
`models/pvp_ppo_combo_final.zip`. Its architecture is
`34 -> Linear(192) -> Tanh -> Linear(192) -> Tanh -> Linear(30)`.
`PpoPolicy` reads `src/main/resources/com/arena/aipilot/ppo_policy.bin` and selects
`argmax(logits)` in Java. Python, PyTorch, Stable-Baselines3, and network access are
not needed in the game. The client runs one inference every two Minecraft ticks
(10 decisions per second, matching the arena's `DT = 0.10`).

To regenerate and verify the embedded weights after updating the checkpoint:

```bash
.venv/bin/python forge-mod/tools/export_ppo_weights.py
.venv/bin/python forge-mod/tools/verify_ppo_weights.py
```

The verifier compares all 30 logits against Stable-Baselines3 on 512 seeded
observations. The checked-in resource has SHA-256
`2a0e01dfe0568efed0fa2668ce7986d872ecf75e3afc9a91ef1c9454e5be14d1`.
If the resource is missing or has an incompatible header/size, the mod fails closed:
AI combat is disabled; it does not fall back to the old heuristic.

## Observation compatibility

The source contract is `env/pro_vec_env.py::_obs()` (`Box(-1, 2, (34,), float32)`).
Indices and normalization are kept in the same order. Minecraft has no direct
counterpart for some simulator-only state, so the bridge uses the following
measurements/estimates:

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
| 18 | `(buffs + ranged) / 12` | speed/strength potions plus arrows, snowballs, eggs, and crossbows |
| 19 | learner shield up | local player actively blocking with a shield |
| 20 | enemy shield up | target actively blocking with a shield |
| 21 | generic blocks `/ 24` | block items, excluding obsidian, cobweb, and respawn anchor |
| 22 | cobweb + lava + water `/ 6` | corresponding inventory items |
| 23 | obstacle count `/ 6` | `1/6` when the target is not visible, otherwise zero |
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

Inventory counts include the full player inventory, while actions can only select
hotbar items; the target's private inventory may not be synchronized by every server.
Armor tiers, simulator stun/combos, obstacle count, and the custom cage flag are
therefore approximations rather than exact simulator state. This does not change the
network's tensor shape or feature order.

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
| 17 | ranged | throw harming potion, snowball, or egg (when present) |
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
