# PvP hard-terrain fine-tune

## Training

- Starting actor: `models/pvp_ppo_combo_final.zip` (`Box(34,)`, `Discrete(30)`).
- New actor: `pvp_ppo_terrain_10k.zip` (`Box(44,)`, `Discrete(30)`); the original 34 observation entries remain in order and ten terrain sensors are appended.
- Completed fights: **10,000 exactly** (`pro_training_summary.json`).
- Terrain mix: **100%** walls, floor pits, hills, or mixed layouts; 203,924 movement-blocked steps and 10,466 pit stops recorded.
- Training totals: 7,503 wins, 2,450 losses, 47 draws (75.03% over the full run); the last 200 training fights were 69% wins at the fully ramped opponent setting.

The opponent skill and combo-start curriculum ramped during training, so the full-run win rate is not a fixed-skill benchmark.

## Fixed-seed hard-terrain evaluation

100 fights, opponent skill 1.0, terrain mix 1.0, seed 5050:

| Policy | Wins | Losses | Draws | Win rate | Mean reward | Mean steps |
|---|---:|---:|---:|---:|---:|---:|
| Original 34-feature combo actor (no terrain inputs) | 6 | 88 | 6 | 6% | -139.7 | 123.0 |
| 10k-fight terrain actor | 12 | 83 | 5 | 12% | -112.4 | 107.5 |

The same seeded terrain distribution was used in both runs (21 wall, 26 pit, 29 hill, 24 mixed maps). This is a simulator comparison, not a guarantee of performance on a Minecraft server; the hard fixed-skill benchmark remains challenging.

## One-versus-many continuation

The current multi-opponent actor is a 48-feature extension of the terrain policy. Its
best-tested checkpoint remains
`pvp_ppo_terrain_multi_aggressive_aoe_50k.zip`, trained for exactly 50,000 fights after
a separate 10,000-fight AoE diagnostic. At skill 1.0, the independent 200-fight
checks won 58/200 against three opponents, 30/200 against four, 22/200 against five,
12/200 against six, 4/200 against seven, and 1/200 against eight. This is not stable
winning. Hard-crowd sampling, target-focus, and win/death reward-shaping refinements
each ran another exact 50,000 fights but did not improve the benchmark. The full
iteration table is in [`multi_opponent_iterations.md`](multi_opponent_iterations.md).

## Forge export and runtime

The current 48-feature actor was exported to
`forge-mod/src/main/resources/com/arena/aipilot/ppo_policy.bin`.
`forge-mod/tools/verify_ppo_weights.py --samples 2048` passed: maximum logit error
`3.8146973e-06`, zero action mismatches. Resource SHA-256:

`476f41ea115e1c9e73d524ea57b2af86d8d8a6b01c7c3396fefe71f4849c9ee5`

GitHub Actions built both ForgeGradle 5 and 6 variants successfully for commit
`4ad7204`; the published `mods/pvp/aipilot-1.16.5.jar` contains the same policy bytes.
Live Minecraft group-combat performance has not been verified; the Python arena's
explosive splash behavior still differs from Forge runtime behavior.
