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

The verified 48-feature AoE checkpoint is the reference baseline; it was trained for
exactly 50,000 fights after a separate 10,000-fight AoE diagnostic. At skill 1.0, its
independent 200-fight checks won 58/200 against three opponents, 30/200 against four,
22/200 against five, 12/200 against six, 4/200 against seven, and 1/200 against eight.
The 52-feature crowd-bearing continuation won 63, 34, 23, 9, 2, and 1 fights against
groups of three through eight, respectively. It had five more wins overall across the
six groups, but regressed against groups of six and seven and tied at eight. Neither
model wins consistently across all group sizes. Seven more exact 50,000-fight
refinements (hard-crowd sampling, target focus, win/death shaping, weighted-group
continuation, full-difficulty training, 50-feature threat positions, and 52-feature
crowd-bearing) have failed to produce stable wins. The full iteration table is in
[`multi_opponent_iterations.md`](multi_opponent_iterations.md).

## Forge export and runtime

The trained policy has 52 inputs. Its four new threat/crowd-direction columns were
zero-initialized in the warm-start seed, preserving the original 48-feature logits
exactly at initialization (512-observation comparison: zero max delta, zero action
mismatches). The final trained actor was exported and verified on 2,048 52-feature
observations with max logit error `3.8146973e-06` and zero action mismatches. Resource
SHA-256:

`f1391f903942bbc3f3dd6dba2a10f292626f25893eb4c4b72e714f95584e12be`

Forge CI must rebuild both ForgeGradle variants with these final weights. Live
Minecraft multi-opponent performance remains unverified, and Python explosive splash
still differs from Forge runtime behavior.
