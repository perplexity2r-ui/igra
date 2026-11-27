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

The best-tested multi-opponent checkpoint is the 48-feature
`pvp_ppo_terrain_multi_aggressive_aoe_50k.zip`, trained for exactly 50,000 fights
after a separate 10,000-fight AoE diagnostic. At skill 1.0, its independent 200-fight
checks won 58/200 against three opponents, 30/200 against four, 22/200 against five,
12/200 against six, 4/200 against seven, and 1/200 against eight. This is not stable
winning. Six more exact 50,000-fight refinements (hard-crowd sampling, target focus,
win/death shaping, weighted-group continuation, full-difficulty training, and a
50-feature threat-aware policy) failed to produce stable wins. The threat-aware model
won 56/200, 31/200, 10/200, 2/200, 0/200, and 1/200 against groups of three through
eight. A 52-feature continuation adds a weighted crowd-bearing vector and is in
training; it has no evaluation result yet. The full iteration table is in
[`multi_opponent_iterations.md`](multi_opponent_iterations.md).

## Forge export and runtime

The embedded policy has been expanded to 52 inputs from the best-tested 48-feature
actor. Its four new threat/crowd-direction columns are zero in the warm-start seed, so
the expanded actor preserves the old logits exactly (512-observation comparison: zero
max delta, zero action mismatches). Export verification on 2,048 52-feature inputs
passed with max logit error `3.8146973e-06` and zero action mismatches. Resource
SHA-256:

`ca75414097fee6f8edd7f9b38aa82adf680bf095342592c082335d907a82f6c3`

The 52-feature training run is in progress; after it finishes, the final actor must be
re-exported and verified before treating it as a replacement. CI must rebuild both
ForgeGradle variants with this resource. Live Minecraft multi-opponent performance
remains unverified, and Python explosive splash still differs from Forge runtime behavior.
