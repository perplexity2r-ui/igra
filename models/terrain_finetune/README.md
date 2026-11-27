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

## Forge export

The actor was exported to `forge-mod/src/main/resources/com/arena/aipilot/ppo_policy.bin`.
`forge-mod/tools/verify_ppo_weights.py --samples 2048` passed: maximum logit error `3.8146973e-06`, zero action mismatches. Resource SHA-256:

`91b2c436f3a3ac14a3fbbcefd4fffaa61c4d718ead7fbcda9adfa3bbc7f9cc49`

The Forge Java source now expects 44 observations and includes obstacle-aware pursuit plus the K-key style panel. **The Forge JAR was not rebuilt in this environment**: Java, `javac`, Gradle, and a Gradle wrapper are absent. The existing JAR therefore remains the old build until it is rebuilt with the 1.16.5 Forge toolchain.
