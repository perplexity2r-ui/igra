# Multi-opponent PPO iteration log

All evaluation rows below use deterministic policy inference, enemy skill `1.0`, terrain mix `1.0`, seed `6061`, and 200 fights per fixed group size unless noted. These are Python-arena results; live Minecraft group combat has not been verified.

| Training run | Training W/L/D | Mean foes | Independent evaluation wins per 200 for groups 3, 4, 5, 6, 7, 8 |
|---|---:|---:|---|
| Initial multi-opponent 50k (before collateral AoE fix) | 5,918 / 43,976 / 106 | 4.674 | Initial 100-fight check: 6 wins vs 3; 0 wins vs 8 |
| AoE diagnostic 10k (not part of a 50k total) | 1,485 / 8,491 / 24 | 4.591 | 38; —; 14; —; —; 1 |
| Corrected AoE 50k, warm-started from the 10k diagnostic | 8,395 / 41,447 / 158 | 4.667 | **58 (29%); 30 (15%); 22 (11%); 12 (6%); 4 (2%); 1 (0.5%)** |
| Hard-crowd 50k, weighted toward larger groups | 5,041 / 44,848 / 111 | 5.899 | 48 (24%); 28 (14%); 13 (6.5%); 6 (3%); 0; 1 (0.5%) |
| Target-focus 50k | 5,118 / 44,785 / 97 | 5.890 | 30 (15%); 9 (4.5%); 9 (4.5%); 6 (3%); 1 (0.5%); 1 (0.5%) |
| Win/death-reward 50k | 5,153 / 44,750 / 97 | 5.894 | 52 (26%); 29 (14.5%); 13 (6.5%); 5 (2.5%); 3 (1.5%); 1 (0.5%) |
| Weighted-group continuation 50k (seed 20261204) | 5,139 / 44,750 / 111 | 5.907 | 59 (29.5%); 38 (19%); 16 (8%); 4 (2%); 4 (2%); 0 |
| Max-difficulty 50k (seed 20261205; lr 1e-4) | 4,468 / 45,447 / 85 | 5.910 | 54 (27%); 26 (13%); 10 (5%); 6 (3%); 5 (2.5%); 0 |
| Threat-position 50k (seed 20261206; six crowd/threat features) | 5,247 / 44,649 / 104 | 5.896 | 56 (28%); 31 (15.5%); 10 (5%); 2 (1%); 0; 1 (0.5%) |

The 48-feature AoE checkpoint remains the baseline and is stronger against groups of six and seven. The 52-feature crowd-bearing model recorded 132 wins across the six 200-fight groups versus 127 for AoE: it improved groups 3–5, regressed against 6–7, and tied at 8. **Neither model wins consistently across all group sizes.** The training win rate is not a substitute for the fixed independent benchmark.

Every refinement used an exact completed-fight quota of 50,000. The `*_summary.json` files record training histograms and conditions; the threat-position and crowd-bearing summaries also include their independent evaluation counts.

## Completed iteration: crowd-bearing 52-feature

Seed `20261207`, warm-started from the 48-feature AoE checkpoint through the zero-initialized 52-feature seed, completed exactly 50,000 fights in 1,797.3 seconds. It used the standard difficulty curriculum, 128 arenas, and player aggression 1.0. Training W/L/D was **5,320 / 44,588 / 92** (recent-200 win rate 12.0%). The independent results are in `pvp_ppo_terrain_multi_aggressive_pressure_50k_summary.json`; the trained actor was exported and verified against 2,048 observations.

## Next iteration

An exact 50,000-fight 52-feature full-difficulty run (seed `20261208`) is now in progress, warm-started from `pvp_ppo_terrain_multi_aggressive_pressure_50k.zip` at learning rate `1e-4`. Enemy skill/aggression and combo curriculum are at maximum from the start. The output checkpoint is `pvp_ppo_terrain_multi_aggressive_pressure_maxdiff_50k.zip`. The fixed seed-6061 evaluation protocol remains unchanged; no run counts as success without reliable independent wins against every group size from three through eight.
