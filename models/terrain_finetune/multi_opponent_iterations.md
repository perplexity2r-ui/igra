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

The best tested checkpoint remains `pvp_ppo_terrain_multi_aggressive_aoe_50k.zip`. It does **not** win consistently: its win rates fall from 29% against three opponents to 0.5% against eight. The other 50k refinements did not improve the larger-group results. None of these runs supports a claim that the bot does not lose.

The corrected AoE 50k run itself completed exactly 50,000 fights; it was initialized from the separate 10k diagnostic model. Subsequent 50k iterations also each used exact completed-fight quotas. The summary JSON records the best checkpoint's full training histogram and evaluation counts.
