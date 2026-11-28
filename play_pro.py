"""Evaluate / watch the trained PRO model in the crystal & TNT-minecart arena.

Works with legacy 34-feature policies and the terrain-aware 44-feature fine-tuned policy; actions remain `Discrete(30)` for the current PRO actor.

    .venv/bin/python play_pro.py --episodes 20 --stochastic
    .venv/bin/python play_pro.py --model models/pvp_ppo_combo_final.zip --skill 1.0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from env.pro_vec_env import OBS_DIM, TERRAIN_OBS_DIM, TECHNIQUES, ProPvPEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/terrain_finetune/pvp_ppo_terrain_aggressive_50k.zip")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--skill", type=float, default=1.0, help="professional opponent skill 0..1")
    p.add_argument("--terrain-mix", type=float, default=0.0,
                    help="share (0..1) of evaluation fights using randomized hard terrain")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.terrain_mix <= 1.0:
        raise SystemExit("--terrain-mix must be between 0 and 1")
    model = PPO.load(str(Path(args.model).with_suffix("")), device="cpu")
    obs_shape = tuple(model.observation_space.shape)
    if obs_shape not in ((OBS_DIM,), (TERRAIN_OBS_DIM,)):
        raise SystemExit(f"Unsupported model observation shape: {obs_shape}")
    if int(model.action_space.n) != 30:
        raise SystemExit(f"Expected Discrete(30), got {model.action_space}")
    terrain_obs = obs_shape == (TERRAIN_OBS_DIM,)
    if args.terrain_mix > 0.0 and not terrain_obs:
        print("Warning: legacy checkpoint has no terrain sensors; evaluating it blind on hard terrain")
    env = ProPvPEnv(seed=args.seed, terrain_mix=args.terrain_mix, terrain_obs=terrain_obs)
    env.core.set_enemy_skill(args.skill)
    print(f"Loaded {args.model} | obs={obs_shape} actions={model.action_space.n} "
          f"| opponent skill={args.skill} terrain_mix={args.terrain_mix}")

    wins = losses = draws = 0
    totals: dict[str, int] = {}
    terrain_kinds: dict[int, int] = {}
    blocked_steps = pit_steps = 0
    rewards, lengths = [], []
    for ep in range(1, args.episodes + 1):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        ep_rew, steps = 0.0, 0
        used: dict[str, int] = {}
        last = {}
        while not done:
            act, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, rew, term, trunc, last = env.step(act)
            ep_rew += rew
            steps += 1
            blocked_steps += int(bool(last.get("terrain_blocked", False)))
            pit_steps += int(bool(last.get("pit_blocked", False)))
            name = last.get("used_consumable")
            if name:
                used[name] = used.get(name, 0) + 1
                totals[name] = totals.get(name, 0) + 1
            done = term or trunc
        rewards.append(ep_rew)
        lengths.append(steps)
        won = last.get("enemy_dead") and not last.get("learner_dead")
        lost = last.get("learner_dead") and not last.get("enemy_dead")
        wins += won
        losses += lost
        draws += not (won or lost)
        terrain_kind = int(last.get("terrain_kind", 0))
        terrain_kinds[terrain_kind] = terrain_kinds.get(terrain_kind, 0) + 1
        tag = "WIN" if won else ("LOSS" if lost else "DRAW")
        if not args.quiet:
            print(f"Episode {ep:>3}/{args.episodes}: {tag:<5} | "
                  f"{last.get('learner_kit'):<34} vs {last.get('enemy_kit'):<34} | "
                  f"steps={steps:>3} rew={ep_rew:>8.1f} HP={last.get('learner_hp', 0):>4.1f}/"
                  f"{last.get('enemy_hp', 0):>4.1f} | {used}")
    n = max(1, args.episodes)
    print("-" * 110)
    print(f"WinRate={(wins / n) * 100:.1f}% ({wins}W/{losses}L/{draws}D) | "
          f"MeanReward={np.mean(rewards):.1f} | MeanSteps={np.mean(lengths):.1f}")
    print(f"Techniques used: {dict(sorted(totals.items(), key=lambda kv: -kv[1]))}")
    print(f"Terrain kinds: {dict(sorted(terrain_kinds.items()))} | blocked steps={blocked_steps} | pit stops={pit_steps}")


if __name__ == "__main__":
    main()
