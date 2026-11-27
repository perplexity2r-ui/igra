"""Evaluate / watch the trained PRO model in the crystal & TNT-minecart arena.

Works with both `models/pvp_ppo_pro_500k.zip` (Box(30,) / Discrete(28)) and the
legacy `models/pvp_ppo_latest.zip` (Box(24,) / Discrete(22)).

    .venv/bin/python play_pro.py --episodes 20 --stochastic
    .venv/bin/python play_pro.py --model models/pvp_ppo_latest.zip --skill 1.0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from env.pro_vec_env import TECHNIQUES, ProPvPEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/pvp_ppo_pro_500k.zip")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--skill", type=float, default=1.0, help="professional opponent skill 0..1")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env = ProPvPEnv(seed=args.seed)
    env.core.set_enemy_skill(args.skill)
    model = PPO.load(str(Path(args.model).with_suffix("")), device="cpu")
    print(f"Loaded {args.model} | obs={model.observation_space.shape} actions={model.action_space.n} "
          f"| opponent skill={args.skill}")

    wins = losses = draws = 0
    totals: dict[str, int] = {}
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


if __name__ == "__main__":
    main()
