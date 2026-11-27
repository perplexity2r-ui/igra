"""Evaluate a trained builder against a uniform random design baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from env.builder_env import BuilderEnv, DESIGN_CHOICES, THEMES, design_quality


def evaluate_policy(model: PPO, episodes_per_theme: int, seed: int) -> dict:
    env = BuilderEnv()
    results = {}
    all_qualities: list[float] = []
    for theme_id, theme_name in enumerate(THEMES):
        qualities = []
        first_choices: list[int] | None = None
        for episode in range(episodes_per_theme):
            obs, _ = env.reset(seed=seed + episode, options={"theme": theme_id})
            choices = []
            for _ in DESIGN_CHOICES:
                action, _ = model.predict(obs, deterministic=True)
                choice = int(np.asarray(action).item())
                choices.append(choice)
                obs, _, terminated, truncated, info = env.step(choice)
                if terminated or truncated:
                    if "quality" not in info:
                        raise RuntimeError("builder environment did not report terminal quality")
                    qualities.append(float(info["quality"]))
                    break
            if first_choices is None:
                first_choices = choices
            if len(choices) != len(DESIGN_CHOICES):
                raise RuntimeError(f"policy produced an incomplete plan for theme {theme_name}")
        results[theme_name] = {
            "episodes": len(qualities),
            "mean_quality": float(np.mean(qualities)),
            "min_quality": float(np.min(qualities)),
            "max_quality": float(np.max(qualities)),
            "first_plan_choices": first_choices,
        }
        all_qualities.extend(qualities)
    env.close()
    return {
        "episodes": len(all_qualities),
        "mean_quality": float(np.mean(all_qualities)),
        "per_theme": results,
    }


def evaluate_random(episodes: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    qualities = []
    for _ in range(episodes):
        theme = int(rng.integers(len(THEMES)))
        choices = [int(rng.integers(count)) for count in DESIGN_CHOICES]
        qualities.append(design_quality(choices, theme))
    return {"episodes": episodes, "mean_quality": float(np.mean(qualities))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="SB3 checkpoint path, with or without .zip")
    parser.add_argument("--episodes-per-theme", type=int, default=100)
    parser.add_argument("--random-episodes", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--out", type=Path, default=Path("models/builder/evaluation.json"))
    args = parser.parse_args()
    if args.episodes_per_theme <= 0 or args.random_episodes <= 0:
        raise SystemExit("evaluation episode counts must be positive")

    model = PPO.load(args.model, device="cpu")
    policy_result = evaluate_policy(model, args.episodes_per_theme, args.seed)
    random_result = evaluate_random(args.random_episodes, args.seed)
    payload = {
        "model": str(args.model),
        "seed": args.seed,
        "reward_note": "Scores are the handcrafted design-quality proxy, not human or screenshot ratings.",
        "policy": policy_result,
        "uniform_random_baseline": random_result,
        "proxy_gain": policy_result["mean_quality"] - random_result["mean_quality"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
