"""Evaluate or watch a trained PPO agent (or random baseline) in Minecraft PvP."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO

from env import MinecraftPvPEnv, SimulatedPvPEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained Minecraft PvP agent")
    parser.add_argument(
        "--mode",
        choices=["live", "sim"],
        default="live",
        help="'live' connects to bot/bot.js WebSocket; 'sim' runs offline PvP arena simulation",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/pvp_ppo_latest.zip",
        help="Path to trained PPO .zip model (or 'random' to test random baseline)",
    )
    parser.add_argument(
        "--ws-url",
        default="ws://127.0.0.1:8765",
        help="WebSocket URL of bot/bot.js when --mode=live",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of PvP episodes to run",
    )
    parser.add_argument(
        "--kit",
        type=int,
        default=None,
        help="Optional forced weapon pool index (0..11) for LearnerBot; fully random asymmetric loadout if omitted",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions stochastically instead of deterministic greedy policy",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_random = args.model.lower() == "random"
    model = None
    if not use_random:
        model_path = Path(args.model)
        if not model_path.exists() and not model_path.with_suffix(".zip").exists():
            raise FileNotFoundError(
                f"Model file not found at '{args.model}'. Train one first with 'python train.py' "
                f"or pass '--model random'."
            )
        model = PPO.load(str(model_path), device="cpu")
        print(f"Loaded PPO model from {model_path}")
    else:
        print("Evaluating random baseline policy")

    env = SimulatedPvPEnv() if args.mode == "sim" else MinecraftPvPEnv(host=args.ws_url)
    wins = 0
    losses = 0
    timeouts = 0
    rewards: list[float] = []
    lengths: list[int] = []
    total_consumables: dict[str, int] = {}

    try:
        for ep in range(1, args.episodes + 1):
            reset_opts = {"kit": args.kit} if args.kit is not None else None
            obs, _ = env.reset(seed=args.seed + ep, options=reset_opts)
            done = False
            ep_reward = 0.0
            steps = 0
            last_info: dict = {}
            ep_consumables: dict[str, int] = {}

            while not done:
                if model is not None:
                    action, _ = model.predict(obs, deterministic=not args.stochastic)
                else:
                    action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                steps += 1
                used_c = info.get("used_consumable")
                if used_c:
                    ep_consumables[used_c] = ep_consumables.get(used_c, 0) + 1
                    total_consumables[used_c] = total_consumables.get(used_c, 0) + 1
                last_info = info
                done = terminated or truncated

            won = bool(last_info.get("enemy_dead") and not last_info.get("learner_dead"))
            lost = bool(last_info.get("learner_dead") and not last_info.get("enemy_dead"))
            if won:
                wins += 1
                outcome = "WIN"
            elif lost:
                losses += 1
                outcome = "LOSS"
            else:
                timeouts += 1
                outcome = "DRAW/TIMEOUT"

            rewards.append(ep_reward)
            lengths.append(steps)
            l_kit = last_info.get("learner_kit", "-")
            e_kit = last_info.get("enemy_kit", "-")
            print(
                f"Episode {ep:>3}/{args.episodes}: {outcome:<12} | "
                f"Loadout: {l_kit:<32} vs {e_kit:<32} | "
                f"Steps: {steps:>3} | Reward: {ep_reward:>7.2f} | "
                f"HP: {last_info.get('learner_hp', 0):>4.1f}/{last_info.get('enemy_hp', 0):>4.1f} | "
                f"Items: {ep_consumables}"
            )
    finally:
        env.close()

    win_rate = (wins / max(1, args.episodes)) * 100.0
    print("-" * 96)
    print(
        f"Summary ({args.episodes} episodes, mode={args.mode}): "
        f"WinRate={win_rate:.1f}% ({wins}W / {losses}L / {timeouts}T) | "
        f"MeanReward={float(np.mean(rewards)):.2f} | "
        f"MeanSteps={float(np.mean(lengths)):.1f} | "
        f"ConsumablesUsed={total_consumables}"
    )


if __name__ == "__main__":
    main()
