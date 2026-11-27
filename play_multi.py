"""Evaluate the 48-feature PPO actor in one-versus-many PvP fights."""
from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3 import PPO

from env.multi_opponent import MAX_OPPONENTS, MIN_OPPONENTS, MULTI_OBS_DIM, MultiOpponentVecEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/terrain_finetune/pvp_ppo_terrain_multi_aggressive_aoe_50k.zip")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--min-opponents", type=int, default=MIN_OPPONENTS)
    parser.add_argument("--opponent-cap", type=int, default=MAX_OPPONENTS)
    parser.add_argument("--skill", type=float, default=1.0)
    parser.add_argument("--terrain-mix", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=5050)
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    if not MIN_OPPONENTS <= args.min_opponents <= args.opponent_cap <= MAX_OPPONENTS:
        raise SystemExit(
            f"opponent range must satisfy {MIN_OPPONENTS} <= --min-opponents <= "
            f"--opponent-cap <= {MAX_OPPONENTS}"
        )
    if not 0.0 <= args.terrain_mix <= 1.0 or not 0.0 <= args.skill <= 1.0:
        raise SystemExit("--terrain-mix and --skill must be between 0 and 1")

    model = PPO.load(args.model, device="cpu")
    if tuple(model.observation_space.shape) != (MULTI_OBS_DIM,):
        raise SystemExit(f"Expected Box({MULTI_OBS_DIM},), got {model.observation_space}")
    if int(model.action_space.n) != 30:
        raise SystemExit(f"Expected Discrete(30), got {model.action_space}")

    env_count = min(args.n_envs, args.episodes)
    env = MultiOpponentVecEnv(
        env_count, seed=args.seed, terrain_mix=args.terrain_mix,
        min_opponents=args.min_opponents, max_opponents=args.opponent_cap,
    )
    env.set_opponent_cap(args.opponent_cap)
    env.set_enemy_skill(args.skill)
    env.set_aggression(1.0)
    env.set_player_aggression(1.0)
    env.set_fight_limit(args.episodes)
    obs = env.reset()

    rewards = np.zeros(env_count, dtype=np.float64)
    steps = np.zeros(env_count, dtype=np.int32)
    episode_rewards: list[float] = []
    episode_steps: list[int] = []
    while env.episodes < args.episodes:
        actions, _ = model.predict(obs, deterministic=not args.stochastic)
        obs, batch_rewards, dones, _ = env.step(actions)
        rewards += batch_rewards
        steps += 1
        finished = np.flatnonzero(dones)
        for index in finished:
            episode_rewards.append(float(rewards[index]))
            episode_steps.append(int(steps[index]))
            rewards[index] = 0.0
            steps[index] = 0

    total = max(1, env.episodes)
    print(f"Loaded {args.model} | obs={MULTI_OBS_DIM} actions=30 | skill={args.skill:.2f} "
          f"terrain={args.terrain_mix:.0%} | opponents={args.min_opponents}..{args.opponent_cap}")
    print(f"Results: {env.wins}W/{env.losses}L/{env.draws}D "
          f"({env.wins / total:.1%} wins) | recent200={sum(x == 'W' for x in env.last_results) / max(1, len(env.last_results)):.1%}")
    print(f"Mean opponents={env.opponents_total / total:.2f} | histogram="
          f"{dict((i, int(env.opponents_histogram[i])) for i in range(args.min_opponents, args.opponent_cap + 1))}")
    print(f"Mean reward={np.mean(episode_rewards) if episode_rewards else 0.0:.1f} | "
          f"Mean steps={np.mean(episode_steps) if episode_steps else 0.0:.1f} | "
          f"enemy kills={env.enemy_kills}")
    print(f"Area damage: hits={env.aoe_hits}, kills={env.aoe_kills}, damage={env.aoe_damage:.1f}")
    print(f"Combos: hits={env.combo_hits}, breaks={env.combo_breaks}, counters={env.combo_counters}")
    print(f"Techniques: {dict(sorted(env.tech_counts.items(), key=lambda item: -item[1]))}")


if __name__ == "__main__":
    main()
