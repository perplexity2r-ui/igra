"""Fine-tune the aggressive terrain PPO actor for one-versus-many PvP fights."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from env.multi_opponent import MAX_OPPONENTS, MIN_OPPONENTS, MultiOpponentVecEnv
from train_pro import warm_start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fights", type=int, default=50_000,
                        help="number of completed one-versus-many fights")
    parser.add_argument("--exact-fights", action="store_true",
                        help="freeze each vector row at its allocated fight quota")
    parser.add_argument("--n-envs", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--net", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20261128)
    parser.add_argument("--terrain-mix", type=float, default=1.0)
    parser.add_argument("--player-aggression", type=float, default=1.0)
    parser.add_argument("--min-opponents", type=int, default=MIN_OPPONENTS)
    parser.add_argument("--max-opponents", type=int, default=MAX_OPPONENTS)
    parser.add_argument("--load", default="models/terrain_finetune/pvp_ppo_terrain_aggressive_50k.zip")
    parser.add_argument("--save", default="models/terrain_finetune/pvp_ppo_terrain_multi_aggressive_50k")
    parser.add_argument("--log-every-fights", type=int, default=5_000)
    parser.add_argument("--time-budget", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fights <= 0 or args.n_envs <= 0:
        raise SystemExit("--fights and --n-envs must be positive")
    if not 0.0 <= args.terrain_mix <= 1.0:
        raise SystemExit("--terrain-mix must be between 0 and 1")
    if not 0.0 <= args.player_aggression <= 1.0:
        raise SystemExit("--player-aggression must be between 0 and 1")
    if not MIN_OPPONENTS <= args.min_opponents <= args.max_opponents <= MAX_OPPONENTS:
        raise SystemExit(f"opponent range must satisfy {MIN_OPPONENTS} <= min <= max <= {MAX_OPPONENTS}")

    save_path = Path(args.save).with_suffix("")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    status_path = save_path.with_name(save_path.name + "_status.json")
    summary_path = save_path.with_name(save_path.name + "_summary.json")
    checkpoint_dir = save_path.with_name(save_path.name + "_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = MultiOpponentVecEnv(
        args.n_envs,
        seed=args.seed,
        terrain_mix=args.terrain_mix,
        min_opponents=args.min_opponents,
        max_opponents=args.min_opponents,
    )
    env.set_player_aggression(args.player_aggression)
    if args.exact_fights:
        env.set_fight_limit(args.fights)
    env.set_enemy_skill(0.30)
    env.set_aggression(0.55)
    env.set_combo_curriculum(0.0)
    env.set_opponent_cap(args.min_opponents)
    env.reset()

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"net_arch": [args.net, args.net]},
        verbose=0,
        seed=args.seed,
        device="cpu",
    )
    if args.load:
        warm_start(model, args.load)

    steps_per_rollout = args.n_steps * args.n_envs
    total_timesteps = 0
    last_log = 0
    dumps = 0
    start = time.time()
    print(
        f"MULTI PvP training: target={args.fights:,} fights | {args.n_envs} arenas | "
        f"opponents={args.min_opponents}..{args.max_opponents} | "
        f"player aggression={args.player_aggression:.0%} | terrain={args.terrain_mix:.0%} | "
        f"exact={args.exact_fights} | warm start={args.load}",
        flush=True,
    )

    while env.episodes < args.fights:
        elapsed = time.time() - start
        if args.time_budget > 0.0 and elapsed >= args.time_budget:
            print(f"Time budget reached at {env.episodes:,} fights", flush=True)
            break
        progress = min(1.0, env.episodes / max(1.0, args.fights * 0.85))
        skill = 0.30 + 0.70 * progress
        opponent_progress = min(1.0, env.episodes / max(1.0, args.fights * 0.65))
        current_cap = args.min_opponents + int(
            math.floor((args.max_opponents - args.min_opponents) * opponent_progress + 0.5)
        )
        enemy_aggression = 0.55 + 0.45 * progress
        combo_curriculum = 0.20 * progress
        env.set_enemy_skill(skill)
        env.set_aggression(enemy_aggression)
        env.set_combo_curriculum(combo_curriculum)
        env.set_opponent_cap(current_cap)

        model.learn(total_timesteps=steps_per_rollout, reset_num_timesteps=False, log_interval=None)
        total_timesteps += steps_per_rollout

        if env.episodes - last_log >= args.log_every_fights or env.episodes >= args.fights:
            last_log = env.episodes
            elapsed = time.time() - start
            recent_n = max(1, len(env.last_results))
            recent_wr = sum(result == "W" for result in env.last_results) / recent_n
            completed = max(1, env.episodes)
            avg_opponents = env.opponents_total / completed
            active_counts = ",".join(
                f"{count}:{int(env.opponents_histogram[count])}"
                for count in range(args.min_opponents, args.max_opponents + 1)
            )
            print(
                f"[{env.episodes:>7,}/{args.fights:,}] {elapsed:6.0f}s "
                f"({env.episodes / max(elapsed, 1e-6):.1f} fights/s) | "
                f"skill={skill:.2f} cap={current_cap} avg-foes={avg_opponents:.2f} | "
                f"W={env.wins:,} L={env.losses:,} D={env.draws:,} "
                f"recent200={recent_wr:.1%} kills={env.enemy_kills:,} | "
                f"foe-counts[{active_counts}] | "
                f"W-tap={env.tech_counts.get('wtap_combo', 0):,} "
                f"AoE-hits={env.aoe_hits:,} AoE-kills={env.aoe_kills:,} "
                f"combo-breaks={env.combo_breaks:,} counters={env.combo_counters:,}",
                flush=True,
            )
            status = {
                "target_fights": args.fights,
                "fights": env.episodes,
                "wins": env.wins,
                "losses": env.losses,
                "draws": env.draws,
                "win_rate_recent_200": round(recent_wr, 4),
                "opponents_min": args.min_opponents,
                "opponents_max_curriculum_cap": current_cap,
                "opponents_average": round(avg_opponents, 3),
                "opponents_histogram": env.opponents_histogram.tolist(),
                "enemy_kills": env.enemy_kills,
                "aoe_enemy_hits": env.aoe_hits,
                "aoe_enemy_kills": env.aoe_kills,
                "aoe_enemy_damage": round(env.aoe_damage, 2),
                "enemy_skill": round(skill, 3),
                "enemy_aggression": round(enemy_aggression, 3),
                "combo_curriculum": round(combo_curriculum, 3),
                "terrain_mix": args.terrain_mix,
                "player_aggression": args.player_aggression,
                "timesteps": total_timesteps,
                "elapsed_sec": round(elapsed, 1),
                "techniques": dict(sorted(env.tech_counts.items())),
                "combos": {
                    "breaks": env.combo_breaks,
                    "counters": env.combo_counters,
                    "hits": env.combo_hits,
                },
                "terrain_blocked_steps": env.terrain_blocked,
                "pit_blocked_steps": env.pit_blocked,
                "other_enemy_terrain_blocked_steps": env.extra_terrain_blocked,
                "other_enemy_pit_blocked_steps": env.extra_pit_blocked,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            temp = status_path.with_suffix(".tmp")
            temp.write_text(json.dumps(status, indent=2), encoding="utf-8")
            temp.replace(status_path)
            dumps += 1
            if dumps == 1:
                model.save(str(save_path.with_name(save_path.name + "_mid")))
            if dumps % 3 == 0:
                checkpoint = checkpoint_dir / f"multi_{env.episodes:07d}"
                model.save(str(checkpoint))
                old_checkpoints = sorted(checkpoint_dir.glob("multi_*.zip"))
                for old in old_checkpoints[:-4]:
                    old.unlink(missing_ok=True)

    model.save(str(save_path))
    elapsed = time.time() - start
    recent_n = max(1, len(env.last_results))
    recent_wr = sum(result == "W" for result in env.last_results) / recent_n
    summary = {
        "target_fights": args.fights,
        "fights": env.episodes,
        "wins": env.wins,
        "losses": env.losses,
        "draws": env.draws,
        "win_rate_recent_200": round(recent_wr, 4),
        "opponents_min": args.min_opponents,
        "opponents_max": args.max_opponents,
        "opponents_average": round(env.opponents_total / max(1, env.episodes), 3),
        "opponents_histogram": env.opponents_histogram.tolist(),
        "enemy_kills": env.enemy_kills,
        "aoe_enemy_hits": env.aoe_hits,
        "aoe_enemy_kills": env.aoe_kills,
        "aoe_enemy_damage": round(env.aoe_damage, 2),
        "terrain_mix": args.terrain_mix,
        "player_aggression": args.player_aggression,
        "timesteps": total_timesteps,
        "elapsed_sec": round(elapsed, 1),
        "combos": {
            "breaks": env.combo_breaks,
            "counters": env.combo_counters,
            "hits": env.combo_hits,
        },
        "techniques": dict(sorted(env.tech_counts.items())),
        "terrain_blocked_steps": env.terrain_blocked,
        "pit_blocked_steps": env.pit_blocked,
        "other_enemy_terrain_blocked_steps": env.extra_terrain_blocked,
        "other_enemy_pit_blocked_steps": env.extra_pit_blocked,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("FINISHED", json.dumps(summary), flush=True)
    print(f"Saved multi-opponent model to {save_path}.zip", flush=True)


if __name__ == "__main__":
    main()
