"""Train the PRO PvP agent on 500 000 live-style fights (crystals, anchors, TNT carts, W-tap).

Runs a fully-vectorised arena (`env/pro_vec_env.py`) with 1024 parallel fights, logs a
full technique breakdown, ramps the professional opponent from "strong human" to
"tournament pro", saves checkpoints and finally writes `models/pro_500k_*`.

Usage:
    .venv/bin/python train_pro.py --fights 500000 --n-envs 1024
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from env.pro_vec_env import TECHNIQUES, ProPvPVecEnv

TECH_KEYS = [t for t in TECHNIQUES if t != "none"]


class ProMetrics:
    """Lightweight cumulative fight bookkeeping (nothing per-env, O(1) per step)."""

    def __init__(self):
        self.fights = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.tech = {k: 0 for k in TECH_KEYS}
        self.kills_by_crystal = 0
        self.recent: list[int] = []

    def absorb(self, env: ProPvPVecEnv) -> None:
        self.fights = env.episodes
        self.wins = env.wins
        self.losses = env.losses
        self.draws = env.draws
        for k, v in env.tech_counts.items():
            self.tech[k] = v
        self.recent = env.last_results

    @property
    def win_rate(self) -> float:
        n = max(1, len(self.recent))
        return sum(1 for r in self.recent if r == "W") / n


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--fights", type=int, default=500_000)
    p.add_argument("--n-envs", type=int, default=1024)
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--n-epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.02)
    p.add_argument("--net", type=int, default=192)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--save", default="models/pvp_ppo_pro_500k")
    p.add_argument("--load", default=None)
    p.add_argument("--fresh", action="store_true", help="ignore warm-start from the 50k model")
    p.add_argument("--time-budget", type=float, default=0.0, help="optional wall-clock cap (s)")
    p.add_argument("--log-every-fights", type=int, default=2000)
    return p.parse_args()


def warm_start(model: PPO, path: str) -> None:
    """Copy overlapping weights from the older 24-obs / 22-action policy into 30/28."""
    old = PPO.load(path, device="cpu")
    old_sd = old.policy.state_dict()
    new_sd = model.policy.state_dict()
    copied = 0
    with torch.no_grad():
        for key, new_param in new_sd.items():
            if key not in old_sd:
                continue
            op = old_sd[key]
            if op.shape == new_param.shape:
                new_param.copy_(op)
                copied += 1
            elif op.ndim == new_param.ndim == 2:
                r = min(op.shape[0], new_param.shape[0])
                c = min(op.shape[1], new_param.shape[1])
                new_param[:r, :c].copy_(op[:r, :c])
                copied += 1
            elif op.ndim == new_param.ndim == 1:
                r = min(op.shape[0], new_param.shape[0])
                new_param[:r].copy_(op[:r])
                copied += 1
    model.policy.load_state_dict(new_sd)
    print(f"Warm-started {copied} tensor layers from {path}", flush=True)


def main() -> None:
    args = parse_args()
    save_path = Path(args.save).with_suffix("")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    status_file = save_path.parent / "training_pro_status.json"
    ckpt_dir = save_path.parent / "pro_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env = ProPvPVecEnv(args.n_envs, seed=args.seed)
    env.set_enemy_skill(0.30)
    env.reset()

    model = PPO(
        "MlpPolicy", env,
        n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.n_epochs,
        learning_rate=args.lr, gamma=0.995, gae_lambda=0.95, clip_range=0.2,
        ent_coef=args.ent_coef, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs={"net_arch": [args.net, args.net]},
        verbose=0, seed=args.seed, device="cpu",
    )
    if args.load:
        warm_start(model, args.load)
    elif not args.fresh:
        base = Path("models/pvp_ppo_latest.zip")
        if base.exists():
            warm_start(model, str(base))

    metrics = ProMetrics()
    start = time.time()
    steps_per_rollout = args.n_steps * args.n_envs
    # remember how many fights each rollout produced, to log per-fight cadence
    last_log = 0
    total_timesteps = 0
    fight_target = args.fights
    dumps = 0

    print(
        f"PRO training: target {fight_target:,} fights | {args.n_envs} parallel arenas | "
        f"{steps_per_rollout} steps/rollout | checkpoint -> {ckpt_dir}",
        flush=True,
    )

    while metrics.fights < fight_target:
        elapsed = time.time() - start
        if args.time_budget and elapsed > args.time_budget:
            print(f"Time budget {args.time_budget}s reached at {metrics.fights:,} fights.", flush=True)
            break

        # ramp the professional opponent: strong human -> tournament pro
        progress = min(1.0, metrics.fights / max(1.0, fight_target * 0.85))
        skill = 0.30 + 0.70 * progress
        env.set_enemy_skill(skill)

        model.learn(total_timesteps=steps_per_rollout, reset_num_timesteps=False, log_interval=None)
        total_timesteps += steps_per_rollout
        metrics.absorb(env)

        if metrics.fights - last_log >= args.log_every_fights or metrics.fights >= fight_target:
            last_log = metrics.fights
            el = time.time() - start
            active = {k: v for k, v in metrics.tech.items() if v > 0}
            print(
                f"[{metrics.fights:>7,}/{fight_target:,} fights] {el:6.0f}s "
                f"({metrics.fights / max(el, 1e-6):6.0f} fights/s) | skill={skill:.2f} | "
                f"W={metrics.wins:,} L={metrics.losses:,} T={metrics.draws:,} | "
                f"win-rate(200)={metrics.win_rate * 100:5.1f}% | "
                f"crystals={metrics.tech.get('end_crystal', 0):,} "
                f"anchors={metrics.tech.get('respawn_anchor', 0):,} "
                f"carts={metrics.tech.get('tnt_minecart', 0):,} "
                f"wtap={metrics.tech.get('wtap_combo', 0):,}",
                flush=True,
            )
            status = {
                "target_fights": fight_target,
                "fights": metrics.fights,
                "wins": metrics.wins, "losses": metrics.losses, "draws": metrics.draws,
                "win_rate_recent_200": round(metrics.win_rate, 4),
                "enemy_skill": round(skill, 3),
                "timesteps": total_timesteps,
                "elapsed_sec": round(el, 1),
                "fights_per_sec": round(metrics.fights / max(el, 1e-6), 1),
                "techniques": dict(sorted(metrics.tech.items())),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            tmp = status_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(status, indent=2), encoding="utf-8")
            tmp.replace(status_file)
            dumps += 1
            if dumps % 3 == 0:
                model.save(str(ckpt_dir / f"pro_{metrics.fights:07d}"))
                # keep the checkpoint folder tidy
                olds = sorted(ckpt_dir.glob("pro_*.zip"))
                for old in olds[:-4]:
                    try:
                        old.unlink()
                    except OSError:
                        pass

    model.save(str(save_path))
    metrics.absorb(env)
    summary = {
        "fights": metrics.fights,
        "wins": metrics.wins, "losses": metrics.losses, "draws": metrics.draws,
        "win_rate_recent_200": round(metrics.win_rate, 4),
        "timesteps": total_timesteps,
        "elapsed_sec": round(time.time() - start, 1),
        "techniques": dict(sorted(metrics.tech.items())),
    }
    (save_path.parent / "pro_training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print("FINISHED", json.dumps(summary), flush=True)
    print(f"Saved model to {save_path}.zip", flush=True)


if __name__ == "__main__":
    main()
