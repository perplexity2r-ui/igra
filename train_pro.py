"""Train or terrain-fine-tune the PRO PvP agent in the vectorized combat arena.

Can train the legacy 34-feature policy or a terrain-aware 44-feature policy with
walls, floor holes, and hills. It logs techniques and ramps the opponent from a
strong human to a tournament pro.

Usage:
    .venv/bin/python train_pro.py --fights 500000 --n-envs 1024
    .venv/bin/python train_pro.py --fights 10000 --exact-fights --n-envs 128 \\
        --terrain-mix 1.0 --load models/pvp_ppo_combo_final.zip
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
        self.combo = {"breaks": 0, "counters": 0, "hits": 0,
                      "chains": 0, "chain_len": 0}

    def absorb(self, env: ProPvPVecEnv) -> None:
        self.fights = env.episodes
        self.wins = env.wins
        self.losses = env.losses
        self.draws = env.draws
        for k, v in env.tech_counts.items():
            self.tech[k] = v
        self.recent = env.last_results
        self.combo = {
            "breaks": env.combo_breaks, "counters": env.combo_counters,
            "hits": env.combo_hits, "chains": env.chains_suffered,
            "chain_len": env.chain_len_sum,
        }

    @property
    def win_rate(self) -> float:
        n = max(1, len(self.recent))
        return sum(1 for r in self.recent if r == "W") / n


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--fights", type=int, default=500_000,
                   help="number of completed fights for this training run")
    p.add_argument("--exact-fights", action="store_true",
                   help="allocate exact per-arena fight quotas; idle arenas freeze at their quota")
    p.add_argument("--n-envs", type=int, default=1024)
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--n-epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.02)
    p.add_argument("--net", type=int, default=192)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--terrain-mix", type=float, default=0.0,
                    help="share (0..1) of fights using walls, pits, hills, or mixed terrain")
    p.add_argument("--player-aggression", type=float, default=0.0,
                    help="0..1 shaping that rewards closing to melee and tolerates damage trades")
    p.add_argument("--save", default="models/pvp_ppo_pro_500k")
    p.add_argument("--load", default=None)
    p.add_argument("--fresh", action="store_true", help="train from random initialization")
    p.add_argument("--time-budget", type=float, default=0.0, help="optional wall-clock cap (s)")
    p.add_argument("--log-every-fights", type=int, default=2000)
    return p.parse_args()


def warm_start(model: PPO, path: str) -> None:
    """Warm-start compatible overlapping layers (including legacy-to-terrain inputs)."""
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
    model.policy.load_state_dict(new_sd, strict=False)
    print(f"Warm-started {copied} tensor layers from {path}", flush=True)
    # Only neutralise rows 28/29 when the head actually GREW (a brand-new action space).
    # When fine-tuning a model that already knows them, keep what was learned.
    import torch as _t
    grew = old_sd.get("action_net.bias") is None or \
        old_sd["action_net.bias"].shape[0] < model.policy.state_dict()["action_net.bias"].shape[0]
    if grew:
        with _t.no_grad():
            for key in ("action_net.weight", "action_net.bias"):
                p_ = model.policy.state_dict().get(key)
                if p_ is not None and p_.shape[0] > 28:
                    p_[28:] *= 0.02
        print("Action head rows 28/29 (combo break / counter) initialised neutral", flush=True)
    else:
        print("Combo actions kept from the warm-start model", flush=True)


def main() -> None:
    args = parse_args()
    save_path = Path(args.save).with_suffix("")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    status_file = save_path.parent / "training_pro_status.json"
    ckpt_dir = save_path.parent / "pro_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.fights <= 0:
        raise SystemExit("--fights must be positive")
    if not 0.0 <= args.terrain_mix <= 1.0:
        raise SystemExit("--terrain-mix must be between 0 and 1")
    if not 0.0 <= args.player_aggression <= 1.0:
        raise SystemExit("--player-aggression must be between 0 and 1")
    terrain_obs = args.terrain_mix > 0.0
    env = ProPvPVecEnv(args.n_envs, seed=args.seed,
                       terrain_mix=args.terrain_mix, terrain_obs=terrain_obs)
    env.set_player_aggression(args.player_aggression)
    if args.exact_fights:
        env.set_fight_limit(args.fights)
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
        f"terrain share={args.terrain_mix:.0%} | player aggression={args.player_aggression:.0%} | "
        f"exact={args.exact_fights} | {steps_per_rollout} steps/rollout | checkpoint -> {ckpt_dir}",
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
        # combo curriculum: the opponent starts casual and becomes a chain machine
        env.set_aggression(min(1.0, 0.55 + 0.45 * progress))
        # combo curriculum: up to 45% of fights begin with the learner already reeling
        env.set_combo_curriculum(0.45 * min(1.0, progress * 1.4))

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
                f"wtap={metrics.tech.get('wtap_combo', 0):,} | "
                f"combo: hits={metrics.combo['hits']:,} "
                f"breaks={metrics.combo['breaks']:,} "
                f"counters={metrics.combo['counters']:,} "
                f"chains={metrics.combo['chains']:,}",
                flush=True,
            )
            status = {
                "target_fights": fight_target,
                "fights": metrics.fights,
                "wins": metrics.wins, "losses": metrics.losses, "draws": metrics.draws,
                "win_rate_recent_200": round(metrics.win_rate, 4),
                "enemy_skill": round(skill, 3),
                "terrain_mix": args.terrain_mix,
                "player_aggression": args.player_aggression,
                "terrain_blocked_steps": env.terrain_blocked,
                "pit_blocked_steps": env.pit_blocked,
                "timesteps": total_timesteps,
                "elapsed_sec": round(el, 1),
                "fights_per_sec": round(metrics.fights / max(el, 1e-6), 1),
                "combos": dict(metrics.combo),
                "techniques": dict(sorted(metrics.tech.items())),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            tmp = status_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(status, indent=2), encoding="utf-8")
            tmp.replace(status_file)
            dumps += 1
            if dumps == 1:
                model.save(str(save_path.parent / f"{save_path.name}_mid"))
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
        "terrain_mix": args.terrain_mix,
        "player_aggression": args.player_aggression,
        "terrain_blocked_steps": env.terrain_blocked,
        "pit_blocked_steps": env.pit_blocked,
        "timesteps": total_timesteps,
        "elapsed_sec": round(time.time() - start, 1),
        "combos": dict(metrics.combo),
        "techniques": dict(sorted(metrics.tech.items())),
    }
    (save_path.parent / "pro_training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print("FINISHED", json.dumps(summary), flush=True)
    print(f"Saved model to {save_path}.zip", flush=True)


if __name__ == "__main__":
    main()
