"""Train a PPO agent for Minecraft 1.16.5 PvP (Asymmetric Randomizer & Blocks Mode).

Supports two backends with identical observation (`Box(24,)`) and action (`Discrete(22)`) spaces:
  - `--mode live` (default): connects over WebSocket to `bot/bot.js` + Paper 1.16.5 server.
  - `--mode sim`: trains in the fast local 1.16.5 PvP physics simulator (`env/sim_env.py`),
    allowing rapid training before fine-tuning or playing on a live Minecraft server.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from env import MinecraftPvPEnv, SimulatedPvPEnv


class PvPMetricsCallback(BaseCallback):
    """Logs PvP win/loss/timeout/block/trap/consumable stats and saves periodic episode backups."""

    def __init__(
        self,
        log_freq_episodes: int = 20,
        backup_every_episodes: int = 50,
        backup_dir: Path | None = None,
        max_backups: int = 25,
        target_episodes: int | None = None,
        status_file: Path | None = None,
        initial_episodes: int = 0,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.log_freq_episodes = log_freq_episodes
        self.backup_every_episodes = backup_every_episodes
        self.backup_dir = backup_dir
        self.max_backups = max_backups
        self.target_episodes = target_episodes
        self.status_file = status_file
        self.episodes = initial_episodes
        self.wins = 0
        self.losses = 0
        self.timeouts = 0
        self.shields_broken = 0
        self.blocked_hits = 0
        self.blocks_placed = 0
        self.enemies_trapped = 0
        self.totems_popped = 0
        self.consumable_counts: dict[str, int] = {}
        self.weapon_wins: dict[str, int] = {}
        self.weapon_games: dict[str, int] = {}
        self.recent_outcomes: list[int] = []  # 1=win, 0=draw/loss
        self.saved_backups: list[Path] = []
        self.start_time = time.time()

    def _save_episode_backup(self, win_rate: float) -> None:
        if self.backup_dir is None or self.backup_every_episodes <= 0:
            return
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        rolling_path = self.backup_dir / "pvp_ppo_backup_latest"
        self.model.save(str(rolling_path))

        ep_backup_base = self.backup_dir / f"pvp_ppo_ep_{self.episodes:06d}"
        self.model.save(str(ep_backup_base))
        ep_zip = ep_backup_base.with_suffix(".zip")
        self.saved_backups.append(ep_zip)

        if self.max_backups > 0:
            existing = sorted(
                self.backup_dir.glob("pvp_ppo_ep_*.zip"),
                key=lambda p: p.stat().st_mtime,
            )
            while len(existing) > self.max_backups:
                oldest = existing.pop(0)
                try:
                    oldest.unlink()
                except OSError:
                    pass

        if self.status_file is not None:
            weapon_win_rates = {
                w: round(self.weapon_wins.get(w, 0) / max(1, g), 4)
                for w, g in sorted(self.weapon_games.items())
            }
            status = {
                "episodes": self.episodes,
                "target_episodes": self.target_episodes,
                "timesteps": int(self.num_timesteps),
                "wins": self.wins,
                "losses": self.losses,
                "timeouts": self.timeouts,
                "recent_win_rate_100": round(win_rate, 4),
                "shields_broken": self.shields_broken,
                "blocked_hits": self.blocked_hits,
                "blocks_placed": self.blocks_placed,
                "enemies_trapped": self.enemies_trapped,
                "totems_popped": self.totems_popped,
                "consumables_used": dict(sorted(self.consumable_counts.items())),
                "weapon_win_rates": weapon_win_rates,
                "latest_backup": str(ep_zip),
                "elapsed_sec": round(time.time() - self.start_time, 2),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            self.status_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = self.status_file.with_suffix(".tmp")
            tmp_file.write_text(json.dumps(status, indent=2), encoding="utf-8")
            tmp_file.replace(self.status_file)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for done, info in zip(dones, infos):
            if info.get("shield_broken"):
                self.shields_broken += 1
            if info.get("blocked_hit"):
                self.blocked_hits += 1
            if info.get("placed_block"):
                self.blocks_placed += 1
            if info.get("trapped_enemy"):
                self.enemies_trapped += 1
            if info.get("totem_popped_learner") or info.get("totem_popped_enemy"):
                self.totems_popped += 1
            used_c = info.get("used_consumable")
            if used_c:
                self.consumable_counts[used_c] = (
                    self.consumable_counts.get(used_c, 0) + 1
                )

            if not done:
                continue

            self.episodes += 1
            won = bool(info.get("enemy_dead") and not info.get("learner_dead"))
            lost = bool(info.get("learner_dead") and not info.get("enemy_dead"))
            raw_kit = str(info.get("learner_kit", "random"))
            weapon_key = str(info.get("learner_weapon") or raw_kit.split("+")[0])
            self.weapon_games[weapon_key] = self.weapon_games.get(weapon_key, 0) + 1

            if won:
                self.wins += 1
                self.weapon_wins[weapon_key] = self.weapon_wins.get(weapon_key, 0) + 1
                self.recent_outcomes.append(1)
            elif lost:
                self.losses += 1
                self.recent_outcomes.append(0)
            else:
                self.timeouts += 1
                self.recent_outcomes.append(0)

            if len(self.recent_outcomes) > 100:
                self.recent_outcomes.pop(0)

            win_rate = sum(self.recent_outcomes) / len(self.recent_outcomes)
            if self.episodes % 10 == 0:
                self.logger.record("pvp/win_rate_100", win_rate)
                self.logger.record("pvp/episodes", self.episodes)

            if (
                self.backup_every_episodes > 0
                and self.episodes % self.backup_every_episodes == 0
            ):
                self._save_episode_backup(win_rate)

            if self.verbose and self.episodes % self.log_freq_episodes == 0:
                print(
                    f"[Step {self.num_timesteps:>7}] Runs: {self.episodes:>6} | "
                    f"Wins: {self.wins} | Losses: {self.losses} | Timeouts: {self.timeouts} | "
                    f"WinRate(100): {win_rate * 100:.1f}% | "
                    f"Blocks: {self.blocks_placed} | Traps: {self.enemies_trapped} | "
                    f"ShieldBreaks: {self.shields_broken}",
                    flush=True,
                )

            if self.target_episodes is not None and self.episodes >= self.target_episodes:
                self._save_episode_backup(win_rate)
                if self.verbose:
                    print(
                        f"Reached target of {self.target_episodes} episodes (runs) at step {self.num_timesteps}.",
                        flush=True,
                    )
                return False
        return True


def make_env(mode: str, ws_url: str, n_envs: int = 1):
    if mode == "sim":
        if n_envs > 1:
            return DummyVecEnv([lambda: Monitor(SimulatedPvPEnv()) for _ in range(n_envs)])
        return Monitor(SimulatedPvPEnv())
    return Monitor(MinecraftPvPEnv(host=ws_url))


def build_new_ppo(
    env,
    lr: float,
    n_steps: int,
    batch_size: int,
    n_epochs: int,
    verbose: int,
    seed: int,
    device: str,
) -> PPO:
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.015,
        policy_kwargs={"net_arch": [128, 128]},
        verbose=verbose,
        seed=seed,
        device=device,
    )


def warm_start_from_checkpoint(old_model: PPO, new_model: PPO) -> int:
    """Transfers overlapping MLP weights when observation/action spaces are expanded."""
    old_sd = old_model.policy.state_dict()
    new_sd = new_model.policy.state_dict()
    copied = 0
    with torch.no_grad():
        for key, new_param in new_sd.items():
            if key not in old_sd:
                continue
            old_param = old_sd[key]
            if old_param.shape == new_param.shape:
                new_param.copy_(old_param)
                copied += 1
            elif old_param.ndim == new_param.ndim == 2:
                r = min(old_param.shape[0], new_param.shape[0])
                c = min(old_param.shape[1], new_param.shape[1])
                new_param[:r, :c].copy_(old_param[:r, :c])
                copied += 1
            elif old_param.ndim == new_param.ndim == 1:
                r = min(old_param.shape[0], new_param.shape[0])
                new_param[:r].copy_(old_param[:r])
                copied += 1
    new_model.policy.load_state_dict(new_sd)
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO agent for Minecraft 1.16.5 PvP")
    parser.add_argument(
        "--mode",
        choices=["live", "sim"],
        default="live",
        help="'live' connects to bot/bot.js WebSocket; 'sim' runs fast offline PvP arena simulation",
    )
    parser.add_argument(
        "--ws-url",
        default="ws://127.0.0.1:8765",
        help="WebSocket URL of bot/bot.js when --mode=live",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
        help="Total training timesteps (ignored if --episodes stops earlier, or acts as upper bound)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Optional target number of PvP episodes (runs) to train for",
    )
    parser.add_argument(
        "--stage-timesteps",
        type=int,
        default=0,
        help="If > 0, train gradually in stages of N timesteps, saving the main model after each stage",
    )
    parser.add_argument(
        "--backup-every-episodes",
        type=int,
        default=50,
        help="Save a model backup every N episodes/runs (default: 50)",
    )
    parser.add_argument(
        "--max-backups",
        type=int,
        default=25,
        help="Maximum numbered episode backups to keep on disk (0 = keep all, default: 25)",
    )
    parser.add_argument(
        "--backup-dir",
        type=str,
        default="models/backups",
        help="Directory for periodic episode backups",
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Optional path to an existing PPO .zip checkpoint to continue or warm-start from",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/pvp_ppo_latest",
        help="Output path for the trained PPO model (.zip added automatically by SB3)",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=25_000,
        help="Save periodic step checkpoints every N steps",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=12,
        help="Number of vectorized SimulatedPvPEnv arenas when --mode=sim (default: 12)",
    )
    parser.add_argument(
        "--initial-episodes",
        type=int,
        default=0,
        help="Resume episode counter from an earlier partial run",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch device ('cpu', 'cuda', 'auto')")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_path = Path(args.save_path)
    if save_path.suffix == ".zip":
        save_path = save_path.with_suffix("")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = save_path.parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = Path(args.backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    status_file = save_path.parent / "training_status.json"

    total_timesteps = args.timesteps
    if args.episodes is not None and total_timesteps < args.episodes * 600:
        total_timesteps = args.episodes * 600

    n_envs = max(1, args.n_envs) if args.mode == "sim" else 1
    env = make_env(args.mode, args.ws_url, n_envs=n_envs)
    try:
        if args.mode == "sim" and n_envs > 1:
            n_steps = 256
            batch_size = 256
            n_epochs = 5
        elif args.mode == "sim":
            n_steps = 1024
            batch_size = 64
            n_epochs = 10
        else:
            n_steps = 256
            batch_size = 64
            n_epochs = 10
        sb3_verbose = 0 if (args.episodes and args.episodes >= 2000) else 1

        if args.load:
            print(f"Loading existing model from {args.load}...", flush=True)
            try:
                old_loaded = PPO.load(args.load, device=args.device)
                if (
                    old_loaded.observation_space.shape != env.observation_space.shape
                    or old_loaded.action_space != env.action_space
                ):
                    raise ValueError(
                        f"Space mismatch: {old_loaded.observation_space}->{old_loaded.action_space} "
                        f"vs {env.observation_space}->{env.action_space}"
                    )
                model = build_new_ppo(
                    env, args.lr, n_steps, batch_size, n_epochs, sb3_verbose, args.seed, args.device
                )
                model.policy.load_state_dict(old_loaded.policy.state_dict())
                model.num_timesteps = old_loaded.num_timesteps
            except ValueError as space_err:
                print(
                    f"Space expanded ({space_err}) -> Warm-starting expanded policy from {args.load}!",
                    flush=True,
                )
                old_model = PPO.load(args.load, device=args.device)
                model = build_new_ppo(
                    env, args.lr, n_steps, batch_size, n_epochs, sb3_verbose, args.seed, args.device
                )
                copied = warm_start_from_checkpoint(old_model, model)
                obs_dim = env.observation_space.shape[0]
                act_dim = env.action_space.n
                print(
                    f"Transferred {copied} tensor layers from previous PvP model into {obs_dim}-obs / {act_dim}-action policy.",
                    flush=True,
                )
        else:
            model = build_new_ppo(
                env, args.lr, n_steps, batch_size, n_epochs, sb3_verbose, args.seed, args.device
            )

        log_freq = (
            1000
            if (args.episodes and args.episodes >= 5000)
            else (250 if (args.episodes and args.episodes >= 2000) else (50 if args.mode == "sim" else 10))
        )
        metrics_cb = PvPMetricsCallback(
            log_freq_episodes=log_freq,
            backup_every_episodes=args.backup_every_episodes,
            backup_dir=backup_dir,
            max_backups=args.max_backups,
            target_episodes=args.episodes,
            status_file=status_file,
            initial_episodes=args.initial_episodes,
            verbose=1,
        )
        ckpt_cb = CheckpointCallback(
            save_freq=max(args.checkpoint_freq // n_envs, 1000),
            save_path=str(ckpt_dir),
            name_prefix="pvp_ppo",
        )
        callbacks = [metrics_cb, ckpt_cb]

        target_desc = (
            f"{args.episodes} episodes (runs)"
            if args.episodes is not None
            else f"{total_timesteps} timesteps"
        )
        print(
            f"Starting PPO training in '{args.mode}' mode for {target_desc} "
            f"(backups every {args.backup_every_episodes} runs -> {backup_dir})...",
            flush=True,
        )

        try:
            if args.stage_timesteps and args.stage_timesteps > 0:
                remaining = total_timesteps
                stage = 0
                while remaining > 0:
                    stage += 1
                    chunk = min(args.stage_timesteps, remaining)
                    print(
                        f"\n--- Stage {stage}: training next {chunk} timesteps "
                        f"(completed runs so far: {metrics_cb.episodes}) ---",
                        flush=True,
                    )
                    model.learn(
                        total_timesteps=chunk,
                        callback=callbacks,
                        reset_num_timesteps=False,
                    )
                    remaining -= chunk
                    model.save(str(save_path))
                    print(
                        f"--- Stage {stage} saved to {save_path}.zip "
                        f"(runs: {metrics_cb.episodes}, steps: {model.num_timesteps}) ---",
                        flush=True,
                    )
                    if args.episodes is not None and metrics_cb.episodes >= args.episodes:
                        break
            else:
                model.learn(total_timesteps=total_timesteps, callback=callbacks)
        except KeyboardInterrupt:
            print("\nTraining interrupted by user — saving current model...", flush=True)

        model.save(str(save_path))
        win_rate = (
            sum(metrics_cb.recent_outcomes) / len(metrics_cb.recent_outcomes)
            if metrics_cb.recent_outcomes
            else 0.0
        )
        metrics_cb._save_episode_backup(win_rate)
        print(
            f"Finished! Saved trained model to {save_path}.zip "
            f"(Total runs: {metrics_cb.episodes}, Total steps: {model.num_timesteps}, "
            f"Wins: {metrics_cb.wins}, Losses: {metrics_cb.losses}, Timeouts: {metrics_cb.timeouts})",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
