"""Train or resume the separate Minecraft builder PPO for an episode target."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from env.builder_env import BuilderEnv, N_STAGES


class EpisodeBudgetCallback(BaseCallback):
    def __init__(
        self,
        target_episodes: int,
        starting_episodes: int,
        save_dir: Path,
        save_every: int,
        prior_metrics: dict[str, Any] | None = None,
        seed: int = 0,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.target_episodes = target_episodes
        self.starting_episodes = starting_episodes
        self.run_target = target_episodes - starting_episodes
        self.completed = 0
        self.save_dir = save_dir
        self.save_every = save_every
        self.prior_metrics = prior_metrics or {}
        self.seed = seed
        self.episode_rewards: list[float] = []
        self.episode_qualities: list[float] = []
        self.started = 0.0

    @property
    def total_completed(self) -> int:
        return self.starting_episodes + self.completed

    def _on_training_start(self) -> None:
        self.started = time.monotonic()
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue
            self.completed += 1
            reward = float(info["episode"]["r"])
            if "quality" not in info:
                raise RuntimeError("terminal BuilderEnv info did not include its quality score")
            quality = float(info["quality"])
            self.episode_rewards.append(reward)
            self.episode_qualities.append(quality)
            if self.verbose and (self.completed % 1_000 == 0 or self.completed == 1):
                recent_rewards = self.episode_rewards[-100:]
                recent_qualities = self.episode_qualities[-100:]
                print(
                    f"run_episodes={self.completed}/{self.run_target} "
                    f"total_episodes={self.total_completed}/{self.target_episodes} "
                    f"mean_reward_100={np.mean(recent_rewards):.3f} "
                    f"mean_quality_100={np.mean(recent_qualities):.3f}",
                    flush=True,
                )
            if self.save_every and self.total_completed % self.save_every == 0:
                checkpoint_dir = self.save_dir / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                self.model.save(checkpoint_dir / f"builder_{self.total_completed:06d}_episodes")
                self._write_metrics()
            if self.total_completed >= self.target_episodes:
                return False
        return True

    def _combined_mean(self, field: str, run_values: list[float]) -> float | None:
        previous_count = int(self.prior_metrics.get("completed_episodes", self.starting_episodes))
        previous_mean = self.prior_metrics.get(field)
        if previous_count != self.starting_episodes:
            previous_mean = None
        run_count = len(run_values)
        if previous_mean is None:
            if self.starting_episodes:
                return None
            return float(np.mean(run_values)) if run_values else None
        total_count = previous_count + run_count
        if total_count <= 0:
            return None
        return float((float(previous_mean) * previous_count + float(np.sum(run_values))) / total_count)

    def _write_metrics(self) -> None:
        elapsed = time.monotonic() - self.started if self.started else 0.0
        prior_elapsed = float(self.prior_metrics.get("cumulative_elapsed_seconds",
                                                     self.prior_metrics.get("elapsed_seconds", 0.0)) or 0.0)
        payload = {
            "target_episodes": self.target_episodes,
            "completed_episodes": self.total_completed,
            "starting_episodes_this_run": self.starting_episodes,
            "episodes_this_run": self.completed,
            "seed_this_run": self.seed,
            "mean_reward": self._combined_mean("mean_reward", self.episode_rewards),
            "mean_quality": self._combined_mean("mean_quality", self.episode_qualities),
            "mean_reward_this_run": float(np.mean(self.episode_rewards)) if self.episode_rewards else None,
            "mean_quality_this_run": float(np.mean(self.episode_qualities)) if self.episode_qualities else None,
            "recent_mean_reward_100": float(np.mean(self.episode_rewards[-100:])) if self.episode_rewards else None,
            "recent_mean_quality_100": float(np.mean(self.episode_qualities[-100:])) if self.episode_qualities else None,
            "elapsed_seconds_this_run": elapsed,
            "cumulative_elapsed_seconds": prior_elapsed + elapsed,
        }
        (self.save_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _on_training_end(self) -> None:
        self._write_metrics()
        self.model.save(self.save_dir / f"builder_{self.total_completed:06d}_episodes")


def _load_prior_metrics(resume: str | None, out_dir: Path, starting_episodes: int) -> dict[str, Any] | None:
    candidates = [out_dir / "metrics.json"]
    if resume:
        model_dir = Path(resume).parent
        candidates.extend((model_dir / "metrics.json", model_dir.parent / "metrics.json"))
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if int(payload.get("completed_episodes", -1)) == starting_episodes:
            return payload
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50_000,
                        help="total target episodes, including any resumed checkpoint")
    parser.add_argument("--resume", help="continue from a builder PPO checkpoint")
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=50_000,
                        help="save an interim checkpoint every N total episodes; 0 disables")
    parser.add_argument("--out-dir", type=Path, default=Path("models/builder"))
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    if args.batch_size < 2 or args.n_steps < 2:
        raise SystemExit("--batch-size and --n-steps must be at least 2")
    if args.checkpoint_every < 0:
        raise SystemExit("--checkpoint-every must be nonnegative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def make_env():
        return Monitor(BuilderEnv())

    env = DummyVecEnv([make_env])
    if args.resume:
        model = PPO.load(args.resume, env=env, device=args.device)
        if model.num_timesteps % N_STAGES:
            env.close()
            raise SystemExit(
                f"checkpoint has {model.num_timesteps} timesteps, not an exact {N_STAGES}-step episode boundary"
            )
        starting_episodes = model.num_timesteps // N_STAGES
        if args.episodes <= starting_episodes:
            env.close()
            raise SystemExit(
                f"target {args.episodes} episodes is not above the checkpoint's {starting_episodes} episodes"
            )
        prior_metrics = _load_prior_metrics(args.resume, args.out_dir, starting_episodes)
        if prior_metrics:
            snapshot = args.out_dir / f"metrics_{starting_episodes:06d}.json"
            if not snapshot.exists():
                snapshot.write_text(json.dumps(prior_metrics, indent=2) + "\n", encoding="utf-8")
        model.seed = args.seed
        env.seed(args.seed + starting_episodes)
    else:
        starting_episodes = 0
        prior_metrics = None
        model = PPO(
            "MlpPolicy", env, seed=args.seed, device=args.device, verbose=0,
            n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=4,
            learning_rate=3e-4, gamma=0.98, gae_lambda=0.95, clip_range=0.2,
            policy_kwargs={"net_arch": [128, 128]},
        )

    remaining_episodes = args.episodes - starting_episodes
    callback = EpisodeBudgetCallback(
        args.episodes,
        starting_episodes,
        args.out_dir,
        args.checkpoint_every,
        prior_metrics=prior_metrics,
        seed=args.seed,
    )
    print(
        f"Training builder PPO: target={args.episodes} total episodes, "
        f"starting={starting_episodes}, this_run={remaining_episodes}, "
        f"seed={args.seed}, device={args.device}",
        flush=True,
    )
    # Extra rollout steps let the callback stop exactly on the target episode boundary.
    model.learn(
        total_timesteps=remaining_episodes * N_STAGES + args.n_steps,
        callback=callback,
        progress_bar=False,
        reset_num_timesteps=not bool(args.resume),
    )
    env.close()
    callback._write_metrics()
    print(
        f"Training complete: {callback.total_completed} total episodes; "
        f"model saved as {args.out_dir / f'builder_{callback.total_completed:06d}_episodes.zip'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
