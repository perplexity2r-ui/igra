"""Behavior-clone the ranged decision from valid bow/crossbow simulator states.

PPO exploration produced ranged shots, but its deterministic actor did not choose
action 17. This focused post-training pass teaches the action on valid bow/crossbow
states and behavior-clones the input actor elsewhere. The 52-feature observation and
30-action interface are unchanged.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO

from env.multi_opponent import MultiOpponentVecEnv
from env.pro_vec_env import RANGED_TYPES

BOW_INDEX = next(i for i, spec in enumerate(RANGED_TYPES) if spec.ranged_id == "bow")
CROSSBOW_INDEX = next(i for i, spec in enumerate(RANGED_TYPES) if spec.ranged_id == "crossbow")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="PPO checkpoint to fine-tune")
    parser.add_argument("--save", required=True, help="output PPO checkpoint (without or with .zip)")
    parser.add_argument("--seed", type=int, default=20261130)
    parser.add_argument("--n-envs", type=int, default=128)
    parser.add_argument("--sample-steps", type=int, default=1500)
    parser.add_argument("--terrain-mix", type=float, default=0.5)
    parser.add_argument("--max-examples-per-type", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--teacher-weight", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    return parser.parse_args()


def _sample_indices(rng: np.random.Generator, size: int, count: int) -> np.ndarray:
    if size < count:
        raise ValueError(f"Need {count} examples, only collected {size}")
    return rng.choice(size, size=count, replace=False)


def collect_contexts(model: PPO, *, seed: int, n_envs: int, sample_steps: int,
                     terrain_mix: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect valid bow, valid crossbow, and non-ranged teacher states."""
    env = MultiOpponentVecEnv(
        n_envs, seed=seed, terrain_mix=terrain_mix, min_opponents=3, max_opponents=8
    )
    env.set_enemy_skill(0.75)
    env.set_aggression(0.85)
    env.set_player_aggression(1.0)
    obs = env.reset()

    bow_chunks: list[np.ndarray] = []
    crossbow_chunks: list[np.ndarray] = []
    other_chunks: list[np.ndarray] = []
    arena = env.core
    sim = arena.core
    learner = sim.L
    target = sim.E
    rows = np.arange(n_envs)

    for _ in range(sample_steps):
        distance = np.linalg.norm(target.pos - learner.pos, axis=1)
        active_target = arena.alive[rows, arena.active_slot]
        equipped = (learner.has_ranged_weapon
                    & ((learner.ranged > 0.0) | learner.ranged_loaded)
                    & (learner.ranged_charge <= 0.0)
                    & (learner.use_lock <= 0.0))
        target_clear = ~sim._obstacle_between(learner.pos, target.pos)
        target_unshielded = ~(target.has_shield & target.shield_up
                              & (target.shield_disabled <= 0.0))
        usable_context = (active_target & equipped & target_clear & target_unshielded
                          & (distance >= 4.0) & (distance <= 16.0)
                          & (learner.hp > 9.0))
        bow = usable_context & (learner.ranged_type == BOW_INDEX)
        crossbow = usable_context & (learner.ranged_type == CROSSBOW_INDEX)

        if bow.any():
            bow_chunks.append(obs[bow].copy())
        if crossbow.any():
            crossbow_chunks.append(obs[crossbow].copy())
        other = ~(bow | crossbow)
        if other.any():
            other_chunks.append(obs[other].copy())

        actions, _ = model.predict(obs, deterministic=False)
        obs, _, _, _ = env.step(actions)

    env.close()
    if not bow_chunks or not crossbow_chunks or not other_chunks:
        raise RuntimeError("Could not collect both ranged types and teacher states")
    return (np.concatenate(bow_chunks), np.concatenate(crossbow_chunks),
            np.concatenate(other_chunks))


def deterministic_action17_rate(policy, observations: np.ndarray, batch_size: int = 4096) -> float:
    total = selected = 0
    for start in range(0, len(observations), batch_size):
        batch = observations[start:start + batch_size]
        actions, _ = policy.predict(batch, deterministic=True)
        selected += int(np.count_nonzero(actions == 17))
        total += len(actions)
    return selected / max(1, total)


def main() -> None:
    args = parse_args()
    if args.n_envs <= 0 or args.sample_steps <= 0 or args.max_examples_per_type <= 0:
        raise SystemExit("n-envs, sample-steps, and max-examples-per-type must be positive")
    if args.epochs <= 0 or args.batch_size < 4 or args.batch_size % 2:
        raise SystemExit("epochs must be positive and batch-size must be an even number >= 4")
    if not 0.0 <= args.terrain_mix <= 1.0:
        raise SystemExit("terrain-mix must be between 0 and 1")

    torch.manual_seed(args.seed)
    print(f"Collecting valid ranged contexts from {args.model}", flush=True)
    model = PPO.load(args.model, device="cpu")
    teacher = PPO.load(args.model, device="cpu")
    bow_obs, crossbow_obs, other_obs = collect_contexts(
        model, seed=args.seed, n_envs=args.n_envs,
        sample_steps=args.sample_steps, terrain_mix=args.terrain_mix,
    )
    rng = np.random.default_rng(args.seed)
    per_type = min(args.max_examples_per_type, len(bow_obs), len(crossbow_obs))
    # Keep both positive weapon contexts and a larger set of teacher states so the
    # action-17 preference remains conditional instead of becoming a global bias.
    negative_count = 2 * per_type
    if len(other_obs) < negative_count:
        per_type = len(other_obs) // 2
        negative_count = 2 * per_type
    if per_type < 256:
        raise RuntimeError(
            f"Too few training examples (bow={len(bow_obs)}, crossbow={len(crossbow_obs)}, "
            f"other={len(other_obs)})"
        )

    bow_obs = bow_obs[_sample_indices(rng, len(bow_obs), per_type)]
    crossbow_obs = crossbow_obs[_sample_indices(rng, len(crossbow_obs), per_type)]
    other_obs = other_obs[_sample_indices(rng, len(other_obs), negative_count)]
    positive_np = np.concatenate([bow_obs, crossbow_obs]).astype(np.float32)
    negative_np = other_obs.astype(np.float32)
    print(
        f"Dataset: bow={len(bow_obs):,}, crossbow={len(crossbow_obs):,}, "
        f"teacher states={len(negative_np):,}; obs={positive_np.shape[1]}",
        flush=True,
    )

    device = torch.device("cpu")
    positive = torch.as_tensor(positive_np, dtype=torch.float32, device=device)
    negative = torch.as_tensor(negative_np, dtype=torch.float32, device=device)
    teacher_actions_np, _ = teacher.predict(negative_np, deterministic=True)
    teacher_actions = torch.as_tensor(
        teacher_actions_np, dtype=torch.long, device=device
    ).reshape(-1)

    positive_batch = max(1, args.batch_size // 4)
    negative_batch = args.batch_size - positive_batch
    batches_per_epoch = math.ceil(max(len(positive), len(negative)) / negative_batch)
    # Fine-tune the shared representation with a fresh, low-rate optimizer. The
    # teacher action loss on the larger negative set anchors all unrelated states.
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=args.learning_rate)
    model.policy.set_training_mode(True)
    for epoch in range(args.epochs):
        positive_loss_sum = 0.0
        teacher_ce_sum = 0.0
        for _ in range(batches_per_epoch):
            pos_idx = torch.randint(len(positive), (positive_batch,), device=device)
            neg_idx = torch.randint(len(negative), (negative_batch,), device=device)
            batch_obs = torch.cat([positive[pos_idx], negative[neg_idx]], dim=0)
            distribution = model.policy.get_distribution(batch_obs).distribution
            log_probs = distribution.logits
            target_actions = torch.full((positive_batch,), 17, dtype=torch.long, device=device)
            imitation_loss = F.nll_loss(log_probs[:positive_batch], target_actions)
            preserve_loss = F.nll_loss(
                log_probs[positive_batch:], teacher_actions[neg_idx]
            )
            loss = imitation_loss + args.teacher_weight * preserve_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            optimizer.step()
            positive_loss_sum += float(imitation_loss.detach())
            teacher_ce_sum += float(preserve_loss.detach())
        print(
            f"epoch {epoch + 1}/{args.epochs}: "
            f"ranged CE={positive_loss_sum / batches_per_epoch:.4f} "
            f"teacher CE={teacher_ce_sum / batches_per_epoch:.4f}",
            flush=True,
        )

    for parameter in model.policy.parameters():
        parameter.requires_grad_(True)
    model.policy.set_training_mode(False)
    bow_rate = deterministic_action17_rate(model.policy, bow_obs)
    crossbow_rate = deterministic_action17_rate(model.policy, crossbow_obs)
    other_rate = deterministic_action17_rate(model.policy, negative_np)
    save_path = Path(args.save).with_suffix("")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(
        f"Saved {save_path}.zip; deterministic action-17 rates: "
        f"bow={bow_rate:.1%}, crossbow={crossbow_rate:.1%}, "
        f"other/invalid={other_rate:.1%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
