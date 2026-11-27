# Separate Minecraft Builder PPO

This is an independent architecture-design agent. It does not load, alter, or share weights with the PvP model. Seven PPO decisions choose a footprint, material palette, roof, windows, entrance, ornament, and landscape. The current reward is a **handcrafted preference heuristic**; it is a prototype, not a learned visual-appeal judge.

## Train

Use the repository virtualenv (the system Python may not have Gymnasium installed):

```bash
.venv/bin/python -m builder.train_builder --episodes 500000 --seed 20261125
```

The callback stops on exactly the requested total number of completed construction episodes. To continue the existing 50,000-episode checkpoint to 500,000 total, run:

```bash
.venv/bin/python -m builder.train_builder \
  --resume models/builder/builder_050000_episodes \
  --episodes 500000 --seed 20261125 --checkpoint-every 50000 \
  --out-dir models/builder
```

For training from scratch, omit `--resume`. The final checkpoint is named `builder_500000_episodes.zip`; interim checkpoints go under `models/builder/checkpoints/`, and `metrics.json` reports the cumulative completed count and quality proxy. `--episodes` is the total target even when resuming; the trainer derives the starting episode count from the checkpoint. Override output with `--out-dir`.

## Sample and export

```bash
.venv/bin/python -m builder.sample_builder \
  --model models/builder/builder_500000_episodes \
  --theme cottage --out build-export/plan.json \
  --mcfunction-dir build-export/function --origin 0 64 0
```

The exporter creates `plan.json` and `generated_house.mcfunction`, with ordinary `setblock` commands targeting Minecraft Java 1.16.5. `--origin X Y Z` is the plan's ground-level local anchor (the house's front entrance points toward local +Z before rotation); the floor is one block above the foundation. Rotation is quarter-turns and the sampled lot size can be overridden with 0=small, 1=normal, 2=large.

## Neural voxel mansion

The high-level PPO selects the theme and design choices. A separate coordinate-conditioned neural network then predicts the block token at every coordinate in a 31×17×31 grid. Train that low-level model with:

```bash
.venv/bin/python -m builder.train_voxel_model \
  --policy models/builder/builder_500000_episodes \
  --examples 96 --epochs 30 --out models/builder/voxel_mansion_net.pt
```

The examples come from a procedural teacher renderer, but **inference does not call that renderer**: it uses the voxel network's predictions for all placements. The training set is synthetic, so this is neural imitation of those examples, not a model that learned from real Minecraft screenshots.

Generate the network-built mansion and datapack with:

```bash
.venv/bin/python -m builder.build_neural_mansion \
  --policy models/builder/builder_500000_episodes \
  --voxel-model models/builder/voxel_mansion_net.pt --theme desert \
  --origin 100 64 100 --out-dir build-export/neural-mansion
```

The command writes a preview, plan, `.mcfunction`, and a Java 1.16.5 datapack ZIP. The sample uses origin X=100, Y=64, Z=100 and clears the target bounding box before placement; use an empty area or regenerate with a safe `--origin`. Put `neural-mansion-datapack.zip` in the world's `datapacks` folder, run `/reload`, then `/function builder:mansion`.

Reproduce the saved per-theme evaluation and compare it with uniformly random plans using:

```bash
.venv/bin/python -m builder.evaluate_builder \
  --model models/builder/builder_500000_episodes \
  --out models/builder/evaluation.json
```

The reported score is still only the handcrafted proxy. To regenerate the four isometric previews from actual model block plans:

```bash
.venv/bin/python -m builder.render_preview \
  --model models/builder/builder_500000_episodes --out-dir build-previews
```

To use it in a single-player Java 1.16.5 world, copy just the `.mcfunction` into `<world>/datapacks/house/data/builder/functions/generated_house.mcfunction` and create `<world>/datapacks/house/pack.mcmeta` with:

```json
{"pack":{"pack_format":6,"description":"Generated builder house"}}
```

Then run `/reload` and `/function builder:generated_house` (creative/operator permissions required). The function clears its own bounding box before placement, so choose an empty flat site and test in a disposable world first. It does not terraform uneven terrain. Avoid executing over builds you want to keep.

## Current limitation

The 500,000-episode PPO optimizes only the handcrafted preference score, not human ratings or screenshots. The voxel network is trained to imitate procedural block plans; it predicts every output cell, but this does not establish independent architectural creativity or live-game robustness. The generated datapack has not been executed in a Minecraft world and does not terraform uneven terrain. These builder models remain separate from the PvP Forge model.
