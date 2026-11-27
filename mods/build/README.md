# Neural Woodland Cottage datapack

**Installable artifact:** [`neural-woodland-cottage-datapack.zip`](neural-woodland-cottage-datapack.zip)

The build PPO policy selects the design; the coordinate-conditioned voxel network predicts every block placement. The procedural renderer is used only for training examples, not at inference. The included plan was generated with the 606,000-episode builder checkpoint. Its policy style score is a handcrafted proxy, not a visual-quality rating.

Supporting artifacts:

- [`mansion-plan.json`](mansion-plan.json) — selected design and predicted block histogram.
- [`mansion.mcfunction`](mansion.mcfunction) — placement commands.
- [`mansion-preview.png`](mansion-preview.png) — isometric rendering of the block plan, not a live Minecraft screenshot.

## Install

For Minecraft Java 1.16.5, copy the ZIP into the world's `datapacks` folder, run `/reload`, then run `/function builder:mansion`. The function clears **X=85–114, Y=64–79, Z=88–116** before placing blocks, so use an empty site. It does not terraform uneven terrain. The datapack has not been tested in a live Minecraft world.

Source and generator: [`../../builder/`](../../builder/), especially `build_neural_mansion.py`; active voxel weights are in `models/builder/voxel_mansion_net.pt`.