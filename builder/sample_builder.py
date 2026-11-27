"""Sample an architectural plan from the separately trained builder checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from builder.export_plan import export_plan, plan_from_choices
from env.builder_env import BuilderEnv, OPTION_NAMES, THEMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="SB3 model path, with or without .zip")
    parser.add_argument("--theme", choices=tuple(str(i) for i in range(len(THEMES))) + THEMES,
                        help="theme index 0-3 or name: cottage, alpine, desert, fantasy")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="builder-plan.json")
    parser.add_argument("--mcfunction-dir", help="also export a Minecraft function file")
    parser.add_argument("--origin", nargs=3, type=int, default=(0, 64, 0))
    parser.add_argument("--rotation", type=int, choices=range(4), help="override the sampled quarter-turn rotation")
    parser.add_argument("--lot-scale", type=int, choices=range(3), help="override the sampled lot-size variant")
    args = parser.parse_args()

    model = PPO.load(args.model, device="cpu")
    env = BuilderEnv()
    theme_request = args.theme
    if theme_request is not None and theme_request.isdigit():
        theme_request = int(theme_request)
    obs, _ = env.reset(seed=args.seed, options={"theme": theme_request} if theme_request is not None else None)
    choices: list[int] = []
    for _ in OPTION_NAMES:
        action, _ = model.predict(obs, deterministic=not args.stochastic)
        obs, _, terminated, truncated, _ = env.step(int(np.asarray(action).item()))
        choices.append(int(np.asarray(action).item()))
        if terminated or truncated:
            break
    theme_id = int(env.theme)
    plan = plan_from_choices(
        theme_id,
        choices,
        rotation=env.rotation if args.rotation is None else args.rotation,
        lot_scale=env.lot_scale if args.lot_scale is None else args.lot_scale,
        season=env.season,
        checkpoint=str(args.model),
    )
    plan["reward"] = round(float(plan["reward"]), 6)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"plan": str(output), "theme": plan["theme"], "choices": plan["design"],
                      "quality_reward": plan["reward"]}, indent=2))
    if args.mcfunction_dir:
        result = export_plan(plan, args.mcfunction_dir, tuple(args.origin))
        print(json.dumps(result, indent=2))
    env.close()


if __name__ == "__main__":
    main()
