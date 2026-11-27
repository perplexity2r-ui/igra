from __future__ import annotations

import json
import re

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from builder.export_plan import export_plan, plan_from_choices, placement_commands
from builder.mansion import render_mansion_blocks
from builder.render import render_blocks
from env.builder_env import BuilderEnv, DESIGN_CHOICES, N_STAGES, THEMES, design_quality


def default_choices() -> list[int]:
    return [0] * N_STAGES


def test_builder_environment_contract_and_episode_reward() -> None:
    env = BuilderEnv()
    check_env(env, skip_render_check=True)
    obs1, info1 = env.reset(seed=31, options={"theme": "alpine"})
    obs2, info2 = env.reset(seed=31, options={"theme": "alpine"})
    assert obs1.shape == (74,)
    assert np.array_equal(obs1, obs2)
    assert info1 == info2
    assert info1["theme"] == THEMES.index("alpine")

    final_info = {}
    for stage in range(N_STAGES):
        obs, reward, terminated, truncated, final_info = env.step(stage % 8)
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        assert truncated is False
        assert terminated is (stage == N_STAGES - 1)
    assert 0.0 <= final_info["quality"] <= 1.0
    assert len(final_info["plan"]) == N_STAGES
    env.close()


def test_quality_score_validates_and_stays_bounded() -> None:
    for theme in range(len(THEMES)):
        for choices in (default_choices(), list(reversed(range(N_STAGES)))):
            assert 0.0 <= design_quality(choices, theme) <= 1.0
    with pytest.raises(ValueError):
        design_quality([0], 0)


def test_renderer_covers_all_design_options_and_uses_1165_blocks() -> None:
    assert len(DESIGN_CHOICES) == N_STAGES == 7
    for stage, count in enumerate(DESIGN_CHOICES):
        for choice in range(count):
            choices = default_choices()
            choices[stage] = choice
            blocks = render_blocks({"choices": choices})
            assert blocks
            assert all(block.startswith("minecraft:") for block in blocks.values())
            assert all(y >= 0 for _, y, _ in blocks)
            # These two blocks were added after the Forge 1.16.5 target version.
            assert not any(block.split("[", 1)[0] in {
                "minecraft:lightning_rod", "minecraft:moss_block"
            } for block in blocks.values())


def test_rendered_door_direction_rotates_with_the_build() -> None:
    choices = default_choices()
    plan = plan_from_choices(0, choices, rotation=0)
    south_door = [block for block in render_blocks(plan).values()
                  if "_door[" in block and "facing=south" in block]
    assert south_door
    plan["rotation"] = 1
    west_door = [block for block in render_blocks(plan).values()
                 if "_door[" in block and "facing=west" in block]
    assert west_door


def test_export_creates_datapack_function_and_validates_command_limits(tmp_path) -> None:
    choices = [1, 2, 0, 3, 2, 5, 4]
    plan = plan_from_choices("desert", choices, rotation=3, lot_scale=2)
    result = export_plan(plan, tmp_path / "pack" / "data" / "demo" / "functions", (10, 64, -7))
    plan_file = tmp_path / "pack" / "data" / "demo" / "functions" / "plan.json"
    function_file = tmp_path / "pack" / "data" / "demo" / "functions" / "generated_house.mcfunction"
    saved_plan = json.loads(plan_file.read_text())
    commands = function_file.read_text().splitlines()
    assert saved_plan["theme"] == "Desert Courtyard"
    assert saved_plan["block_count"] == result["block_count"]
    assert result["command_count"] == len(commands) == result["block_count"] + 1
    assert result["fill_volume"] <= 32_768
    assert commands[0].startswith("fill  ") is False
    assert commands[0].endswith(" air")
    assert all(command.startswith("setblock ") for command in commands[1:])
    assert all(re.match(r"setblock -?\d+ -?\d+ -?\d+ minecraft:[a-z0-9_]+", command)
               for command in commands[1:])

    parsed = placement_commands(render_blocks(plan), (10, 64, -7))
    assert parsed == commands


def test_plan_and_function_name_validation(tmp_path) -> None:
    with pytest.raises(ValueError):
        plan_from_choices(4, default_choices())
    with pytest.raises(ValueError):
        plan_from_choices(0, [0] * 6)
    with pytest.raises(ValueError):
        plan_from_choices(0, [8] + [0] * 6)
    plan = plan_from_choices(0, default_choices())
    with pytest.raises(ValueError):
        export_plan(plan, tmp_path, (0, 64, 0), "../escape")


def test_mansion_renderer_expands_policy_plan_to_a_large_executable_build() -> None:
    plan = plan_from_choices(2, [2, 2, 6, 0, 6, 6, 2], rotation=0, lot_scale=2)
    blocks = render_mansion_blocks(plan)
    commands = placement_commands(blocks, (0, 64, 0))
    xs = [x for x, _, _ in blocks]
    ys = [y for _, y, _ in blocks]
    zs = [z for _, _, z in blocks]
    assert len(blocks) > 1_000
    assert max(xs) - min(xs) >= 25
    assert max(zs) - min(zs) >= 25
    assert max(ys) >= 16
    assert min(ys) == 0
    assert sum("_door[" in block for block in blocks.values()) == 4
    assert sum(block == "minecraft:water" for block in blocks.values()) >= 9
    assert commands[0].startswith("fill ")
    assert len(commands) < 65_536
