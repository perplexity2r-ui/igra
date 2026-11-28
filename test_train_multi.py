import pytest

from train_multi import difficulty_schedule


def test_standard_difficulty_curriculum_ramps_all_enemy_settings():
    assert difficulty_schedule(0.0) == pytest.approx((0.30, 0.55, 0.0))
    assert difficulty_schedule(0.5) == pytest.approx((0.65, 0.775, 0.10))
    assert difficulty_schedule(1.0) == pytest.approx((1.0, 1.0, 0.20))


def test_max_difficulty_uses_full_enemy_settings_from_fight_one():
    assert difficulty_schedule(0.0, max_difficulty=True) == (1.0, 1.0, 0.20)


def test_difficulty_progress_is_clamped():
    assert difficulty_schedule(-1.0) == difficulty_schedule(0.0)
    assert difficulty_schedule(2.0) == difficulty_schedule(1.0)
