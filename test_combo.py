"""Тесты механики комбо (запуск: .venv/bin/python -m pytest test_combo.py -q).

Проверяем ровно то, что просил пользователь: агент умеет *выходить* из комбо-цепочки
и *ставить* её сам, а среда это вознаграждает.
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")

from env.pro_vec_env import (  # noqa: E402
    A_COMBO_BREAK,
    A_COMBO_COUNTER,
    T_COMBO_BREAK,
    T_COMBO_COUNTER,
    ProArena,
)


def make(agg: float = 1.0, curriculum: float = 0.0, n: int = 64, seed: int = 5) -> ProArena:
    a = ProArena(n, seed=seed)
    a.set_aggression(agg)
    a.set_combo_curriculum(curriculum)
    a.reset()
    return a


def test_observation_is_34_dims_with_combo_state():
    a = make(n=4)
    obs = a._obs()
    assert obs.shape == (4, 34)
    a.L.combo_taken[:] = 3
    assert a._obs()[0, 30] > 0.5


def test_curriculum_starts_fights_in_a_chain():
    a = make(curriculum=0.5, n=256)
    in_chain = int((a.L.combo_taken > 0).sum())
    assert 80 < in_chain < 200, f"ожидали ~половину боёв в комбо, получили {in_chain}"
    dist = np.linalg.norm(a.L.pos - a.E.pos, axis=1)
    assert (dist[a.L.combo_taken > 0] < 2.6).all()


def test_combo_break_clears_chain_and_grants_immunity():
    a = make(curriculum=0.9, n=32)
    a.L.combo_taken[:] = 2
    a.L.combo_window[:] = 1.5
    assert (a.L.combo_taken > 0).all()
    _, _, _, _, _, info = a.step(np.full(32, A_COMBO_BREAK, np.int32))
    assert (a.L.combo_taken == 0).all(), "цепочка должна обнуляться"
    assert bool(info["combo_broken"].all())
    assert (a.L.hurt_cd > 0).any(), "после выхода должны быть i-frames"
    assert (a.L.counter_until > 0).all(), "должно открыться окно контратаки"
    assert (info["tech"] == T_COMBO_BREAK).any()


def test_combo_counter_punishes_and_deals_damage():
    a = make(n=32, seed=9)
    for _ in range(12):
        a.step(np.full(32, 10, np.int32))
    a.L.counter_until[:] = 1.0
    a.L.cd[:] = 0.0
    a.L.pos[:] = a.E.pos[:] + np.array([1.6, 0.0], np.float32)
    before = a.E.hp.copy()
    _, _, _, _, _, info = a.step(np.full(32, A_COMBO_COUNTER, np.int32))
    assert bool(info["combo_countered"].any())
    assert (info["tech"] == T_COMBO_COUNTER).any()
    assert (a.E.hp < before).any(), "контратака должна наносить урон"


def test_enemy_chains_the_learner():
    """Противник должен уметь нанести минимум два удара подряд в одну цепочку.

    Проверяем на одном окружении с перезапусками: в большой пачке первый же
    смертельный удар маскирует цепочку (после reset состояние обнуляется).
    """
    a = make(agg=1.0, n=1, seed=11)
    best = 0
    cooldown = 0
    for _ in range(400):
        _, _, _, _, done, info = a.step(np.zeros(1, np.int32))
        best = max(best, int(np.asarray(info["combo_taken"])[0]))
        if best >= 2:
            break
        if bool(done[0]):
            a.reset(np.ones(1, bool))
            cooldown += 1
    assert best >= 2, f"противник должен строить цепочки, максимум {best}"


def test_learner_can_build_a_chain_on_the_enemy():
    a = make(agg=1.0, n=128, seed=33)
    max_chain = 0
    for _ in range(120):
        dist = np.linalg.norm(a.E.pos - a.L.pos, axis=1)
        act = np.where(dist < 3.0, 9, 10)
        _, _, _, _, done, _ = a.step(act.astype(np.int32))
        max_chain = max(max_chain, int(a.E.combo_taken.max()))
        if bool(done.all()):
            a.reset(np.ones(128, bool))
    assert max_chain >= 2, f"ученик должен уметь ставить комбо, максимум {max_chain}"


def test_combo_window_expires():
    a = make(n=8)
    a.L.combo_taken[:] = 3
    a.L.combo_window[:] = 0.05
    a.step(np.zeros(8, np.int32))
    assert (a.L.combo_taken == 0).all(), "просроченное окно должно сбрасывать цепочку"


def test_fight_outcome_plumbing():
    a = make(agg=0.0, n=64, seed=2)
    for _ in range(200):
        _, _, _, _, done, info = a.step(np.zeros(64, np.int32))
        assert info["learner_dead"].dtype == bool and info["enemy_dead"].dtype == bool
        if bool(done.all()):
            break
    assert a.L.hp.min() >= 0.0


def test_human_can_break_and_counter_in_pve():
    """PvE-арена: человек на месте врага тоже может выйти из комбо (R) и наказать (T)."""
    a = make(n=1, seed=9)
    a.pve = True
    a.E.combo_taken[0] = 3
    a.E.combo_window[0] = 1.2
    a.E.stun[0] = 0.3
    a.E.pos[0] = a.L.pos[0] + np.array([1.3, 0.0], np.float32)
    a.E.cd[0] = 0.0
    a.L.cd[0] = 0.0
    a.set_human_action(np.array([A_COMBO_BREAK]))
    _, _, _, _, _, info = a.step(np.array([0]))
    assert bool(info["human_combo_broken"][0]), "R должен разрывать цепочку на игроке"
    assert int(a.E.combo_taken[0]) == 0, "после R цепочка на игроке сброшена"
    assert float(a.E.counter_until[0]) > 0.0, "выход из комбо открывает окно контр-атаки"

    hp0 = float(a.L.hp[0])
    a.E.cd[0] = 0.0
    a.set_human_action(np.array([A_COMBO_COUNTER]))
    _, _, _, _, _, info2 = a.step(np.array([0]))
    assert bool(info2["human_combo_countered"][0]), "T должен засчитывать контр-комбо игроку"
    assert float(a.L.hp[0]) < hp0, "контр-комбо должен наносить урон"


def test_human_chain_reels_the_ai_in_pve():
    """Игрок тоже умеет ставить цепочку: удары подряд станят ИИ и растят combo_taken."""
    a = make(n=1, seed=13)
    a.pve = True
    a.L.stun[0] = 0.0
    a.L.hurt_cd[0] = 0.0
    a.E.pos[0] = a.L.pos[0] + np.array([1.2, 0.0], np.float32)
    a.E.cd[0] = 0.0
    a.set_human_action(np.array([9]))          # sprint-hit
    _, _, _, _, _, info = a.step(np.array([0]))
    assert int(a.L.combo_taken[0]) >= 1 or bool(info["combo_hit"][0]), \
        "удар игрока должен вести цепочку на ИИ"
