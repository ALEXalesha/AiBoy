"""Общий закон выбора (спецификация 2.0, раздел 13): ни одно занятие не длится вечно - на 5
человечках за 30 минут игрового времени каждый сеанс каждой игры (гитара, игрушки, плавание,
собака, гора, маска) кончается сам: медиана 2-6 минут, ни один не дольше 12."""
import numpy as np
import pytest

from tools.measure_sessions import BODIES, all_sessions


@pytest.fixture(scope="module")
def runs():
    return [all_sessions(b, 30) for b in BODIES]


def test_every_session_of_every_game_ends_by_itself(runs):
    done = [length for run in runs for _, length in run if length > 0]
    every = [abs(length) for run in runs for _, length in run]
    assert done, runs
    med = float(np.median(done)) / 60
    assert 2.0 <= med <= 6.0, (med, sorted(round(v / 60, 1) for v in done))
    assert max(every) / 60 <= 12.0, max(every) / 60


def test_he_plays_more_than_one_game(runs):
    kinds = {a for run in runs for a, _ in run}
    assert len(kinds) >= 4, kinds
