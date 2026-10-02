"""Одна скука и один выбор на все занятия (acts/choice.py).

Замер до (1.1.0, план 2.0): у 4 из 5 человечков сеанс гитары не кончался за 30 минут x8
(медиана 26.9 мин), пятый не сел ни разу: интерес к миру пока он играет рос только до 0.55,
интерес к музыке держался на новизне фраз около 0.7, скуки от длительности не было.
"""
import math

import numpy as np
import pytest

from acts import choice as ch
from acts.choice import Choice, pleasure
from tools.measure_sessions import BODIES, sessions


def test_boredom_takes_half_the_pull_in_half_minutes_and_fades_away_after():
    c = Choice("sometimes", np.random.default_rng(0))
    c.switch("guitar")
    g = c.acts["guitar"]
    for _ in range(int(ch.HALF["sometimes"] * 60 / 0.5)):
        c.tick(0.5)
    assert g.boredom == pytest.approx(0.5, abs=0.02)
    c.switch("explore")
    for _ in range(int(ch.RECOVER * 60 / 0.5)):
        c.tick(0.5)
    assert g.boredom == pytest.approx(0.5 / math.e, abs=0.02)
    assert c.acts["explore"].boredom > 0.2               # гулял - надоедает и мир


def test_a_flat_result_is_boring_too():
    c = Choice("sometimes", np.random.default_rng(0))
    c.switch("guitar")
    for _ in range(ch.FLAT_N):
        c.after_attempt("guitar", 0.4)
    flat = c.acts["guitar"].boredom
    c2 = Choice("sometimes", np.random.default_rng(0))
    c2.switch("guitar")
    for i in range(ch.FLAT_N):
        c2.after_attempt("guitar", 0.2 + 0.05 * i)
    assert flat > c2.acts["guitar"].boredom + 0.04


def test_no_switch_before_three_attempts_or_twenty_seconds_unless_a_need_is_urgent():
    c = Choice("sometimes", np.random.default_rng(0))
    c.acts["guitar"].liking = 1.0
    c.acts["explore"].liking = 0.0
    assert c.choose(["explore", "guitar"]) == "explore"       # в мире меньше 20 с
    c.since = ch.MIN_WORLD
    assert c.choose(["explore", "guitar"]) == "guitar"
    c.acts["guitar"].liking, c.acts["explore"].liking = 0.0, 1.0
    for _ in range(2):
        c.after_attempt("guitar", 0.0)
        assert c.choose(["explore", "guitar"]) == "guitar"
    assert c.can_switch({"explore": 0.9})


def test_often_is_longer_and_stronger_than_rare():
    rare, often = Choice("rare"), Choice("often")
    assert often.acts["guitar"].half > rare.acts["guitar"].half
    assert often.acts["guitar"].pull() > rare.acts["guitar"].pull()
    rare.freq = "often"
    assert rare.acts["guitar"].half == often.acts["guitar"].half


def test_pleasure_counts_success_growth_and_the_owner():
    assert pleasure(0.8, [0.2] * 20) > pleasure(0.8, [0.8] * 20) > pleasure(0.2, [0.8] * 20)
    assert pleasure(0.5, [], owner=1) > pleasure(0.5, []) > pleasure(0.5, [], owner=-1)
    assert 0.0 <= pleasure(-3, [5]) <= 1.0


def test_state_survives_as_dict_and_broken_values_are_ignored():
    c = Choice("often")
    c.acts["guitar"].liking, c.acts["guitar"].boredom, c.mood = 0.7, 0.3, 0.2
    d = Choice("often").load(c.to_dict())
    assert (d.acts["guitar"].liking, d.acts["guitar"].boredom, d.mood) == (0.7, 0.3, 0.2)
    e = Choice("often").load({"acts": {"guitar": {"liking": 7, "boredom": "x"}}, "mood": True})
    assert e.acts["guitar"].liking == 0.5 and e.mood == 0.0


@pytest.fixture(scope="module")
def sometimes():
    return [sessions(b, 30, "sometimes") for b in BODIES]


def test_every_guitar_session_ends_by_itself_in_reasonable_time(sometimes):
    runs = [r[0] for r in sometimes]
    lengths = [s["length"] / 60 for out in runs for s in out if not s.get("unfinished")]
    assert lengths and 2.0 <= float(np.median(lengths)) <= 6.0, lengths
    assert max(s["length"] / 60 for out in runs for s in out) <= 12.0
    assert all(s["attempts"] >= ch.MIN_ATTEMPTS for out in runs for s in out if not s.get("unfinished"))


def test_after_a_session_he_walks_and_later_plays_again(sometimes):
    runs = [r[0] for r in sometimes]
    # с 2.0 у него есть и мяч, и вода, и нужды: гитаре достаётся меньше; кто-то из пяти может
    # за полчаса так к ней и не вернуться (замер: 2, 2, 0, 3, 2 сеанса)
    assert sum(len(out) >= 2 for out in runs) >= 3, [len(o) for o in runs]
    for out in runs:
        for a, b in zip(out, out[1:]):
            assert b["start"] - (a["start"] + a["length"]) >= ch.MIN_WORLD - 1e-6


def test_often_means_longer_and_more_play_than_rare():
    rare = [sessions(b, 30, "rare")[0] for b in BODIES[:3]]
    often = [sessions(b, 30, "often")[0] for b in BODIES[:3]]
    total = lambda runs: sum(s["length"] for out in runs for s in out)  # noqa: E731
    med = lambda runs: float(np.median([s["length"] for out in runs for s in out]))  # noqa: E731
    assert total(often) > total(rare) * 1.2 and med(often) > med(rare)


# --- положить гитару или взять с собой ---

def test_standing_up_he_sometimes_leaves_the_guitar_and_sometimes_takes_it(sometimes):
    ended = [s for out, *_ in sometimes for s in out if not s.get("unfinished")]
    put = sum(1 for s in ended if s.get("put"))
    assert 0 < put < len(ended), (put, len(ended))


def test_a_laid_guitar_stays_where_he_put_it_and_he_comes_back_for_it(sometimes):
    from game.life import PICKUP
    laid = [x for *_, lay in sometimes for x in lay]
    back = [(x, y) for _, x, y in laid if y is not None]
    assert back, laid
    for x, y in back:
        assert abs(((y - x) + 80) % 160 - 80) < PICKUP + 0.2
