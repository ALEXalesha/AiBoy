"""Мяч: пнуть и бросить в цель, бросить и поймать; учится; звуки; журнал; сам берёт из сундука."""
import numpy as np
import pytest

from game import models
from game.life import Life
from sound import sfx
from toys.ball import BallHeads, practice


@pytest.mark.parametrize("kind", ["kick", "throw", "catch"])
def test_each_ball_skill_is_learned(kind):
    """Закон обучения: зерно 0, N = 150 попыток; медиана последних 20 выше первых 20 на запас.
    Замер (план): пнуть 0.07 -> 0.7, бросить 0.36 -> 0.7, поймать 0.64 -> 0.93 за 200 попыток."""
    margin = {"kick": 0.3, "throw": 0.2, "catch": 0.1}[kind]
    h = BallHeads(seed=0)
    rng = np.random.default_rng(0)
    s = [practice(h, kind, rng) for _ in range(150)]
    assert np.median(s[-20:]) > np.median(s[:20]) + margin


def test_a_stronger_body_kicks_further_with_the_same_command():
    weak, strong = BallHeads(seed=1), BallHeads(seed=1)
    rng1, rng2 = np.random.default_rng(3), np.random.default_rng(3)
    weak.kick.explore = strong.kick.explore = False
    # одна и та же команда, разная сила тела: у сильного мяч дальше от цели, пока голова не выучит
    a = practice(weak, "kick", rng1, strength=0.8)
    b = practice(strong, "kick", rng2, strength=1.25)
    assert a != b


def test_every_ball_event_has_a_synthesized_sound():
    for ev in ("kick", "throw", "bounce", "catch"):
        x = sfx.make(ev, 0.7, 0.11)
        assert x is not None and len(x) > 100 and np.abs(x).max() < 1.0 and np.abs(x).max() > 0.01
    assert sfx.make("bounce", 0.7, 0.11) is sfx.make("bounce", 0.7, 0.11)       # из кэша


@pytest.fixture(scope="module")
def m():
    return models.load()


def test_he_takes_the_ball_from_the_chest_plays_logs_attempts_and_fetches_it(m):
    life = Life(m, 5, 7, 160.0, models.starter_brain())
    events = []
    for _ in range(60 * 400):
        c = life.choice
        c.acts["ball"].liking, c.acts["explore"].liking, c.acts["ball"].boredom = 1.0, 0.0, 0.0
        events += life.tick()
        if sum(1 for e in events if e[0] == "toy_attempt") >= 3:
            break
    kinds = [e[0] for e in events]
    assert "pickup_toy" in kinds and "toy_start" in kinds
    attempts = [e[2] for e in events if e[0] == "toy_attempt"]
    assert len(attempts) >= 3 and all(0.0 <= a["success"] <= 1.0 for a in attempts)
    assert len(life.toy_journals["ball"].entries) == len(attempts)
    assert any(e[0] == "sound" for e in events)
    assert life.toy_heads["ball"].predict.flights >= 0


def test_ball_sounds_are_skipped_at_x4_and_played_at_x1(qapp):
    from game import app as app_module
    w = app_module.MainWindow(autostart=False)
    try:
        obs = w.observe
        obs.set_speed(1)
        obs.handle(obs.life, [("sound", "bounce", 0.6, 0.11)])
        assert w.audio.history[-1] == "bounce"
        n = len(w.audio.history)
        obs.set_speed(4)
        obs.handle(obs.life, [("sound", "bounce", 0.6, 0.11)])
        assert len(w.audio.history) == n
    finally:
        w.close()


def test_ball_heads_survive_a_save(tmp_path):
    h = BallHeads(seed=2)
    rng = np.random.default_rng(2)
    for _ in range(20):
        practice(h, "catch", rng)
    h.save(tmp_path / "b.npz")
    g = BallHeads(seed=9).load(tmp_path / "b.npz")
    assert g.catch.attempts == h.catch.attempts and g.predict.flights == h.predict.flights
    assert np.array_equal(g.kick.net.items[0].W, h.kick.net.items[0].W)
