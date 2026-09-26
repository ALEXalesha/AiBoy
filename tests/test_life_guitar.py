"""Гитара в жизни: лежит перед ним, подбирает, садится, когда мир надоел, попытки идут
своим временем, мозг ходьбы в это время не учится, встаёт, когда надоела музыка."""
import numpy as np
import pytest

from brain.brain import Brain
from game import models
from game.life import GUITAR_AHEAD, SIT_TIME, Life
from music.journal import Journal
from music.mood import Mood
from music.musician import Musician
from music.session import Guitarist
from music.songbook import Songbook


@pytest.fixture(scope="module")
def m():
    return models.load()


def make(m, tmp_path, has_guitar=True, mood=None):
    g = Guitarist(Musician(seed=1), Journal(tmp_path / "j.json"), Songbook(tmp_path / "s.json"),
                  rng=np.random.default_rng(0))
    return Life(m, 3, 4, 80.0, Brain(seed=0), seed=0, guitarist=g, has_guitar=has_guitar, mood=mood)


def bored_mood():
    mood = Mood("often")
    mood.world, mood.music, mood.world_time = 0.0, 0.8, 30.0
    return mood


def test_guitar_lies_ahead_and_is_picked_up_when_he_passes(m, tmp_path):
    life = make(m, tmp_path, has_guitar=False)
    assert life.guitar_x == pytest.approx(life.human.px + GUITAR_AHEAD)
    life.human.px = life.guitar_x - 0.3
    events = life.tick()
    assert ("pickup",) in events and life.has_guitar


def test_bored_of_the_world_he_sits_down_and_plays(m, tmp_path):
    life = make(m, tmp_path, mood=bored_mood())
    events = []
    for _ in range(3):
        events += life.tick()
    assert ("sit",) in events and life.playing


def test_an_attempt_takes_its_own_time_and_the_walking_brain_rests(m, tmp_path):
    life = make(m, tmp_path, mood=bored_mood())
    life.tick()
    assert life.playing
    steps = life.brain.steps
    weights = life.brain.cell.Wx.copy()
    kinds, started = [], None
    for i in range(60 * 20):
        for e in life.tick():
            kinds.append(e[0])
            if e[0] == "attempt_start" and started is None:
                started = (life.time, e[1])
            if e[0] == "attempt":
                break
        if "attempt" in kinds:
            break
    assert started is not None and started[0] == pytest.approx(SIT_TIME, abs=0.05)
    assert life.brain.steps == steps
    np.testing.assert_array_equal(weights, life.brain.cell.Wx)
    assert len(life.guitarist.journal.entries) == 1
    attempt = started[1]
    assert life.time - started[0] == pytest.approx(attempt.duration + 0.8, abs=0.05)


def test_he_sits_while_playing_and_arms_follow_the_attempt(m, tmp_path):
    life = make(m, tmp_path, mood=bored_mood())
    for _ in range(int(60 * 3)):
        life.tick()
    a = life.attempt
    assert a is not None
    fr = min(len(a.arms) - 1, int(life.play_t * 60) - 1)
    assert life.human.angles[2] == pytest.approx(a.arms[fr, 0], abs=1e-6)
    assert life.human.angles[6] > 1.0                     # бёдра вперёд - сидит


def test_boring_music_makes_him_stand_up(m, tmp_path):
    life = make(m, tmp_path, mood=bored_mood())
    life.tick()
    life.mood.music = 0.0
    life.mood.world = 1.0
    stood = False
    for _ in range(60 * 60):
        life.mood.music = 0.0
        if ("stand",) in life.tick():
            stood = True
            break
    assert stood and not life.playing
    assert len(life.guitarist.journal.entries) >= 3            # не раньше трёх попыток


def test_owner_rating_during_an_attempt_goes_into_it(m, tmp_path):
    life = make(m, tmp_path, mood=bored_mood())
    for _ in range(60 * 3):
        life.tick()
    life.rate(1)
    entry = None
    for _ in range(60 * 20):
        for e in life.tick():
            if e[0] == "attempt":
                entry = e[1]
        if entry:
            break
    assert entry["rating"] == 1 and entry["C"] == 1.0


def test_no_guitarist_no_guitar_play(m, tmp_path):
    life = Life(m, 3, 4, 80.0, Brain(seed=0), seed=0, has_guitar=True, mood=bored_mood())
    for _ in range(30):
        life.tick()
    assert not life.playing
