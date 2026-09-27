"""Гитара в жизни: лежит перед ним, подбирает, садится, когда гитара тянет сильнее мира,
попытки идут своим временем, мозг ходьбы в это время не учится, встаёт, когда надоела."""
import numpy as np
import pytest

from brain.brain import Brain
from game import models
from acts.choice import DECIDE_EVERY, Choice
from game.life import GUITAR_AHEAD, SIT_TIME, Life
from music.journal import Journal
from music.musician import Musician
from music.session import Guitarist
from music.songbook import Songbook


@pytest.fixture(scope="module")
def m():
    return models.load()


def make(m, tmp_path, has_guitar=True, choice=None):
    g = Guitarist(Musician(seed=1), Journal(tmp_path / "j.json"), Songbook(tmp_path / "s.json"),
                  rng=np.random.default_rng(0))
    return Life(m, 3, 4, 80.0, Brain(seed=0), seed=0, guitarist=g, has_guitar=has_guitar, choice=choice)


def bored_mood():
    """Мир ему сейчас неинтересен, гитара тянет: выбор - сесть играть при первой точке выбора."""
    c = Choice("often", np.random.default_rng(0))
    c.acts["explore"].liking, c.acts["guitar"].liking, c.since = 0.0, 0.9, 30.0
    return c


def first_choice_now(life):
    life.decide_clock = DECIDE_EVERY


def test_the_guitar_waits_in_the_chest_and_he_fetches_it_when_he_wants_to_play(m, tmp_path):
    life = make(m, tmp_path, has_guitar=False, choice=bored_mood())
    assert life.inv.chest_x == pytest.approx(life.human.px + GUITAR_AHEAD * life.human.facing)
    assert life.inv.where["guitar"] == "chest" and life.guitar_x is None
    life.human.px = life.inv.chest_x + 0.3                 # проходит мимо - не берёт
    for e in life.tick():
        assert e != ("pickup",)
    first_choice_now(life)
    life.tick()
    assert life.seeking is not None and life.walk.pulls.get("гитара") == 1.0
    assert life.walk.places.marks["гитара"] == pytest.approx(life.inv.chest_x)
    events = []
    for _ in range(60 * 60):
        events += life.tick()
        if life.playing:
            break
    assert ("pickup",) in events and ("sit",) in events and life.has_guitar


def test_bored_of_the_world_he_sits_down_and_plays(m, tmp_path):
    life = make(m, tmp_path, choice=bored_mood())
    first_choice_now(life)
    events = []
    for _ in range(3):
        events += life.tick()
    assert ("sit",) in events and life.playing


def test_an_attempt_takes_its_own_time_and_the_walking_brain_rests(m, tmp_path):
    life = make(m, tmp_path, choice=bored_mood())
    first_choice_now(life)
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
    life = make(m, tmp_path, choice=bored_mood())
    first_choice_now(life)
    for _ in range(int(60 * 3)):
        life.tick()
    a = life.attempt
    assert a is not None
    fr = min(len(a.arms) - 1, int(life.play_t * 60) - 1)
    assert life.human.angles[2] == pytest.approx(a.arms[fr, 0], abs=1e-6)
    assert life.human.angles[6] > 1.0                     # бёдра вперёд - сидит


def test_boring_music_makes_him_stand_up_and_say_so(m, tmp_path):
    life = make(m, tmp_path, choice=bored_mood())
    first_choice_now(life)
    life.tick()
    assert life.playing
    c = life.choice
    flags_when_speaking = []
    say = life.say

    def spy():
        flags_when_speaking.append(dict(life.music_flags))
        return say()
    life.say = spy
    stood, said = False, None
    for _ in range(60 * 60):
        c.acts["guitar"].liking, c.acts["explore"].liking = 0.0, 1.0
        events = life.tick()
        if ("stand",) in events:
            stood = True
            said = [e[1] for e in events if e[0] == "phrase"]
            break
    assert stood and not life.playing
    assert len(life.guitarist.journal.entries) >= 3            # не раньше трёх попыток
    assert said and said[-1].strip()                           # и говорит об этом:
    assert flags_when_speaking[-1].get("stood_up") == 1.0      # фраза - о том, что встал


def test_owner_rating_during_an_attempt_goes_into_it(m, tmp_path):
    life = make(m, tmp_path, choice=bored_mood())
    first_choice_now(life)
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
    life = Life(m, 3, 4, 80.0, Brain(seed=0), seed=0, has_guitar=True, choice=bored_mood())
    for _ in range(30):
        life.tick()
    assert not life.playing
