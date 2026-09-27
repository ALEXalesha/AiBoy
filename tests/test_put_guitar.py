"""«Положи гитару» - просьба владельца, а не команда; положенная гитара лежит в мире."""
import numpy as np
import pytest

from brain.brain import Brain
from game import app as app_module
from game import models
from game.life import GUITAR_AHEAD, Life
from music.journal import Journal
from music.musician import Musician
from music.session import Guitarist
from music.songbook import Songbook


@pytest.fixture(scope="module")
def m():
    return models.load()


def make(m, tmp_path, seed=0):
    g = Guitarist(Musician(seed=1), Journal(tmp_path / "j.json"), Songbook(tmp_path / "s.json"),
                  rng=np.random.default_rng(seed))
    return Life(m, 3, 4, 80.0, Brain(seed=0), seed=seed, guitarist=g, has_guitar=True)


def playing(life):
    c = life.choice
    c.acts["explore"].liking, c.acts["guitar"].liking, c.since = 0.0, 0.9, 60.0
    life.decide_clock = 5.0
    for _ in range(60 * 3):
        life.tick()
    assert life.playing and life.attempt is not None
    lukewarm(life)


def lukewarm(life):
    """Играть ему ни то ни сё: согласится примерно в половине случаев."""
    c = life.choice
    c.acts["guitar"].liking, c.acts["guitar"].boredom, c.acts["explore"].liking = 0.5, 0.3, 0.55


def agree_rate(m, tmp_path, liking, boredom, world, n=120):
    yes = 0
    for i in range(n):
        life = make(m, tmp_path, seed=i)
        c = life.choice
        c.acts["guitar"].liking, c.acts["guitar"].boredom, c.acts["explore"].liking = liking, boredom, world
        agree, _ = life.ask_put()
        yes += agree
    return yes / n


def test_an_engaged_player_says_no_more_often_than_a_bored_one(m, tmp_path):
    keen = agree_rate(m, tmp_path, liking=0.9, boredom=0.0, world=0.3)
    bored = agree_rate(m, tmp_path, liking=0.3, boredom=0.6, world=0.8)
    assert keen < 0.3 < 0.7 < bored, (keen, bored)


def test_after_no_he_plays_on_and_after_yes_he_stands_and_puts_it_down(m, tmp_path):
    for seed in range(40):
        life = make(m, tmp_path, seed=seed)
        playing(life)
        n = len(life.guitarist.journal.entries)
        a = life.attempt
        agree, events = life.ask_put()
        said = [e[1] for e in events if e[0] == "phrase"]
        assert said and said[0].strip()
        if agree:
            assert ("stand",) in events and ("put",) in events
            assert not life.playing and not life.has_guitar and life.guitar_laid
            assert life.guitar_x == pytest.approx(life.human.px)
            assert len(life.guitarist.journal.entries) == n            # брошенная попытка без записи
            return
        assert life.playing and life.attempt is a                    # доигрывает
        for _ in range(60 * 12):
            life.tick()
        assert len(life.guitarist.journal.entries) > n
    pytest.fail("ни разу не согласился за 40 просьб")


def test_no_guitar_no_request(m, tmp_path):
    life = make(m, tmp_path)
    life.has_guitar = False
    assert life.ask_put() is None


def test_a_laid_guitar_is_not_picked_up_in_passing_only_when_he_comes_to_play(m, tmp_path):
    life = make(m, tmp_path)
    life.put_guitar()
    for _ in range(30):
        life.tick()
    assert not life.has_guitar and life.walk.places.marks["гитара"] == pytest.approx(life.guitar_x)
    life.seeking = life.time
    life.human.px = life.guitar_x
    events = life.tick()
    assert ("pickup",) in events and life.has_guitar and "гитара" not in life.walk.places.marks


def bore(win):
    life = win.observe.life
    life.has_guitar = True
    c = life.choice
    c.acts["explore"].liking, c.acts["guitar"].liking, c.since = 0.0, 0.9, 60.0
    life.decide_clock = 5.0


def test_buttons_are_on_only_with_the_guitar_and_the_answer_goes_to_the_chat(qapp):
    w = app_module.MainWindow(autostart=False)
    try:
        w.show()
        w.show_page("observe")
        obs = w.observe
        obs.life.has_guitar = False
        obs.refresh_put()
        assert not obs.guitar_panel.put.isEnabled() and not obs.put_button.isEnabled()
        bore(w)
        for _ in range(60 * 3):
            obs.tick()
        assert obs.life.playing and obs.guitar_panel.put.isEnabled()
        lukewarm(obs.life)
        chat = len(obs.life.phrases)
        for _ in range(60):
            agree = obs.ask_put()
            if agree:
                break
            for _ in range(60):
                obs.tick()
        assert len(obs.life.phrases) > chat
        assert agree and not obs.life.has_guitar and not obs.guitar_panel.put.isEnabled()
        # новый мир: гитара снова в сундуке, а сундук - рядом с ним
        obs.new_world()
        life = obs.life
        assert life.inv.where["guitar"] == "chest"
        assert life.inv.chest_x == pytest.approx(life.human.px + GUITAR_AHEAD * life.human.facing)
    finally:
        w.close()
