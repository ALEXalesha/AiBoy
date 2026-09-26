"""«Руками»: нота звучит, только если руки дотянулись; промах в награде не слышен; руки
учатся на своём опыте."""
import numpy as np
import pytest

from body.cppn import generate_body
from brain.reach import Reacher
from game import models
from music import hands
from music.journal import Journal
from music.musician import Musician
from music.session import Guitarist
from music.songbook import Songbook


@pytest.fixture(scope="module")
def body():
    return generate_body(models.load().body_genome, 7)


class Exact:
    """Тянется точно в цель (геометрией) - все удары попадают, если рука успевает."""
    trace, prev = {}, {}

    def aim(self, body, lean, target, arm):
        return hands.reach(body, lean, target)

    def learn(self, attempt, body):
        pass


class Away(Exact):
    def aim(self, body, lean, target, arm):
        return (-1.0, 0.0)                        # рука висит - мимо всего


def guitarist(tmp_path, reacher):
    return Guitarist(Musician(seed=1), Journal(tmp_path / "j.json"), Songbook(tmp_path / "s.json"), mode="hands",
                     reacher=reacher, rng=np.random.default_rng(0))


def test_a_miss_is_silent_and_a_hit_sounds(tmp_path, body):
    miss = guitarist(tmp_path, Away()).attempt(body)
    assert miss.intended and all(s == 0 for s in miss.sounded)
    assert miss.notes() == [] and miss.played() == []
    hit = guitarist(tmp_path, Exact()).attempt(body)
    assert np.mean([s == 1 for s in hit.sounded]) > 0.7


def test_a_note_that_did_not_sound_earns_nothing(tmp_path, body):
    g = guitarist(tmp_path, Away())
    a = g.attempt(body)
    entry, _, _ = g.finish(a, body)
    if a.tune is not None:
        assert entry["A"] == 0.0
    assert all(n[5] == 0 for n in entry["notes"])


def test_in_musician_mode_hands_do_not_decide(tmp_path, body):
    g = guitarist(tmp_path, Away())
    g.mode = "musician"
    a = g.attempt(body)
    assert all(s == 1 for s in a.sounded)


def test_hands_learn_to_reach_from_their_own_experience(tmp_path, body):
    """Замер - в плане: за 300 попыток доля ударов, где прозвучала верная нота, растёт с
    долей процента до 20-26 %, средний промах кисти - с 9 до 4 см."""
    g = guitarist(tmp_path, Reacher(seed=1))
    hits, motor = [], []
    for _ in range(300):
        a = g.attempt(body)
        hits.append(np.mean([s == 1 for s in a.sounded]) if a.sounded else np.nan)
        entry, _, _ = g.finish(a, body)
        motor.append(entry["motor"] if entry["motor"] is not None else np.nan)
    first, last = np.nanmean(hits[:50]), np.nanmean(hits[-100:])
    assert last > first + 0.05 and last > 0.08, (first, last)
    assert np.nanmean(motor[-100:]) < 0.7 * np.nanmean(motor[:50])


def test_reacher_save_and_load(tmp_path, body):
    r = Reacher(seed=2)
    r.save(tmp_path / "h.npz")
    other = Reacher(seed=5).load(tmp_path / "h.npz")
    r.explore = other.explore = False
    t = hands.left_target(2, 5, body)
    assert r.aim(body, 0.1, t, "left") == pytest.approx(other.aim(body, 0.1, t, "left"))
