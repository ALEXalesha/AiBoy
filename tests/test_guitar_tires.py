"""Гитару он любит меньше, чем раньше (Алексей, 2.0: «он будто бы слишком любит гитару ПРЯМ ОЧЕНЬ,
надо сделать, чтобы он быстрее уставал и меньше её брал»)."""
import numpy as np

from acts.choice import Choice


def test_the_guitar_tires_him_faster_than_a_toy():
    for freq in ("rare", "sometimes", "often"):
        c = Choice(freq=freq, rng=np.random.default_rng(0))
        c.add("ball", 5.0, liking=0.5)
        assert c.acts["guitar"].half < c.acts["ball"].half, freq


def test_with_the_same_liking_he_picks_a_toy_over_the_guitar():
    picks = {"guitar": 0, "ball": 0}
    for seed in range(400):
        c = Choice(rng=np.random.default_rng(seed))
        c.add("ball", 5.0, liking=0.6)
        c.acts["guitar"].liking = 0.6
        c.acts["explore"].liking = 0.1
        c.since = 60.0
        name = c.choose(["explore", "guitar", "ball"])
        picks[name] = picks.get(name, 0) + 1
    assert picks["ball"] > 3 * picks["guitar"], picks
