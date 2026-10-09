"""Велосипед не проваливается в землю (Алексей, 2.0: «с велосипедом заезжает в текстуры, у него
должны быть нормальные хитбоксы», снимки 211119/211311). Замер 09.10: на крутом склоне колесо
проскакивало сквозь тонкий отрезок рельефа и оставалось внутри холма - до 0,88 м."""
import numpy as np

from game import models
from toys.bicycle import WHEEL_R, BikeHeads, RideAttempt
from toys.inplace import Ctx
from world.generate import generate

M = models.load()


def worst_sink(seed, starts):
    w = generate(M.world_genome, seed, 160.0)
    rng = np.random.default_rng(0)
    worst = 0.0
    for start in starts:
        att = RideAttempt(BikeHeads(seed=1), Ctx(w, float(start), rng=rng))
        ev = []
        for _ in range(360):
            if att.step(1 / 30, ev) is not None:
                break
            for wh in (att.bike.rear, att.bike.front):
                worst = max(worst, w.height_at(wh.position.x) + WHEEL_R - wh.position.y)
            fr = att.bike.frame.position
            worst = max(worst, w.height_at(fr.x) - fr.y)
    return worst


def test_wheels_and_frame_stay_on_top_of_the_ground():
    starts = np.linspace(5, 150, 12)
    for seed in (7100, 7123):
        assert worst_sink(seed, starts) <= 0.12, seed
