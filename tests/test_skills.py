"""Навыки тела: учатся на своём опыте или REINFORCE; «куда прилетит» - на виденных полётах;
файлы по зерну тела, руки 1.1.0 читаются."""
import math

import numpy as np


import paths
from physics.world import PhysWorld
from skills import store
from skills.base import ExperienceSkill, GaussianSkill
from skills.predict import Predictor
from skills.reach import Reacher


def toy_body(cmd):
    """Нелинейное «тело»: команда -> куда попало (две оси)."""
    a, b = cmd
    return np.array([math.cos(a) + 0.5 * math.cos(a + b), math.sin(a) + 0.5 * math.sin(a + b)])


def test_experience_skill_learns_its_own_body():
    sk = ExperienceSkill(2, [0.0, 0.0], [1.5, 1.5], seed=0)
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(400):
        target = toy_body(rng.uniform(0.1, 1.4, 2))
        cmd = sk.act(target)
        got = toy_body(cmd)
        errs.append(float(np.linalg.norm(got - target)))
        sk.remember(got, cmd)
        sk.learn()
    # замер: медиана промаха за 50 попыток 0.155 -> 0.07 (остаток - шум исследования)
    assert np.median(errs[-50:]) < 0.6 * np.median(errs[:50])


def test_gaussian_skill_climbs_the_reward():
    """Попытки с шумом - шумные; учёба видна по тому, что сеть делает без шума (замер: средний
    промах 0.023 после 3000 попыток)."""
    sk = GaussianSkill(1, [-1.0], [1.0], seed=0, sigma=0.2)

    def greedy_error():
        sk.explore = False
        e = np.mean([abs(float(sk.act(np.array([v]))[0][0]) - 0.7 * v) for v in np.linspace(-1, 1, 21)])
        sk.explore = True
        return e
    before = greedy_error()
    rng = np.random.default_rng(1)
    for _ in range(3000):
        x = rng.uniform(-1, 1, 1)
        cmd, rec = sk.act(x)
        sk.learn(rec, -abs(float(cmd[0]) - 0.7 * float(x[0])))
    assert greedy_error() < min(0.05, before / 4)


class Flat:
    width, dx = 1000.0, 0.25

    def height_at(self, x):
        return 0.0


def flight(p, vx, vy, hand=1.0, start=1.5):
    """Мяч брошен с высоты start; где и когда он опустился до высоты руки."""
    p.remove("мяч")
    p.add_ball("мяч", 0.0, start)
    b = p.body("мяч")
    b.velocity = (vx, vy)
    t = 0.0
    prev = b.position
    while t < 5.0:
        p.step(1 / 60)
        t += 1 / 60
        if b.position.y <= hand and b.velocity.y < 0:
            return b.position.x, t
        prev = b.position
    return prev.x, t


def test_the_predictor_learns_where_a_throw_comes_down_from_flights_it_saw():
    p = PhysWorld(Flat(), center=0.0)
    pred = Predictor(seed=0)
    rng = np.random.default_rng(2)
    errs = []
    for _ in range(80):
        vx, vy = rng.uniform(-6, 6), rng.uniform(2, 8)
        guess_dx, _ = pred.predict(vx, vy, 0.5)
        dx, t = flight(p, vx, vy)
        errs.append(abs(guess_dx - dx))
        pred.saw(vx, vy, 0.5, 0.0, dx, t)
    assert np.median(errs[-10:]) < 0.5 < np.median(errs[:10])



def test_skill_files_live_with_the_person_and_old_hands_are_read(tmp_path):
    old = Reacher(seed=3)
    old.nets["left"].items[0].W[...] = 0.123
    old.save(paths.user_file("guitar", "77", "hands.npz"))       # как в 1.1.0
    r = store.load(77, "reach", lambda: Reacher(seed=1))
    assert np.allclose(r.nets["left"].items[0].W, 0.123)
    assert store.save(77, "reach", r)
    r.nets["left"].items[0].W[...] = 0.5
    assert store.save(77, "reach", r)
    again = store.load(77, "reach", lambda: Reacher(seed=1))
    assert np.allclose(again.nets["left"].items[0].W, 0.5)       # новый файл важнее старого
    open(store.path(78, "reach"), "wb").write(b"not a npz")
    fresh = store.load(78, "reach", lambda: Reacher(seed=9))
    assert np.allclose(fresh.nets["left"].items[0].W, Reacher(seed=9).nets["left"].items[0].W)


def test_old_import_path_still_works():
    from brain.reach import Reacher as R
    assert R is Reacher

