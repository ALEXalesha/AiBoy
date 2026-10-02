"""Собака в жизни: появляется по настройке, он её приручает, неручная уходит и возвращается та
же, ручная - со своей памятью в своём мире, мяч у неё в зубах."""
import os

import numpy as np

from game import life_store, models
from game.life import Life


def life_with(mode="never", seed=5, **kw):
    return Life(models.load(), seed, 7, 160.0, models.starter_brain(), dog_mode=mode, **kw)


def test_never_means_never_and_often_brings_a_dog_to_most_worlds():
    assert all(life_with("never", s).dog is None for s in range(10))
    with_dog = sum(life_with("often", s).dog is not None for s in range(20))
    assert with_dog >= 12                                    # около 80 %
    life = life_with("never")
    for _ in range(60 * 130):                                # и потом не приходит
        life.tick()
    assert life.dog is None


def test_he_tames_the_dog_by_his_own_head_in_the_world(tmp_path, monkeypatch):
    monkeypatch.setenv("AIBOY_HOME", str(tmp_path))
    life = life_with()
    dog = life.spawn_dog()
    c = life.choice
    for n in c.acts:
        if n not in ("dog", "eat", "drink", "rest"):
            c.acts[n].liking = 0.0
    c.acts["dog"].liking = 1.0
    life.dog_stay = 1e9
    best = 0.0
    for _ in range(60 * 480):
        life.tick()
        best = max(best, dog.trust)
        if dog.tamed:
            break
    entries = life.toy_journals["dog"].entries
    assert entries and all(e["kind"] in ("tame", "pet", "throw") for e in entries)
    assert best > 0.3 and any(e.get("took") for e in entries), (best, len(entries))


def test_an_untamed_dog_leaves_and_the_same_one_comes_back(tmp_path, monkeypatch):
    monkeypatch.setenv("AIBOY_HOME", str(tmp_path))
    from animals.dog import spawn
    monkeypatch.setattr(spawn, "minute_chance", lambda mode, mood: 1.0 if mode != "never" else 0.0)
    life = life_with("often")
    if life.dog is None:
        life.spawn_dog()
    life.dog.trust = 0.25
    life.dog_stay = 1.0
    events, left_with = [], None
    for _ in range(60 * 600):
        events += [e[0] for e in life.tick() if e[0] in ("dog_left", "dog_came")]
        if "dog_left" in events and left_with is None:
            left_with = life.dog_saved[0]["trust"]
        if "dog_left" in events and life.dog is not None:
            break
    assert "dog_left" in events and life.dog is not None
    assert life.dog.trust == left_with                      # та же собака, со своим доверием


def test_the_dog_remembers_in_its_world(tmp_path, monkeypatch):
    monkeypatch.setenv("AIBOY_HOME", str(tmp_path))
    life = life_with()
    dog = life.spawn_dog()
    dog.trust, dog.tamed = 0.9, True
    dog.choice.acts["fetch"].liking = 0.77
    assert life_store.save(life)
    saved = life_store.load_dog(5)
    assert saved is not None and saved[0]["tamed"] and saved[1] is not None and saved[2] is not None
    again = life_with("sometimes", dog_saved=saved)
    assert again.dog is not None and again.dog.tamed and again.dog.choice.acts["fetch"].liking == 0.77
    assert abs(again.dog.dist(again.human.px)) < 5.0         # ручная - сразу рядом
    assert life_store.load_dog(6) is None
    jp, _, _ = life_store.dog_paths(7)
    with open(jp, "w", encoding="utf-8") as f:
        f.write("{битый")
    assert life_store.load_dog(7) is None


def test_the_ball_in_the_dogs_mouth_is_drawn_there_and_can_not_be_chosen():
    life = life_with()
    dog = life.spawn_dog(near=True)
    dog.tamed = True
    phys = life.physics()
    phys.add_ball("мяч", dog.x, dog.y)
    life.inv.where["ball"] = float(dog.x)
    assert life.dog_ctx.take_ball()
    dog.carrying = True
    assert life.inv.where["ball"] == "dog" and "ball" not in life.available()
    balls = [v for v in life.toy_visuals() if v[0] == "ball"]
    hx, hy = dog.body.head()
    assert balls and abs(balls[0][1] - hx) < 0.3
    life.dog_ctx.drop_ball(dog.x, dog.y)
    assert isinstance(life.inv.where["ball"], float)
