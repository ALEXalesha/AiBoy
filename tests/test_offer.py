"""«Предложить» из «Каталога» - для каждой игрушки, не только для гитары (Алексей, 2.0: «что сделать,
чтобы он взял самолёт, он вообще ни в какую»).

Законы: согласился - через пару секунд играет именно этой игрушкой (попытка идёт); не согласился -
она лежит у его ног и он её помнит; согласие - выбор (softmax), а не приказ: свежий человечек, который
просто гуляет, берёт каждую игрушку не реже чем в половине случаев, и ни одна не особенная."""
import numpy as np
import pytest

from game import models
from game.life import Life
from toys import registry
from world.generate import generate
from world.water import Water

TOYS = [t.name for t in registry.TOYS.values() if t.name != "guitar" and not t.tool]
M = models.load()
WET = next(s for s in range(7100, 7400) if Water(generate(M.world_genome, s, 160.0)).ponds)   # удочке нужна вода


def fresh(seed, walk=2.0):
    life = Life(M, WET, 7, 160.0, models.starter_brain(), seed=seed)
    for _ in range(int(walk * 60)):              # немного погулял - ничем не занят
        life.tick()
    return life


def test_the_toys_under_test_are_all_the_toys():
    assert set(TOYS) == {"ball", "blocks", "bubbles", "plane", "boomerang", "rod", "bicycle", "marker"}


@pytest.mark.parametrize("name", TOYS)
def test_agreed_he_plays_with_that_toy_within_seconds(name):
    life = fresh(1)
    c = life.choice
    c.acts[name].liking, c.acts[name].boredom = 0.95, 0.0      # нравится - согласится наверняка
    took, events = life.offer(name)
    assert took and any(e[0] == "phrase" for e in events)
    for _ in range(60 * 3):
        if life.toy is not None and life.toy.name == name and life.toy.active:
            break
        life.tick()
    assert life.toy is not None and life.toy.name == name and life.toy.active, (name, life.toy)
    assert c.current == name


@pytest.mark.parametrize("name", TOYS)
def test_refused_the_toy_lies_at_his_feet_and_is_remembered(name):
    life = fresh(2)
    c = life.choice
    c.acts[name].liking, c.acts[name].boredom = 0.1, 0.8        # надоела - откажется
    c.acts["explore"].liking = 0.9
    took, events = life.offer(name)
    assert not took and any(e[0] == "phrase" for e in events)
    x = life.inv.where[name]
    assert isinstance(x, float) and abs(x - life.human.px) < 1.0
    assert abs(life.walk.places.marks[name] - x) < 1e-6
    assert life.toy is None or life.toy.name != name


def test_a_fresh_idle_boy_takes_every_toy_at_least_half_the_time():
    rates = {}
    for name in TOYS:
        took = sum(bool(fresh(seed, walk=1.0).offer(name)[0]) for seed in range(20))
        rates[name] = took / 20
    assert min(rates.values()) >= 0.5, rates
    assert max(rates.values()) - min(rates.values()) <= 0.35, rates      # самолёт не особенный


def test_while_playing_with_one_toy_another_is_offered_and_the_choice_still_counts():
    life = fresh(3)
    c = life.choice
    c.acts["blocks"].liking, c.acts["blocks"].boredom = 0.95, 0.0
    assert life.offer("blocks")[0]
    life.tick()
    assert life.toy is not None and life.toy.name == "blocks"
    c.acts["plane"].liking, c.acts["plane"].boredom = 0.1, 0.8     # увлечён кубиками, самолёт не нужен
    took, _ = life.offer("plane")
    assert not took and life.toy is not None and life.toy.name == "blocks"
    c.acts["plane"].liking = 0.95
    c.acts["blocks"].liking, c.acts["blocks"].boredom = 0.2, 0.7   # кубики надоели - самолёт берёт
    took, _ = life.offer("plane")
    assert took
    for _ in range(60 * 3):
        life.tick()
        if life.toy is not None and life.toy.name == "plane":
            break
    assert life.toy is not None and life.toy.name == "plane" and life.toy.active


def test_absorbed_in_a_toy_he_mostly_keeps_it_idle_he_mostly_takes_the_new_one():
    """Занят игрушкой, которая тянет как прогулка, - новую обычно не берёт (держит гистерезис
    занятия); просто гуляет - берёт. Согласие - выбор, а не приказ и не случайность."""
    busy = idle = 0
    for seed in range(20):
        life = fresh(200 + seed, walk=0.5)
        c = life.choice
        c.acts["blocks"].liking, c.acts["blocks"].boredom = 0.95, 0.0
        assert life.offer("blocks")[0]
        c.acts["blocks"].liking = c.acts["explore"].liking * (1.0 - c.acts["explore"].boredom)
        busy += bool(life.offer("plane")[0])
        idle += bool(fresh(200 + seed, walk=0.5).offer("plane")[0])
    assert busy / 20 <= 0.3 and idle / 20 >= 0.5, (busy, idle)


def test_no_rod_offer_is_taken_without_water():
    dry = next(s for s in range(7100, 7400) if not Water(generate(M.world_genome, s, 160.0)).ponds)
    life = Life(M, dry, 7, 160.0, models.starter_brain(), seed=1)
    life.choice.acts["rod"].liking = 0.95
    res = life.offer("rod")
    assert res is not None and not res[0] and life.toy is None


def test_tools_are_not_offered():
    life = fresh(4, walk=0.2)
    for t in registry.TOYS.values():
        if t.tool:
            assert life.offer(t.name) is None
    assert np.isfinite(life.human.px)


def test_the_catalog_button_offers_the_plane_and_he_plays(qapp, tmp_path, monkeypatch):
    """Как у Алексея: «Каталог» -> «Предложить» у самолётика."""
    from game import app as app_module
    monkeypatch.setenv("AIBOY_HOME", str(tmp_path))
    w = app_module.MainWindow(autostart=False)
    try:
        w.show()
        w.observe.new_world(seed=3)
        life = w.observe.life
        for _ in range(60):
            life.tick()
        w.show_page("catalog")
        life.choice.acts["plane"].liking = 0.95          # без случайности: нравится - берёт
        assert w.catalog_page.offer("plane") is True
        assert life.toy is not None and life.toy.name == "plane"
        assert w.catalog_page.status.text().startswith("Взял")
    finally:
        w.close()


def test_a_ball_in_flight_when_he_switches_lands_and_stays_playable():
    life = fresh(5)
    c = life.choice
    c.acts["ball"].liking, c.acts["ball"].boredom = 0.95, 0.0
    assert life.offer("ball")[0]
    for _ in range(60 * 20):
        life.tick()
        if life.inv.where.get("ball") == "flying":
            break
    assert life.inv.where.get("ball") == "flying"
    c.acts["plane"].liking = 0.95
    c.acts["ball"].liking, c.acts["ball"].boredom = 0.1, 0.8
    assert life.offer("plane")[0]
    x = life.inv.where["ball"]
    assert isinstance(x, float) and abs(life.walk.places.marks["ball"] - x) < 1e-6   # лежит, где упал, и помнит
    assert "ball" in life.available()
