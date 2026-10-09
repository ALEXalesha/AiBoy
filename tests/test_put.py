"""«Положи» - для каждой вещи, не только для гитары, и кнопка - по настоящему положению вещей
(Алексей, 2.0: «Положи гитару» висит, когда гитара уже лежит и он сидит рядом и отдыхает;
кнопка «Положить» нужна у каждой вещи, пока он её держит)."""
import pytest

from game import app as app_module
from game import models
from game.life import Life
from toys import registry
from world.generate import generate
from world.water import Water

M = models.load()
WET = next(s for s in range(7100, 7400) if Water(generate(M.world_genome, s, 160.0)).ponds)
TOYS = [t.name for t in registry.TOYS.values() if t.name != "guitar" and not t.tool]
ITEMS = [n for n in registry.TOYS if n != "guitar"]


def fresh(seed):
    life = Life(M, WET, 7, 160.0, models.starter_brain(), seed=seed)
    for _ in range(30):
        life.tick()
    return life


def hold(life, name):
    held = life.inv.carried
    if held is not None:
        life.put_held()
    life.inv.where[name] = life.human.px
    assert life.inv.take(name, life.human.px)


def test_every_item_has_a_name_for_the_button():
    for name in registry.TOYS:
        assert registry.accusative(name)
    assert registry.accusative("guitar") == "гитару" and registry.accusative("rod") == "удочку"


@pytest.mark.parametrize("name", TOYS)
def test_asked_to_put_a_toy_he_is_bored_with_he_puts_it_at_his_feet(name):
    agreed = 0
    for seed in range(10):
        life = fresh(seed)
        c = life.choice
        c.acts[name].liking, c.acts[name].boredom = 0.95, 0.0
        assert life.offer(name)[0] and life.toy is not None and life.toy.name == name
        c.acts[name].liking, c.acts[name].boredom = 0.1, 0.8
        c.acts["explore"].liking = 0.9
        ok, events = life.ask_put(name)
        assert any(e[0] == "phrase" for e in events)
        if ok:
            agreed += 1
            x = life.inv.where[name]
            assert isinstance(x, float) and abs(x - life.human.px) < 1.0
            assert abs(life.walk.places.marks[name] - x) < 1e-6
            assert life.toy is None or life.toy.name != name     # бросил игру
        else:
            assert life.inv.where[name] == "carried"
    assert agreed >= 8, (name, agreed)


@pytest.mark.parametrize("name", TOYS)
def test_absorbed_in_a_toy_he_mostly_keeps_it(name):
    agreed = 0
    for seed in range(10):
        life = fresh(seed)
        c = life.choice
        c.acts[name].liking, c.acts[name].boredom = 0.95, 0.0
        c.acts["explore"].liking = 0.3
        assert life.offer(name)[0]
        ok, _ = life.ask_put(name)
        agreed += bool(ok)
        if not ok:
            assert life.toy is not None and life.toy.name == name and life.toy.active   # играет дальше
    assert agreed <= 3, (name, agreed)


@pytest.mark.parametrize("name", TOYS[:3])
def test_while_playing_he_gives_it_up_less_than_when_just_holding_it(name):
    """Увлечённость (STICK): посреди игры положить вещь его уговорить труднее, чем когда он
    её просто держит, при той же тяге к ней."""
    def rate(playing):
        agreed = 0
        for seed in range(40):
            life = fresh(seed)
            c = life.choice
            c.acts[name].liking, c.acts[name].boredom = 0.6, 0.0
            c.acts["explore"].liking = 0.6
            if playing:
                assert life.offer(name)[0] and life.toy is not None and life.toy.name == name
            else:
                hold(life, name)
                assert life.toy is None or life.toy.name != name
            agreed += bool(life.ask_put(name)[0])
        return agreed
    busy, idle = rate(True), rate(False)
    assert busy + 6 <= idle, (name, busy, idle)


def test_nothing_to_put_when_he_holds_nothing_or_another_item():
    life = fresh(1)
    if life.inv.carried is not None:
        life.put_held()
    assert life.ask_put("plane") is None
    hold(life, "ball")
    assert life.ask_put("plane") is None and life.ask_put() is not None


@pytest.fixture
def win(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("AIBOY_HOME", str(tmp_path))
    w = app_module.MainWindow(autostart=False)
    w.show()
    w.observe.new_world(seed=3)
    w.show_page("observe")
    yield w
    w.close()


def test_the_guitar_button_hides_once_the_guitar_is_put_down_any_way(win):
    obs = win.observe
    life = obs.life
    hold(life, "guitar")
    obs.advance(1)
    assert obs.put_button.isVisibleTo(obs) and "гитару" in obs.put_button.text()
    life.put_held()                     # положил не по просьбе (идёт за едой, к собаке, отдыхать)
    obs.advance(1)
    assert not obs.put_button.isVisibleTo(obs)
    assert not obs.guitar_panel.put.isEnabled()


@pytest.mark.parametrize("name", ITEMS)
def test_every_item_he_holds_gets_its_own_put_button(win, name):
    obs = win.observe
    life = obs.life
    hold(life, name)
    obs.advance(1)
    assert obs.put_button.isVisibleTo(obs), name
    assert obs.put_button.text() == "Положи " + registry.accusative(name)
    life.put_held()
    obs.advance(1)
    assert not obs.put_button.isVisibleTo(obs)


def test_the_put_button_asks_and_the_answer_goes_to_the_chat(win):
    obs = win.observe
    life = obs.life
    hold(life, "plane")
    life.choice.acts["plane"].liking, life.choice.acts["plane"].boredom = 0.05, 0.9
    life.choice.acts["explore"].liking = 0.95
    chat = len(life.phrases)
    obs.advance(1)
    agree = obs.ask_put()
    assert agree is True and isinstance(life.inv.where["plane"], float)
    assert len(life.phrases) == chat + 1
    obs.advance(1)
    assert not obs.put_button.isVisibleTo(obs)
