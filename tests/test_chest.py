"""Сундук и где игрушки; реестр; перенос сохранений 1.1.0 (гитара за спиной -> в сундук)."""
import json

import numpy as np


import paths
from game import app as app_module
from toys import registry
from toys.chest import REACH, Inventory


def test_everything_starts_in_the_chest_and_one_toy_at_a_time_is_carried():
    inv = Inventory(10.0, 80.0, toys=["guitar", "мяч"])
    assert inv.carried is None and inv.position("guitar") == 10.0
    assert not inv.take("guitar", 10.0 + REACH + 0.1)          # не дотянулся
    assert inv.take("guitar", 10.3) and inv.carried == "guitar"
    assert inv.position("guitar") is None
    assert not inv.take("мяч", 10.0)                          # руки заняты
    assert inv.put("guitar", 30.0) and inv.position("guitar") == 30.0
    assert inv.take("мяч", 10.0) and inv.on_ground() == {"guitar": 30.0}
    inv.stow("guitar")
    assert inv.where["guitar"] == "chest"


def test_the_ring_has_no_seam_for_reaching():
    inv = Inventory(79.8, 80.0, toys=["guitar"])
    assert inv.take("guitar", 0.3)


def test_the_guitar_is_the_first_toy_of_the_registry():
    g = registry.get("guitar")
    assert registry.names()[0] == "guitar" and g.carry == "back" and g.title == "Гитара"


def test_a_1_1_0_save_opens_the_guitar_goes_into_the_chest_the_music_stays(qapp):
    """Сохранение 1.1.0: гитара у него за спиной, журнал с попыткой, музыкант."""
    w = app_module.MainWindow(autostart=False)
    try:
        life = w.observe.life
        seed = life.body_seed
        g = life.guitarist
        g.journal.add({"tune": "Twinkle Twinkle", "phrase": 0, "notes": [], "A": 0.4, "B": 0.1, "C": None,
                       "taste": None, "total": 0.5, "rating": None})
        g.journal.meta["has_guitar"] = True                     # как писала 1.1.0
        g.musician.attempts = 17
        w.observe.save_guitar()
    finally:
        w.close()
    path = paths.user_file("guitar", str(seed), "journal.json")
    raw = json.load(open(path, encoding="utf-8"))
    raw.setdefault("meta", {})["has_guitar"] = True
    json.dump(raw, open(path, "w", encoding="utf-8"))
    again = app_module.MainWindow(autostart=False)
    try:
        life = again.observe.life
        assert life.body_seed == seed and life.inv.where["guitar"] == "chest" and not life.has_guitar
        assert len(life.guitarist.journal.entries) == 1 and life.guitarist.musician.attempts == 17
    finally:
        again.close()


def test_broken_guitar_files_mean_a_fresh_start_not_a_crash(qapp):
    w = app_module.MainWindow(autostart=False)
    seed = w.observe.life.body_seed
    w.close()
    for name in ("journal.json", "musician.npz"):
        open(paths.user_file("guitar", str(seed), name), "wb").write(b"\x00garbage")
    open(paths.user_file("skills", str(seed), "reach.npz"), "wb").write(b"garbage")
    again = app_module.MainWindow(autostart=False)
    try:
        g = again.observe.life.guitarist
        assert g.journal.entries == [] and g.musician.attempts == 0 and np.isfinite(g.musician.params()[0][1]).all()
    finally:
        again.close()



# --- «Каталог» ---

def bored_of_guitar(life, keen):
    c = life.choice
    c.acts["guitar"].liking, c.acts["guitar"].boredom = (0.95, 0.0) if keen else (0.1, 0.8)
    c.acts["explore"].liking = 0.3 if keen else 0.9


def test_an_offer_is_taken_more_often_when_he_likes_the_toy(qapp):
    from game.life import Life
    from game import models
    from music.journal import Journal
    from music.musician import Musician
    from music.session import Guitarist
    from music.songbook import Songbook
    import tempfile
    m = models.load()
    rates = {}
    for keen in (True, False):
        took = 0
        for i in range(60):
            d = tempfile.mkdtemp()
            g = Guitarist(Musician(seed=1), Journal(d + "/j.json"), Songbook(d + "/s.json"))
            life = Life(m, 3, 4, 80.0, models.starter_brain(), seed=i, guitarist=g)
            bored_of_guitar(life, keen)
            ok, events = life.offer("guitar")
            said = [e for e in events if e[0] == "phrase"]
            assert said
            if ok:
                took += 1
                assert life.playing and life.has_guitar
            else:
                assert not life.playing and life.guitar_x is not None      # лежит рядом
                assert abs(life.guitar_x - life.human.px) < 1.0
        rates[keen] = took / 60
    assert rates[True] > 0.8 and rates[False] < 0.3, rates


def test_catalog_shows_the_toys_and_offers(qapp):
    w = app_module.MainWindow(autostart=False)
    try:
        w.show()
        w.show_page("catalog")
        page = w.catalog_page
        assert set(page.cards) >= {"guitar", "explore"}
        card = page.cards["guitar"]
        assert card.offer.isEnabled() and "в сундуке" in card.numbers.text()
        chat = len(w.observe.life.phrases)
        page.offer("guitar")
        assert len(w.observe.life.phrases) == chat + 1 and page.status.text()
        page.refresh()
        assert 0 <= card.liking.value() <= 100
        w.show_page("teach")
        assert w.tabs["catalog"].isChecked()
    finally:
        w.close()
