"""Смена мира во время гитарной попытки: старая жизнь останавливается целиком.

Замер до правки (1.1.0): старая жизнь после «Новый мир» уже не шагала и собиралась
сборщиком мусора, но звук её попытки был отдан микшеру целиком при старте и доигрывал до
конца - до 9 с после смены мира; то же при паузе и в меню (жизнь стоит, а гитара играет) и
на x8 «без звука» (жизнь ушла вперёд, звук - в x1). Законы ниже: звук идёт за жизнью.
"""
import gc
import weakref

import numpy as np
import pytest

import paths
from audio_fake import SR, attach, frame, tail_after
from game import app as app_module
from game import audio
from music.journal import Journal
from music.musician import Musician


@pytest.fixture
def win(qapp):
    w = app_module.MainWindow(autostart=False)
    w.show()
    w.show_page("observe")
    yield w
    w.close()


def bore(win):
    life = win.observe.life
    life.has_guitar = True
    life.mood.world, life.mood.music, life.mood.world_time = 0.0, 0.9, 60.0


def into_attempt(win, dev, at=1.0, limit=60 * 60):
    """Кадры, пока не идёт попытка и в ней не прошло at секунд."""
    for _ in range(limit):
        frame(win, dev)
        life = win.observe.life
        if life.attempt is not None and life.play_t >= at:
            return life
    raise AssertionError("он так и не заиграл")


def finished_attempts(win, dev, n):
    life = win.observe.life
    for _ in range(60 * 60 * 3):
        frame(win, dev)
        if len(life.guitarist.journal.entries) >= n:
            return
    raise AssertionError("мало попыток")


SWITCHES = {
    "new_world": lambda w: w.observe.new_world(),
    "new_human": lambda w: w.observe.new_human(),
    "gallery": lambda w: w.gallery_page.open_entry(w.gallery.entries[-1]["id"]),
    "menu_new": lambda w: w.play_new(),
}


def setup_case(win, speed=1, mode="musician", fast="slow"):
    dev = attach(win)
    win.change_setting("guitar_mode", mode)
    win.change_setting("guitar_fast", fast)
    win.observe.set_speed(speed)
    bore(win)
    return dev


CASES = [("new_world", 1, "musician", "slow"), ("new_human", 1, "musician", "slow"),
         ("gallery", 1, "musician", "slow"), ("menu_new", 1, "musician", "slow"),
         ("new_world", 8, "musician", "slow"), ("new_world", 8, "musician", "silent"),
         ("new_world", 1, "hands", "slow")]


@pytest.mark.parametrize("how,speed,mode,fast", CASES)
def test_after_a_world_change_the_old_session_does_not_take_a_single_step(win, how, speed, mode, fast):
    dev = setup_case(win, speed, mode, fast)
    old = into_attempt(win, dev)
    g = old.guitarist
    before = (old.time, old.play_t, g.musician.attempts, len(g.journal.entries), old.human.px)
    SWITCHES[how](win)
    for _ in range(60 * 5):
        frame(win, dev)
    # и даже если кто-то ещё держит старую жизнь и дёрнет её - она не шагает
    assert old.tick() == []
    assert (old.time, old.play_t, g.musician.attempts, len(g.journal.entries), old.human.px) == before
    assert old.attempt is None and not old.playing


@pytest.mark.parametrize("how,speed,mode,fast", CASES)
def test_old_attempt_sound_fades_out_within_100_ms_without_a_click(win, how, speed, mode, fast):
    dev = setup_case(win, speed, mode, fast)
    into_attempt(win, dev)
    at = sum(len(p) for p in dev.played)
    SWITCHES[how](win)
    # затухание начинается в самой смене мира, а не на следующем кадре
    assert len(win.audio.channels.get("guitar", ())) <= int(audio.FADE * audio.SR)
    for _ in range(60 * 3):
        frame(win, dev)
    seconds, jump, usual = tail_after(dev.stream(), at)
    assert seconds <= 0.1, f"старая попытка звучала ещё {seconds:.2f} с"
    assert jump <= max(1.5 * usual, 0.01), (jump, usual)


@pytest.mark.parametrize("how", ["new_world", "new_human"])
def test_old_journal_and_weights_are_saved_and_nothing_old_gets_into_the_new(win, how):
    dev = setup_case(win)
    finished_attempts(win, dev, 2)
    old = into_attempt(win, dev)
    g = old.guitarist
    ids = [e["id"] for e in g.journal.entries]
    weights = [p.copy() for _, p, _ in g.musician.params()]
    attempts = g.musician.attempts
    seed = old.body_seed
    SWITCHES[how](win)
    for _ in range(60 * 2):
        frame(win, dev)
    # старый человечек: всё законченное - на диске, идущая попытка брошена без следа
    disk = Journal.load(paths.user_file("guitar", str(seed), "journal.json"))
    assert [e["id"] for e in disk.entries] == ids
    saved = Musician(seed=0)
    saved.load(paths.user_file("guitar", str(seed), "musician.npz"))
    assert saved.attempts == attempts
    assert all(np.array_equal(a[1], b) for a, b in zip(saved.params(), weights))
    new = win.observe.life.guitarist
    if how == "new_human":
        assert new.body_seed != seed and new.journal.entries == [] and new.musician.attempts == 0
    else:
        assert [e["id"] for e in new.journal.entries] == ids and new.musician.attempts == attempts


@pytest.mark.parametrize("how", ["new_world", "new_human", "gallery"])
def test_old_life_is_garbage_collected(win, how):
    dev = setup_case(win)
    ref = weakref.ref(into_attempt(win, dev))
    SWITCHES[how](win)
    for _ in range(30):
        frame(win, dev)
    gc.collect()
    assert ref() is None


@pytest.mark.parametrize("action,undo", [
    (lambda w: w.observe.toggle_pause(), lambda w: w.observe.toggle_pause()),
    (lambda w: w.show_page("menu"), lambda w: w.show_page("observe")),
])
def test_pause_and_menu_hush_the_guitar_and_it_goes_on_from_the_same_place(win, action, undo):
    dev = setup_case(win)
    life = into_attempt(win, dev, at=1.0)
    a = life.attempt
    at = sum(len(p) for p in dev.played)
    action(win)
    for _ in range(60 * 2):
        frame(win, dev)
    seconds, jump, usual = tail_after(dev.stream(), at)
    assert seconds <= 0.1 and jump <= max(1.5 * usual, 0.01)
    t_paused = life.play_t
    undo(win)
    k = len(dev.played)
    for _ in range(60):
        frame(win, dev)
    assert life.attempt is a and life.play_t > t_paused
    assert win.audio.history[-1].startswith("попытка") and f"с {t_paused:.1f} с" in win.audio.history[-1]
    # звучит ровно то, что попытка играет с этого места (после мягкого нарастания)
    want = win.bank.render(a.notes(), a.duration)[int(t_paused * SR):]
    got = dev.stream(k)
    skip, n = int(0.03 * SR), int(0.5 * SR)
    assert np.abs(got[skip:n] - want[skip:n]).max() < 0.01


def test_x8_without_sound_hushes_an_attempt_already_sounding(win):
    dev = setup_case(win)
    into_attempt(win, dev)
    at = sum(len(p) for p in dev.played)
    win.change_setting("guitar_fast", "silent")
    win.observe.set_speed(8)
    for _ in range(60 * 3):
        frame(win, dev)
    seconds, jump, usual = tail_after(dev.stream(), at)
    assert seconds <= 0.1 and jump <= max(1.5 * usual, 0.01)


def test_closing_the_window_during_an_attempt_saves_and_hushes(qapp):
    w = app_module.MainWindow(autostart=False)
    w.show()
    w.show_page("observe")
    dev = setup_case(w)
    finished_attempts(w, dev, 1)
    old = into_attempt(w, dev)
    ids = [e["id"] for e in old.guitarist.journal.entries]
    seed = old.body_seed
    w.close()
    assert old.tick() == [] and w.audio.sink is None
    disk = Journal.load(paths.user_file("guitar", str(seed), "journal.json"))
    assert [e["id"] for e in disk.entries] == ids
