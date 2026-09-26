"""«Что видит»: подсказка строится из того же наблюдения, что получил мозг, включается и
выключается, влезает в кадр и не закрывает чат."""
import json

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtTest import QTest

import paths
from brain import observe as ob
from game import app as app_module
from game.perception import perceive


@pytest.fixture
def win(qapp):
    w = app_module.MainWindow(autostart=False)
    w.show()
    w.show_page("observe")
    yield w
    w.close()


def test_vision_is_on_by_default_and_saved(win):
    assert win.settings.show_vision is True and win.observe.view.show_vision
    QTest.keyClick(win, Qt.Key.Key_V)
    assert not win.observe.view.show_vision and not win.observe.vision_button.isChecked()
    assert json.loads(open(paths.user_file("settings.json"), encoding="utf-8").read())["show_vision"] is False
    QTest.mouseClick(win.observe.vision_button, Qt.MouseButton.LeftButton)
    assert win.observe.view.show_vision and win.settings.show_vision


@pytest.mark.parametrize("facing", [1, -1])
def test_overlay_is_built_from_the_brains_own_observation(win, facing):
    obs = win.observe
    obs.advance(90)
    life = obs.life
    life.human.facing = facing                   # глазами человечка - в обе стороны
    life.human.turn_cd = 1.0
    life.tick()
    life.tick()
    per = perceive(life)
    o = life.last_obs
    for (x, y, _), off, v in zip(per.rays, ob.OFFSETS, o[ob.TERRAIN]):
        assert x == pytest.approx(life.obs_x + life.human.facing * off)
        assert (y - life.obs_ground) / 2.0 == pytest.approx(float(v), abs=1e-6)
    seen = int(o[ob.ENTITIES.start + 2] > 0.5) + int(o[ob.ENTITIES.start + ob.ENTITY + 2] > 0.5)
    assert len(per.entities) == seen
    for x, _, _, _ in per.entities:
        dx = (x - life.obs_x) * life.human.facing / ob.SEE
        assert any(abs(dx - o[ob.ENTITIES.start + s * ob.ENTITY]) < 1e-5 for s in range(2))
    assert per.lines[0].startswith("видит: ") and per.lines[1].startswith("хочет: ")
    assert per.lines[2].startswith("прыжок: ")


def test_interest_follows_the_predictor_error(win):
    obs = win.observe
    obs.advance(60)
    life = obs.life
    errs = np.zeros(ob.OBS_DIM)
    errs[3] = 50.0
    life.brain.last_err_vec = errs
    per = perceive(life)
    heats = [h for _, _, h in per.rays]
    assert int(np.argmax(heats)) == 3 and heats[3] > 0.5 and max(heats[:3] + heats[4:]) == 0.0


def test_intent_is_the_last_second_of_wishes(win):
    from game.life import INTENT_K
    life = win.observe.life
    win.observe.advance(2)
    life.intent = 0.0
    life.tick()
    life.tick()                           # одно решение мозга
    assert life.intent == pytest.approx(INTENT_K * life.decision.turn)
    assert perceive(life).intent == life.intent


def rect_of(widget, win):
    return QRect(widget.mapTo(win, QPoint(0, 0)), widget.size())


@pytest.mark.parametrize("minimal", [False, True])
def test_overlay_fits_the_frame_and_never_covers_the_chat(win, minimal):
    if minimal:
        win.resize(win.minimumSize())
    obs = win.observe
    obs.advance(120)
    QTest.qWait(20)
    obs.view.repaint()
    box = obs.view.vision_rect
    assert box is not None
    view = QRect(0, 0, obs.view.width(), obs.view.height())
    assert view.contains(box.toRect())
    on_window = box.toRect().translated(obs.view.mapTo(win, QPoint(0, 0)))
    assert not on_window.intersects(rect_of(obs.chat, win))
    win.change_setting("show_vision", False)
    obs.view.repaint()
    assert obs.view.vision_rect is None


def test_vision_setting_survives_old_files(tmp_path):
    from game.settings import Settings
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"show_vision": "да"}), encoding="utf-8")
    assert Settings.load(path).show_vision is True


def test_thought_arrow_agrees_with_the_intent_arrow(win):
    from game.view import thoughts_of
    life = win.observe.life
    win.observe.advance(20)
    life.intent = 0.6
    life.decision.turn = -0.9                  # одно решение мелькнуло в другую сторону
    assert thoughts_of(life.decision, 0.5, life.intent)[0][0] == "иду →"
    life.intent = -0.4
    assert thoughts_of(life.decision, 0.5, life.intent)[0][0] == "← иду"
