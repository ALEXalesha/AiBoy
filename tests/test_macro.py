"""Большие миры 640 и 1280 м: макро-рельеф без видимого повтора; прежние размеры - те же байты.

Замер до (сеть мира видит волны 80 м и короче): повтор рельефа на сдвигах 80-640 м - 0.82-0.92
в мирах 640 м и 0.94-0.97 в 1280 м (1 - один в один). С макро-сетью - 0.32 в медиане.
"""
import hashlib

import numpy as np
import pytest

from game import models
from world import macro, reach
from world.generate import generate

# хэши миров 1.1.0 (посчитаны кодом тега v1.1.0 на тех же зёрнах)
OLD = {(80, 1): 'acba288dde5dbaa4', (80, 7): 'e2019a06a04ab1cd', (80, 2026): '4eba9bc39bdf5e53',
       (80, 123456): '0ff39b7b7be8f2f2', (160, 1): '11dc62d03304c682', (160, 7): 'ec621eb259f51e45',
       (160, 2026): '2444f38881cce16c', (160, 123456): 'bac8e573a7a81e50', (320, 1): 'de2c90a48508f26d',
       (320, 7): 'a1be2d704989e754', (320, 2026): 'a83bdcb2e075528b', (320, 123456): 'f84231ffc84d63e6'}
# Пороги повтора. Форма холмов (world/macro.shape_repeat_score по холмам без макро-высоты;
# размах, меняясь, не прячет одинаковую форму) - то, что видно глазом в окне в 20-30 м. Замер
# на 20 зёрнах после эволюции: средняя 0.28, худшая 0.40; без изгиба координаты - 0.90;
# в мирах 160 и 320 м (холмы по 80 м) - 0.73. Весь рельеф (с хребтами и котловинами по
# 130-1280 м): средний 0.29, худший 0.69 - это два похожих хребта за 320 м друг от друга, в
# кадр они вместе не попадают.
SHAPE = 0.6
WHOLE = 0.8


@pytest.fixture(scope="module")
def m():
    return models.load()


def digest(wd):
    h = hashlib.sha256()
    h.update(np.asarray(wd.heights, np.float64).tobytes())
    h.update(repr([(e.kind, e.x, e.y, e.size, e.color) for e in wd.entities]).encode())
    h.update(repr((wd.name, wd.palette, wd.night)).encode())
    return h.hexdigest()[:16]


@pytest.mark.parametrize("key", sorted(OLD))
def test_the_old_sizes_make_the_same_worlds_byte_for_byte(m, key):
    assert digest(generate(m.world_genome, key[1], key[0])) == OLD[key]


@pytest.mark.parametrize("width", [640, 1280])
def test_big_worlds_do_not_visibly_repeat_and_have_no_traps(m, width):
    shape = []
    for seed in range(40, 50):
        w = generate(m.world_genome, seed, width)
        # повтор - по рельефу без гор (одна гора 60-100 м - не повтор, а автокорреляцию её
        # склон поднимает до 0.9 на любом сдвиге; горы - свои законы, test_mountains.py)
        assert macro.repeat_score(w.relief, w.dx) < WHOLE, seed
        shape.append(macro.shape_repeat_score(w.hills, w.dx))
        assert shape[-1] < SHAPE, seed
        assert not reach.traps(w)
        # хребты и котловины: перепад высот больше, чем даёт сеть холмов одна (до 12 м)
        assert np.ptp(w.relief) > 8.0, seed
    assert np.median(shape) < 0.4                      # узор холмов не повторяется и в среднем


def test_without_macro_relief_a_big_world_would_repeat(m):
    """Сам замер «повтора» честный: без макро-сети мир 1280 м повторяется."""
    from world import cppn
    from world.generate import _smooth_ring, sig
    z = cppn.seed_vector(41)
    xs = np.arange(int(1280 / 0.25)) * 0.25
    out = cppn.terrain(m.world_genome, xs, 1280, z)
    h = _smooth_ring(np.tanh(out[:, cppn.T_HEIGHT]), 2.0) * (1.2 + 3.3 * sig(0.0))
    assert macro.repeat_score(h, 0.25) > 0.8 and macro.shape_repeat_score(h, 0.25) > 0.8


def test_the_macro_net_was_evolved(m):
    p = models.passport("macro")
    assert p.get("generations", 0) >= 20 and p["check_20_new_seeds"]["evolved"]["worlds_with_traps"] == 0


def test_big_worlds_get_a_ring_map_and_a_whole_ring_in_the_gallery_picture(qapp):
    from game import app as app_module
    from game.view import thumbnail
    w = app_module.MainWindow(autostart=False)
    try:
        w.show()
        w.show_page("observe")
        w.observe.start(5, 7, 160.0)
        assert w.observe.ring.isHidden()
        w.observe.start(5, 7, 1280.0)
        assert not w.observe.ring.isHidden()
        life = w.observe.life
        m = models.load()
        colors = w.colors()
        big = thumbnail(life.world, life.body, 300, 180, colors)
        small_world = generate(m.world_genome, 5, 160.0)
        small = thumbnail(small_world, life.body, 300, 180, colors)
        # в большом - полоса кольца внизу: в её верхнем ряду есть цвет фона полосы
        bg = colors["surface2"].lower()
        row = lambda img: {QColor_at(img, x, 146) for x in range(0, 300, 3)}  # noqa: E731
        assert bg in row(big) and bg not in row(small)
    finally:
        w.close()


def QColor_at(image, x, y):
    return image.pixelColor(x, y).name()
