"""Человечек, идущий влево, - зеркало идущего вправо целиком: голова, руки, корпус, ноги и стопы
(Алексей, 2.0: «верх поворачивается, а ноги всегда смотрят вправо» - носки стоп)."""
import numpy as np
import pytest
from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QImage, QPainter

from body import skeleton
from body.cppn import generate_body
from body.physics import Human
from game import models
from game.view import Frame, draw_human, foot_angle

M = models.load()
SWAP = [skeleton.JOINTS.index(n.replace("_l", "_R").replace("_r", "_l").replace("_R", "_r"))
        for n in skeleton.JOINTS]
W = H = 900                                         # крупно: стопа - десятки точек


class Ground:
    width = 1e9

    def height_at(self, x):
        return 0.0


def render(human):
    img = QImage(W, H, QImage.Format.Format_ARGB32)
    img.fill(QColor(0, 0, 0))
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    draw_human(p, Frame(QRectF(0, 0, W, H), human.px, human.py), human, 0.0)
    p.end()
    a = np.frombuffer(img.constBits(), np.uint8).reshape(H, W, 4)[:, :, :3].astype(np.int16)
    return a


@pytest.mark.parametrize("body_seed", [7, 12, 21])
def test_facing_left_is_the_mirror_of_facing_right(qapp, body_seed):
    body = generate_body(M.body_genome, body_seed)
    rng = np.random.default_rng(body_seed)
    h = Human(body, Ground(), x=0.0)
    lo, hi = skeleton.LO, skeleton.HI
    for _ in range(4):                             # разные позы шага, не только стойка
        pose = lo + (hi - lo) * rng.uniform(0.25, 0.75, len(lo))
        h.angles, h.facing = pose, -1
        left = render(h)
        # повернулся - левая рука и нога стали дальними: зеркало правого с переставленными l и r
        h.angles, h.facing = pose[SWAP], 1
        right = render(h)
        mirrored = right[:, ::-1]
        drawn = (right.sum(axis=2) > 0).sum()
        assert drawn > 2000
        # ось зеркала - центр кадра, между пикселями: допускаем сдвиг на один пиксель
        bad = min(int((np.abs(left - m).sum(axis=2) > 90).sum()) for m in
                  (mirrored, np.roll(mirrored, 1, axis=1), np.roll(mirrored, -1, axis=1)))
        assert bad / drawn < 0.02, (body_seed, bad, drawn)

def test_the_toe_points_where_he_looks():
    for dx, dy in ((0.0, 1.0), (0.3, 1.0), (-0.3, 1.0)):     # голень вниз (в точках экрана y вниз)
        for facing in (1, -1):
            a = np.radians(foot_angle(dx, dy, facing))
            toe = np.cos(a)
            assert toe * facing > 0, (dx, dy, facing, toe)
        # зеркальная голень - зеркальная стопа
        r = np.radians(foot_angle(dx, dy, 1))
        l = np.radians(foot_angle(-dx, dy, -1))
        assert abs(np.cos(r) + np.cos(l)) < 1e-9 and abs(np.sin(r) - np.sin(l)) < 1e-9
