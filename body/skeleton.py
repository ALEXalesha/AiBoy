"""Скелет человечка: 15 точек по 10 углам суставов.

Вид сбоку, человечек смотрит вправо (facing=1) или влево (-1, зеркало по x), y вверх,
начало - таз. Углы (рад):

- lean - наклон корпуса, вперёд положительный; neck - голова относительно корпуса;
- sh - плечо: 0 - рука висит, π/2 - вытянута вперёд, π - вверх; el - локоть сгибает
  предплечье вперёд;
- hip - бедро: 0 - нога вниз, вперёд положительный; kn - колено сгибает голень назад.

Считается на math, не numpy: 15 точек, и так в десятки раз быстрее.
"""
import math

import numpy as np

JOINTS = ("lean", "neck", "sh_l", "sh_r", "el_l", "el_r", "hip_l", "hip_r", "kn_l", "kn_r")
JOINT_LIMITS = ((-0.35, 0.6), (-0.5, 0.6), (-1.2, 3.0), (-1.2, 3.0), (0.0, 2.4), (0.0, 2.4),
                (-0.8, 1.6), (-0.8, 1.6), (0.0, 2.3), (0.0, 2.3))
POINTS = ("head", "neck", "pelvis", "sh_l", "sh_r", "el_l", "el_r", "ha_l", "ha_r",
          "hip_l", "hip_r", "kn_l", "kn_r", "ft_l", "ft_r")
LO = np.array([lo for lo, _ in JOINT_LIMITS])
HI = np.array([hi for _, hi in JOINT_LIMITS])
REST = (0.04, 0.0, 0.12, 0.08, 0.35, 0.3, 0.05, 0.0, 0.1, 0.12)


def rest_angles():
    return np.array(REST, float)


def clip(angles):
    return np.clip(angles, LO, HI)


def random_angles(rng):
    return LO + (HI - LO) * rng.random(len(JOINTS))


def from_unit(u):
    """Выход сети в [-1, 1] -> угол в пределах сустава."""
    return LO + (HI - LO) * (np.clip(u, -1, 1) + 1) * 0.5


def to_unit(angles):
    return (np.asarray(angles) - LO) / (HI - LO) * 2 - 1


def _down(a):
    """Направление по углу от «вниз», вперёд положительный."""
    return math.sin(a), -math.cos(a)


def points(body, angles, facing=1, origin=(0.0, 0.0), tilt=0.0):
    """Точки скелета. origin - где таз; tilt - поворот всего тела вокруг таза
    (падение), положительный - назад."""
    lean, neck, sh_l, sh_r, el_l, el_r, hip_l, hip_r, kn_l, kn_r = (float(a) for a in angles)
    ux, uy = math.sin(lean), math.cos(lean)
    p = {"pelvis": (0.0, 0.0), "hip_l": (0.0, 0.0), "hip_r": (0.0, 0.0)}
    p["neck"] = (body.torso * ux, body.torso * uy)
    sh = (body.torso - 0.045) * ux, (body.torso - 0.045) * uy
    p["sh_l"] = p["sh_r"] = sh
    hd = body.neck + body.head_r
    p["head"] = (p["neck"][0] + hd * math.sin(lean + neck), p["neck"][1] + hd * math.cos(lean + neck))
    for side, s, e in (("l", sh_l, el_l), ("r", sh_r, el_r)):
        au = s + lean
        dx, dy = _down(au)
        el = (sh[0] + body.upper_arm * dx, sh[1] + body.upper_arm * dy)
        dx, dy = _down(au + e)
        p["el_" + side] = el
        p["ha_" + side] = (el[0] + body.forearm * dx, el[1] + body.forearm * dy)
    for side, h, k in (("l", hip_l, kn_l), ("r", hip_r, kn_r)):
        dx, dy = _down(h)
        kn = (body.thigh * dx, body.thigh * dy)
        dx, dy = _down(h - k)
        p["kn_" + side] = kn
        p["ft_" + side] = (kn[0] + body.shin * dx, kn[1] + body.shin * dy)
    ox, oy = origin
    if tilt:
        c, s = math.cos(tilt), math.sin(tilt)
        # назад - против взгляда
        return {n: (ox + facing * (x * c - y * s), oy + x * s + y * c) for n, (x, y) in p.items()}
    return {n: (ox + facing * x, oy + y) for n, (x, y) in p.items()}


def feet(body, angles, facing=1):
    """Только стопы (относительно таза) - для физики, без остальных точек."""
    out = []
    for h, k in ((angles[6], angles[8]), (angles[7], angles[9])):
        kx, ky = body.thigh * math.sin(h), -body.thigh * math.cos(h)
        fx, fy = kx + body.shin * math.sin(h - k), ky - body.shin * math.cos(h - k)
        out.append((facing * fx, fy))
    return out
