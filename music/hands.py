"""Гитара в руках: где на ней лады и струны и куда тянуть руки.

Человечек сидит; гитара - в системе его тела (смотрит вправо, y вверх, начало - таз),
размеры - по его росту. Гитара мультяшно маленькая (мензура 36 см при росте 1.7 м): замер -
у человечка с самыми короткими руками (0.47 м) и ростом 1.95 первый лад настоящей гитары был
на 13 см дальше, чем достаёт рука, а у длиннорукого правая кисть не могла согнуться к
розетке. Перебором мест и углов (40 человечков и два крайних) взяты: корпус на коленях,
гриф под 50 градусов, 40 см до порожка - все лады и струны достижимы. Гриф уходит вперёд и вверх под углом NECK_ANGLE; левая рука - на
грифе (лад), правая - у розетки (струна). Лады - как у настоящей гитары: k-й лад на
расстоянии L·(1 - 2^(-k/12)) от порожка, чем ближе к корпусу, тем теснее.

В режиме «Музыкант» руки только показывают выбранные ноты: углы плеча и локтя до точки
считаются здесь (двухзвенная рука, теорема косинусов) - это рисование, а не игра. В
режиме «Руками» руки ведёт своя голова мозга (brain/reach.py), а здесь только проверка,
попала ли кисть в точку.
"""
import math

from body import skeleton
from music import guitar

NECK_ANGLE = math.radians(50.0)
SCALE_M = 0.36                  # мензура (порожек - подставка) при росте 1.7 м
BODY_AT = (0.18, 0.14)          # центр корпуса гитары от таза (на коленях), м при росте 1.7
NUT_FROM_BODY = 0.40            # от центра корпуса до порожка
STRING_GAP = 0.009              # м между струнами поперёк грифа
FRET_HIT = 0.6                  # доля расстояния между ладами - попал в лад
STRING_HIT = 1.0                # доля расстояния между струнами - попал в струну
SITTING = (0.12, 0.05, 1.3, 0.6, 1.1, 1.3, 1.45, 1.45, 0.7, 0.7)   # суставы сидя с гитарой


def _k(body):
    return body.height() / 1.7


def axis():
    return math.cos(NECK_ANGLE), math.sin(NECK_ANGLE)


def body_center(body):
    k = _k(body)
    return BODY_AT[0] * k, BODY_AT[1] * k


def nut(body):
    k = _k(body)
    cx, cy = body_center(body)
    ax, ay = axis()
    return cx + NUT_FROM_BODY * k * ax, cy + NUT_FROM_BODY * k * ay


def across(s, body):
    """Сдвиг струны поперёк грифа (перпендикуляр к оси, в плоскости картинки)."""
    ax, ay = axis()
    off = (s - (guitar.STRINGS - 1) / 2) * STRING_GAP * _k(body)
    return -ay * off, ax * off


def fret_distance(fret, body):
    """От порожка к корпусу, м."""
    return SCALE_M * _k(body) * (1.0 - 2.0 ** (-fret / 12.0))


def left_target(s, fret, body):
    """Куда поставить палец левой руки: посередине между ладом fret-1 и fret на струне s.
    Открытая струна - палец отдыхает у третьего лада."""
    f = fret if fret > 0 else 3
    d = 0.5 * (fret_distance(f - 1, body) + fret_distance(f, body))
    nx, ny = nut(body)
    ax, ay = axis()
    ox, oy = across(s, body)
    return nx - d * ax + ox, ny - d * ay + oy


def right_target(s, body):
    """Правая рука - у розетки, над струной s."""
    cx, cy = body_center(body)
    ax, ay = axis()
    ox, oy = across(s, body)
    return cx + 0.03 * _k(body) * ax + ox, cy + 0.03 * _k(body) * ay + oy


def fret_spacing(fret, body):
    f = max(fret, 1)
    return fret_distance(f, body) - fret_distance(f - 1, body)


def shoulder(body, lean):
    return (body.torso - 0.045) * math.sin(lean), (body.torso - 0.045) * math.cos(lean)


def hand(body, lean, sh, el):
    """Кисть по углам плеча и локтя (как в body/skeleton.py)."""
    sx, sy = shoulder(body, lean)
    au = sh + lean
    ex, ey = sx + body.upper_arm * math.sin(au), sy - body.upper_arm * math.cos(au)
    return ex + body.forearm * math.sin(au + el), ey - body.forearm * math.cos(au + el)


def reach(body, lean, target):
    """Углы плеча и локтя, чтобы кисть была в точке (или как можно ближе), в пределах
    суставов. Локоть сгибает предплечье вперёд."""
    sx, sy = shoulder(body, lean)
    tx, ty = target
    u, f = body.upper_arm, body.forearm
    d = math.hypot(tx - sx, ty - sy)
    d = min(max(d, abs(u - f) + 1e-6), u + f - 1e-6)
    phi = math.atan2(tx - sx, -(ty - sy))                # угол на цель от «вниз», вперёд +
    alpha = math.acos((u * u + d * d - f * f) / (2 * u * d))
    beta = math.acos((u * u + f * f - d * d) / (2 * u * f))
    au = phi - alpha
    el = math.pi - beta
    lo, hi = skeleton.JOINT_LIMITS[skeleton.JOINTS.index("sh_l")]
    sh = min(max(au - lean, lo), hi)
    elo, ehi = skeleton.JOINT_LIMITS[skeleton.JOINTS.index("el_l")]
    return sh, min(max(el, elo), ehi)


def left_hit(point, s, fret, body):
    """Палец в нужном ладу на нужной струне? Открытая струна - левой руке попадать не надо."""
    if fret == 0:
        return True
    nx, ny = nut(body)
    ax, ay = axis()
    px, py = point[0] - nx, point[1] - ny
    along = -(px * ax + py * ay)                       # от порожка к корпусу
    side = -px * ay + py * ax
    centre = 0.5 * (fret_distance(fret - 1, body) + fret_distance(fret, body))
    ox, oy = across(s, body)
    want_side = -ox * ay + oy * ax
    return (abs(along - centre) <= FRET_HIT * fret_spacing(fret, body) / 2 * 2
            and abs(side - want_side) <= STRING_HIT * STRING_GAP * _k(body))


def right_hit(point, s, body):
    tx, ty = right_target(s, body)
    return math.hypot(point[0] - tx, point[1] - ty) <= 0.015 * _k(body)
