"""Мир и человечек на экране: QPainter, камера следует за человечком.

Всё, что видно, придумали сети: рельеф, цвета неба и земли, горы, облака, сущности, сам
человечек. Здесь - только как это нарисовать: небо градиентом, горы и облака с параллаксом,
земля со слоями и травой, сущности по виду, человечек из капсул, пузырь с фразой.
"""
import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (QColor, QFont, QImage, QLinearGradient, QPainter, QPainterPath, QPen,
                           QPolygonF, QRadialGradient)
from PySide6.QtWidgets import QSizePolicy, QWidget

from body.physics import Human
from game.perception import perceive

VIEW_M = 7.5            # м по высоте в кадре
HORIZON = 0.66          # где по высоте камера держит человечка
FAR_PAR, MID_PAR, CLOUD_PAR = 0.2, 0.45, 0.1


def qc(rgb, a=1.0, k=1.0):
    r, g, b = (min(1.0, max(0.0, c * k)) for c in rgb)
    return QColor.fromRgbF(r, g, b, a)


def blend(a, b, t):
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def lighter(rgb, t):
    return blend(rgb, (1, 1, 1), t)


def darker(rgb, t):
    return blend(rgb, (0, 0, 0), t)


def _rng_for(seed):
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, 7])


class Frame:
    """Перевод метров мира в точки кадра."""

    def __init__(self, rect, cam_x, cam_y):
        self.rect = rect
        self.w, self.h = rect.width(), rect.height()
        self.ppm = self.h / VIEW_M
        self.cam_x, self.cam_y = cam_x, cam_y
        self.half = self.w / 2 / self.ppm

    def X(self, x):
        return self.rect.left() + self.w / 2 + (x - self.cam_x) * self.ppm

    def Y(self, y):
        return self.rect.top() + self.h * HORIZON - (y - self.cam_y) * self.ppm

    def P(self, pt):
        return QPointF(self.X(pt[0]), self.Y(pt[1]))


# --- небо, горы, облака ---

def draw_sky(p, f, world, t):
    pal = world.palette
    r = f.rect
    g = QLinearGradient(0, r.top(), 0, r.top() + f.h * 0.8)
    g.setColorAt(0, qc(pal["sky_top"]))
    g.setColorAt(1, qc(pal["sky_bottom"]))
    p.fillRect(r, g)
    sx, sy = r.left() + f.w * world.sun_x, r.top() + f.h * 0.2
    rad = f.h * 0.055
    if world.night:
        rng = _rng_for(world.seed)
        n = 90
        xs, ys = rng.random(n), rng.random(n) * 0.6
        tw = rng.random(n) * 6.28
        size = 0.6 + 1.4 * rng.random(n) ** 3
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(n):
            a = 0.45 + 0.45 * math.sin(t * (1.2 + size[i]) + tw[i])
            p.setBrush(QColor.fromRgbF(1, 1, 0.92, max(0.1, a)))
            p.drawEllipse(QPointF(r.left() + xs[i] * f.w, r.top() + ys[i] * f.h), size[i], size[i])
        glow = QRadialGradient(QPointF(sx, sy), rad * 3.2)
        glow.setColorAt(0, QColor.fromRgbF(0.9, 0.93, 1.0, 0.25))
        glow.setColorAt(1, QColor.fromRgbF(0.9, 0.93, 1.0, 0.0))
        p.setBrush(glow)
        p.drawEllipse(QPointF(sx, sy), rad * 3.2, rad * 3.2)
        p.setBrush(QColor.fromRgbF(0.95, 0.95, 0.88))
        p.drawEllipse(QPointF(sx, sy), rad, rad)
        p.setBrush(qc(pal["sky_top"], 1.0))
        p.drawEllipse(QPointF(sx + rad * 0.45, sy - rad * 0.2), rad * 0.9, rad * 0.9)
    else:
        warm = blend(lighter(pal["sky_bottom"], 0.6), (1.0, 0.93, 0.7), 0.6)
        glow = QRadialGradient(QPointF(sx, sy), rad * 4)
        glow.setColorAt(0, qc(warm, 0.55))
        glow.setColorAt(1, qc(warm, 0.0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(glow)
        p.drawEllipse(QPointF(sx, sy), rad * 4, rad * 4)
        p.setBrush(qc(lighter(warm, 0.5)))
        p.drawEllipse(QPointF(sx, sy), rad, rad)


def draw_layer(p, f, world, layer, par, base, scale, color):
    r = f.rect
    step = 6.0
    pts = [QPointF(r.left(), r.bottom() + 1)]
    sx = 0.0
    while sx <= f.w + step:
        u = f.cam_x * par + (sx - f.w / 2) / f.ppm
        hgt = float(world.layer_at(layer, u))
        pts.append(QPointF(r.left() + sx, r.top() + f.h * base - hgt * f.ppm * scale))
        sx += step
    pts.append(QPointF(r.left() + f.w + step, r.bottom() + 1))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(color)
    p.drawPolygon(QPolygonF(pts))


def draw_clouds(p, f, world, t):
    if not world.clouds:
        return
    alpha = 0.22 if world.night else 0.85
    tint = lighter(world.palette["sky_bottom"], 0.75)
    span = world.width * CLOUD_PAR + f.w / f.ppm + 12
    p.setPen(Qt.PenStyle.NoPen)
    for c in world.clouds:
        u = (c.x * CLOUD_PAR + t * 0.25 - f.cam_x * CLOUD_PAR) % span - 6
        x = f.rect.left() + u * f.ppm
        y = f.rect.top() + f.h * (0.42 - c.y / 30.0)
        s = c.size * f.ppm * 0.3
        for dx, dy, k in ((-0.9, 0.15, 0.7), (0.0, 0.0, 1.0), (0.95, 0.12, 0.75), (0.4, -0.35, 0.7)):
            p.setBrush(qc(tint, alpha))
            p.drawEllipse(QPointF(x + dx * s, y + dy * s), s * k * 1.25, s * k * 0.8)


# --- земля ---

def visible_xs(f, world, margin=1.0):
    x0 = math.floor((f.cam_x - f.half - margin) / world.dx) * world.dx
    n = int((2 * f.half + 2 * margin) / world.dx) + 3
    return x0 + np.arange(n) * world.dx


def draw_ground(p, f, world, t):
    pal = world.palette
    xs = visible_xs(f, world)
    hs = world.heights_at(xs)
    top = [QPointF(f.X(x), f.Y(y)) for x, y in zip(xs, hs)]
    poly = QPolygonF(top + [QPointF(top[-1].x(), f.rect.bottom() + 2), QPointF(top[0].x(), f.rect.bottom() + 2)])
    g = QLinearGradient(0, f.Y(f.cam_y + 0.8), 0, f.rect.bottom())
    g.setColorAt(0, qc(pal["ground"]))
    g.setColorAt(1, qc(pal["ground_deep"]))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(g)
    p.drawPolygon(poly)
    # слои почвы - линии вдоль поверхности
    p.setBrush(Qt.BrushStyle.NoBrush)
    for depth, a in ((0.45, 0.22), (1.1, 0.18), (1.9, 0.14), (2.9, 0.1)):
        path = QPainterPath(QPointF(top[0].x(), top[0].y() + depth * f.ppm))
        for q in top[1:]:
            path.lineTo(q.x(), q.y() + depth * f.ppm)
        p.setPen(QPen(qc(pal["ground_deep"], a, 0.8), max(1.0, 0.05 * f.ppm)))
        p.drawPath(path)
    # трава: кромка и пучки
    edge = QPainterPath(top[0])
    for q in top[1:]:
        edge.lineTo(q)
    pen = QPen(qc(pal["grass"]), max(2.0, 0.14 * f.ppm))
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.drawPath(edge)
    idx = np.round(xs / world.dx).astype(int) % len(world.grass)
    blade = max(1.0, 0.035 * f.ppm)
    for k in range(0, len(xs), 2):
        gv = float(world.grass[idx[k]])
        if gv < 0.45:
            continue
        x, y = xs[k], hs[k]
        base = QPointF(f.X(x), f.Y(y))
        length = (0.08 + 0.3 * (gv - 0.45)) * f.ppm
        sway = 0.25 * math.sin(t * 1.6 + x * 0.9)
        col = qc(lighter(pal["grass"], 0.15) if k % 4 else darker(pal["grass"], 0.15))
        p.setPen(QPen(col, blade, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        for j, lean in enumerate((-0.35, 0.05, 0.4)):
            a = lean + sway
            p.drawLine(QPointF(base.x() + (j - 1) * 0.05 * f.ppm, base.y()),
                       QPointF(base.x() + (j - 1) * 0.05 * f.ppm + math.sin(a) * length,
                               base.y() - math.cos(a) * length * (0.8 + 0.2 * j % 2)))


# --- сущности ---

def unwrap(x, near, width):
    return x + round((near - x) / width) * width


def draw_entities(p, f, world, t, human_x=None):
    for e in world.entities:
        x = unwrap(e.x, f.cam_x, world.width)
        if abs(x - f.cam_x) > f.half + 3:
            continue
        near = human_x is not None and abs(unwrap(e.x, human_x, world.width) - human_x) < 1.2
        draw_entity(p, f, e, x, t, world.night, near)


def draw_entity(p, f, e, x, t, night, near):
    s = e.size
    base = QPointF(f.X(x), f.Y(e.y) + 0.04 * f.ppm)
    m = f.ppm
    col = e.color
    sway = math.sin(t * 1.3 + x) * (0.07 if near else 0.025)
    p.save()
    p.translate(base)
    p.setPen(Qt.PenStyle.NoPen)
    kind = e.kind
    if kind == "tree":
        p.rotate(math.degrees(sway) * 0.4)
        trunk = darker(blend(col, (0.45, 0.3, 0.18), 0.75), 0.25)
        tw = 0.16 * s * m
        th = 1.7 * s * m
        p.setBrush(qc(trunk))
        p.drawRoundedRect(QRectF(-tw / 2, -th, tw, th + 2), tw * 0.3, tw * 0.3)
        if e.variant > 0.6:                        # ель - ярусы треугольников
            for i, k in enumerate((1.0, 0.8, 0.6)):
                y0 = -th * (0.35 + 0.28 * i)
                wdt = 1.0 * s * m * k
                tri = QPolygonF([QPointF(-wdt, y0), QPointF(wdt, y0), QPointF(0, y0 - 1.0 * s * m * k)])
                p.setBrush(qc(darker(col, 0.15 * (2 - i))))
                p.drawPolygon(tri)
        else:
            rad = 0.62 * s * m
            for dx, dy, k, shade in ((-0.55, -0.1, 0.8, 0.18), (0.55, -0.05, 0.85, 0.12), (0.0, -0.5, 1.0, 0.0),
                                     (0.2, -0.75, 0.6, -0.15)):
                c = lighter(col, -shade) if shade < 0 else darker(col, shade)
                p.setBrush(qc(c))
                p.drawEllipse(QPointF(dx * rad, -th - dy * rad * -1 - rad * 0.2), rad * k, rad * k)
    elif kind == "bush":
        p.rotate(math.degrees(sway) * 0.5)
        rad = 0.36 * s * m
        for dx, dy, k, shade in ((-0.7, 0.35, 0.8, 0.2), (0.7, 0.35, 0.85, 0.15), (0.0, 0.0, 1.0, 0.0),
                                 (0.3, -0.25, 0.55, -0.2)):
            c = lighter(col, 0.2) if shade < 0 else darker(col, shade)
            p.setBrush(qc(c))
            p.drawEllipse(QPointF(dx * rad, -rad * 0.9 + dy * rad), rad * k, rad * k * 0.9)
        if e.variant > 0.5:                        # ягоды
            p.setBrush(qc(lighter(blend(col, (0.9, 0.15, 0.25), 0.7), 0.1)))
            for dx, dy in ((-0.5, -0.1), (0.2, -0.6), (0.6, 0.0), (-0.1, 0.3)):
                p.drawEllipse(QPointF(dx * rad, -rad * 0.9 + dy * rad), rad * 0.12, rad * 0.12)
    elif kind == "stone":
        grey = blend(col, (0.55, 0.55, 0.58), 0.6)
        wdt, hgt = 0.55 * s * m, 0.4 * s * m * (0.7 + 0.5 * e.variant)
        path = QPainterPath()
        path.moveTo(-wdt, 0)
        path.cubicTo(-wdt, -hgt * 0.9, -wdt * 0.3, -hgt * 1.1, wdt * 0.2, -hgt)
        path.cubicTo(wdt * 0.8, -hgt * 0.9, wdt, -hgt * 0.4, wdt, 0)
        path.closeSubpath()
        p.setBrush(qc(grey))
        p.drawPath(path)
        p.setBrush(qc(lighter(grey, 0.3), 0.7))
        p.drawEllipse(QPointF(-wdt * 0.25, -hgt * 0.7), wdt * 0.3, hgt * 0.16)
    elif kind == "flower":
        p.rotate(math.degrees(sway) * 1.5)
        hgt = 0.55 * s * m
        stem = QPen(qc((0.3, 0.6, 0.3)), max(1.0, 0.035 * m))
        p.setPen(stem)
        p.drawLine(QPointF(0, 0), QPointF(0, -hgt))
        p.drawLine(QPointF(0, -hgt * 0.4), QPointF(0.12 * m * s, -hgt * 0.55))
        p.setPen(Qt.PenStyle.NoPen)
        pr = 0.09 * s * m
        for i in range(5):
            a = i * 2 * math.pi / 5 + t * 0.2
            p.setBrush(qc(col))
            p.drawEllipse(QPointF(math.cos(a) * pr * 1.2, -hgt + math.sin(a) * pr * 1.2), pr, pr)
        p.setBrush(qc((1.0, 0.85, 0.3)))
        p.drawEllipse(QPointF(0, -hgt), pr * 0.8, pr * 0.8)
    elif kind == "mushroom":
        if night:
            glow(p, QPointF(0, -0.35 * s * m), 0.9 * s * m, col, 0.35)
        sw, sh = 0.11 * s * m, 0.35 * s * m
        p.setBrush(qc((0.95, 0.92, 0.85)))
        p.drawRoundedRect(QRectF(-sw / 2, -sh, sw, sh), sw * 0.4, sw * 0.4)
        cap = QPainterPath()
        cw = 0.32 * s * m
        cap.moveTo(-cw, -sh + 2)
        cap.cubicTo(-cw, -sh - cw * 1.1, cw, -sh - cw * 1.1, cw, -sh + 2)
        cap.closeSubpath()
        p.setBrush(qc(col))
        p.drawPath(cap)
        p.setBrush(QColor(255, 255, 255, 220))
        for dx, dy in ((-0.45, -0.45), (0.1, -0.7), (0.5, -0.35)):
            p.drawEllipse(QPointF(dx * cw, -sh + dy * cw), cw * 0.12, cw * 0.12)
    elif kind == "crystal":
        pulse = 0.5 + 0.5 * math.sin(t * 2.0 + x)
        glow(p, QPointF(0, -0.5 * s * m), 1.2 * s * m, col, (0.45 if night else 0.2) + 0.15 * pulse)
        for dx, hk, wk, ang in ((0.0, 1.0, 1.0, 0.0), (-0.22, 0.65, 0.7, -18), (0.24, 0.55, 0.65, 20)):
            p.save()
            p.translate(dx * s * m, 0)
            p.rotate(ang)
            hgt, wdt = 1.0 * s * m * hk, 0.16 * s * m * wk
            poly = QPolygonF([QPointF(-wdt, 0), QPointF(-wdt, -hgt * 0.7), QPointF(0, -hgt),
                              QPointF(wdt, -hgt * 0.7), QPointF(wdt, 0)])
            p.setBrush(qc(col, 0.92))
            p.drawPolygon(poly)
            p.setBrush(qc(lighter(col, 0.45), 0.8))
            p.drawPolygon(QPolygonF([QPointF(-wdt, -hgt * 0.7), QPointF(0, -hgt), QPointF(0, 0), QPointF(-wdt, 0)]))
            p.restore()
    p.restore()


def glow(p, center, radius, col, alpha):
    g = QRadialGradient(center, radius)
    g.setColorAt(0, qc(lighter(col, 0.3), alpha))
    g.setColorAt(1, qc(col, 0.0))
    p.setBrush(g)
    p.drawEllipse(center, radius, radius)
    p.setBrush(Qt.BrushStyle.NoBrush)


# --- человечек ---

def _capsule(p, a, b, width, color):
    pen = QPen(color, width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.drawLine(a, b)


def draw_human(p, f, human, t, speaking=0.0, guitar=None):
    """guitar: None - без гитары, "back" - за спиной, "play" - играет (сидит)."""
    b = human.body
    pts = human.points()
    P = f.P
    m = f.ppm
    near, far = ("l", "r") if human.facing > 0 else ("r", "l")
    shade = 0.28

    def arm(side, k):
        _capsule(p, P(pts["sh_" + side]), P(pts["el_" + side]), b.arm_w * m * 1.05, qc(b.shirt, 1, k))
        _capsule(p, P(pts["el_" + side]), P(pts["ha_" + side]), b.arm_w * m * 0.85, qc(b.skin, 1, k))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qc(b.skin, 1, k))
        p.drawEllipse(P(pts["ha_" + side]), b.arm_w * m * 0.55, b.arm_w * m * 0.55)

    def leg(side, k):
        _capsule(p, P(pts["hip_" + side]), P(pts["kn_" + side]), b.leg_w * m * 1.1, qc(b.pants, 1, k))
        _capsule(p, P(pts["kn_" + side]), P(pts["ft_" + side]), b.leg_w * m * 0.95, qc(b.pants, 1, k))
        foot = P(pts["ft_" + side])
        knee = P(pts["kn_" + side])
        # стопа - по направлению голени, носком вперёд
        dx, dy = foot.x() - knee.x(), foot.y() - knee.y()
        ang = math.degrees(math.atan2(dy, dx)) - 90 * human.facing
        p.save()
        p.translate(foot)
        p.rotate(ang if human.facing > 0 else ang + 180)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qc(b.shoes, 1, k))
        L, H = 0.24 * m, 0.085 * m
        p.drawRoundedRect(QRectF(-L * 0.3, -H * 0.5, L, H), H * 0.5, H * 0.5)
        p.restore()

    fk = 1 - shade
    if guitar == "back":
        guitar_on_back(p, f, human)
    elif guitar == "play":
        draw_stump(p, f, human, pts["pelvis"])
    arm(far, fk)
    leg(far, fk)
    # корпус: таз в штанах, туловище в рубашке
    pelvis, neck = pts["pelvis"], pts["neck"]
    ux, uy = neck[0] - pelvis[0], neck[1] - pelvis[1]
    ln = math.hypot(ux, uy) or 1.0
    ux, uy = ux / ln, uy / ln
    tw = b.torso_w * m
    low = (pelvis[0] + ux * 0.02, pelvis[1] + uy * 0.02)
    chest_top = (neck[0] - ux * tw / m * 0.45, neck[1] - uy * tw / m * 0.45)
    _capsule(p, P(low), P((pelvis[0] + ux * 0.14, pelvis[1] + uy * 0.14)), tw * 0.95, qc(b.pants))
    _capsule(p, P((pelvis[0] + ux * (0.1 + tw / m * 0.45), pelvis[1] + uy * (0.1 + tw / m * 0.45))),
             P(chest_top), tw, qc(b.shirt))
    _capsule(p, P(neck), P((neck[0] - ux * 0.06, neck[1] - uy * 0.06)), 0.07 * m, qc(b.skin, 1, 0.92))
    leg(near, 1.0)
    if guitar == "play":
        guitar_in_hands(p, f, human)
        # кисть дальней руки - на струнах, поверх корпуса
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qc(b.skin, 1, fk))
        p.drawEllipse(P(pts["ha_" + far]), b.arm_w * m * 0.6, b.arm_w * m * 0.6)
    arm(near, 1.0)
    draw_head(p, f, human, pts, t, speaking)
    return pts


def draw_head(p, f, human, pts, t, speaking):
    b = human.body
    m = f.ppm
    facing = getattr(human, "gaze", 0) or human.facing    # осматривается - голова смотрит назад
    c = f.P(pts["head"])
    r = b.head_r * m
    nx, ny = pts["head"][0] - pts["neck"][0], pts["head"][1] - pts["neck"][1]
    ln = math.hypot(nx, ny) or 1.0
    up = QPointF(nx / ln, -ny / ln)                   # вверх головы, в точках экрана (y вниз)
    fwd = QPointF(-up.y() * facing, up.x() * facing)   # вперёд - по взгляду

    def at(f_, u_):
        return QPointF(c.x() + fwd.x() * f_ * r + up.x() * u_ * r, c.y() + fwd.y() * f_ * r + up.y() * u_ * r)

    hair = qc(b.hair)
    p.setPen(Qt.PenStyle.NoPen)
    if b.hair_style == 2:                              # длинные - за головой
        p.setBrush(hair)
        p.drawEllipse(at(-0.35, -0.35), r * 0.85, r * 1.05)
    p.setBrush(qc(b.skin))
    p.drawEllipse(c, r, r)
    ang = math.degrees(math.atan2(-up.y(), up.x()))    # угол «вверх» в градусах Qt
    rect = QRectF(c.x() - r * 1.04, c.y() - r * 1.04, r * 2.08, r * 2.08)
    back = 1 if facing > 0 else -1
    if b.hair_style in (0, 2):                         # шапка волос сверху и сзади
        p.setBrush(hair)
        p.drawChord(rect, int((ang - 60 * back) * 16), int(160 * back * 16))
        p.drawEllipse(at(-0.62, 0.05), r * 0.42, r * 0.5)
    elif b.hair_style == 1:                            # ёжик
        p.setBrush(hair)
        p.drawChord(rect, int((ang - 50 * back) * 16), int(150 * back * 16))
        for k in range(5):
            a = -0.9 + k * 0.45
            tip = at(math.sin(a) * 1.45 - 0.1, math.cos(a) * 1.45)
            b1, b2 = at(math.sin(a - 0.25) * 0.9, math.cos(a - 0.25) * 0.9), at(math.sin(a + 0.25) * 0.9,
                                                                                math.cos(a + 0.25) * 0.9)
            p.drawPolygon(QPolygonF([b1, tip, b2]))
    else:                                              # кепка цвета рубашки
        p.setBrush(qc(b.shirt, 1, 0.85))
        p.drawChord(rect, int((ang - 70 * back) * 16), int(175 * back * 16))
        visor = QPolygonF([at(0.2, 0.35), at(1.45, 0.3), at(1.35, 0.15), at(0.3, 0.2)])
        p.drawPolygon(visor)
    # лицо
    blink = (t % 4.3) < 0.13
    e = b.eye
    for fwd_k, size in ((0.52, 1.0), (0.05, 0.8)):
        ec = at(fwd_k, 0.12)
        if blink:
            p.setPen(QPen(QColor(40, 30, 30), max(1.0, 0.018 * m)))
            p.drawLine(QPointF(ec.x() - r * 0.12, ec.y()), QPointF(ec.x() + r * 0.12, ec.y()))
            p.setPen(Qt.PenStyle.NoPen)
            continue
        p.setBrush(QColor(255, 255, 255))
        p.drawEllipse(ec, r * 0.17 * e * size, r * 0.21 * e * size)
        p.setBrush(QColor(35, 30, 40))
        pc = at(fwd_k + 0.07, 0.1)
        p.drawEllipse(pc, r * 0.1 * e * size, r * 0.12 * e * size)
        p.setBrush(QColor(255, 255, 255, 230))
        p.drawEllipse(at(fwd_k + 0.1, 0.16), r * 0.035 * e, r * 0.035 * e)
    if b.cheeks:
        p.setBrush(QColor(255, 120, 130, 80))
        p.drawEllipse(at(0.4, -0.22), r * 0.16, r * 0.11)
    p.setBrush(qc(b.skin, 1, 0.9))
    p.drawEllipse(at(0.95, -0.02), r * 0.12, r * 0.12)
    mouth = at(0.62, -0.42)
    if speaking > 0:
        open_ = 0.5 + 0.5 * abs(math.sin(t * 14))
        p.setBrush(QColor(110, 40, 50))
        p.drawEllipse(mouth, r * 0.13, r * 0.1 * open_ + r * 0.03)
    else:
        p.setPen(QPen(QColor(110, 50, 55), max(1.0, 0.02 * m), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(at(0.45, -0.4), at(0.75, -0.38))
        p.setPen(Qt.PenStyle.NoPen)


# --- гитара ---

WOOD = (0.78, 0.52, 0.27)
WOOD_DARK = (0.36, 0.22, 0.12)


def draw_guitar(p, f, center, axis, k=1.0):
    """Гитара в мировых координатах: center - середина корпуса, axis - единичный вектор от
    корпуса к грифу, k - масштаб (по росту человечка)."""
    ax, ay = axis
    nx, ny = -ay, ax
    m = f.ppm * k

    def at(u, v):
        return QPointF(f.X(center[0] + (ax * u + nx * v) * k), f.Y(center[1] + (ay * u + ny * v) * k))

    p.save()
    p.setPen(QPen(qc(WOOD_DARK), max(1.0, 0.01 * m)))
    p.setBrush(qc(WOOD))
    for u, r in ((-0.06, 0.13), (0.1, 0.095)):          # корпус - две округлости
        c = at(u, 0.0)
        p.drawEllipse(c, r * m, r * m)
    p.setBrush(qc(WOOD_DARK))
    p.drawEllipse(at(0.02, 0.0), 0.035 * m, 0.035 * m)   # розетка
    neck = QPen(qc(WOOD_DARK), 0.04 * m)
    neck.setCapStyle(Qt.PenCapStyle.FlatCap)
    p.setPen(neck)
    p.drawLine(at(0.12, 0.0), at(0.5, 0.0))
    p.setPen(QPen(qc(WOOD), 0.055 * m, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    p.drawLine(at(0.5, 0.0), at(0.58, 0.0))                # головка
    p.setPen(QPen(QColor(235, 235, 225, 200), max(0.6, 0.004 * m)))
    for v in (-0.012, 0.0, 0.012):
        p.drawLine(at(-0.12, v), at(0.52, v))
    p.restore()


def guitar_on_back(p, f, human):
    b = human.body
    k = b.height() / 1.7
    lean = human.angles[0]
    up = (math.sin(lean) * human.facing, math.cos(lean))
    back = (-human.facing * math.cos(lean), math.sin(lean))
    center = (human.px + (up[0] * 0.28 + back[0] * 0.13) * k, human.py + (up[1] * 0.28 + back[1] * 0.13) * k)
    tilt = 0.35
    axis = (up[0] * math.cos(tilt) + back[0] * math.sin(tilt), up[1] * math.cos(tilt) + back[1] * math.sin(tilt))
    draw_guitar(p, f, center, axis, k * 0.9)


def guitar_in_hands(p, f, human):
    from music import hands
    b = human.body
    k = b.height() / 1.7
    cx, cy = hands.body_center(b)
    ax, ay = hands.axis()
    center = (human.px + human.facing * cx, human.py + cy)
    draw_guitar(p, f, center, (human.facing * ax, ay), k * 0.95)


def draw_stump(p, f, human, pelvis):
    """Пенёк, на котором он сидит с гитарой: от земли до таза."""
    x, top = pelvis[0], pelvis[1] - human.body.leg_w * 0.8
    ground = human.world.height_at(x) - 0.05
    if top <= ground:
        return
    half = 0.16 * human.body.height() / 1.7
    p.save()
    p.setPen(QPen(qc((0.30, 0.20, 0.12)), max(1.0, 0.012 * f.ppm)))
    p.setBrush(qc((0.50, 0.36, 0.22)))
    p.drawRect(QRectF(QPointF(f.X(x - half), f.Y(top)), QPointF(f.X(x + half), f.Y(ground))))
    p.setBrush(qc((0.80, 0.66, 0.46)))
    p.drawEllipse(QPointF(f.X(x), f.Y(top)), half * f.ppm, 0.035 * f.ppm)
    p.restore()


def draw_chest(p, f, world, x):
    """Сундук с игрушками: деревянный ящик с крышкой и оковкой."""
    x = unwrap(x, f.cam_x, world.width)
    if abs(x - f.cam_x) > f.half + 1:
        return
    g = world.height_at(x)
    w, h = 0.7, 0.45
    p.save()
    p.setPen(QPen(qc((0.28, 0.17, 0.09)), max(1.0, 0.015 * f.ppm)))
    p.setBrush(qc((0.55, 0.34, 0.16)))
    p.drawRect(QRectF(QPointF(f.X(x - w / 2), f.Y(g + h)), QPointF(f.X(x + w / 2), f.Y(g))))
    p.setBrush(qc((0.62, 0.40, 0.20)))
    p.drawRect(QRectF(QPointF(f.X(x - w / 2 - 0.03), f.Y(g + h + 0.12)), QPointF(f.X(x + w / 2 + 0.03), f.Y(g + h))))
    p.setPen(QPen(qc((0.78, 0.68, 0.30)), max(1.0, 0.03 * f.ppm)))
    for u in (-0.22, 0.22):
        p.drawLine(QPointF(f.X(x + u), f.Y(g + h + 0.12)), QPointF(f.X(x + u), f.Y(g)))
    p.restore()


def guitar_lying(p, f, world, x):
    x = unwrap(x, f.cam_x, world.width)
    if abs(x - f.cam_x) > f.half + 1:
        return
    y = world.height_at(x) + 0.12
    draw_guitar(p, f, (x, y), (1.0, 0.0), 0.9)


# --- пузырь и мысли ---

def draw_bubble(p, f, head, text, alpha, colors):
    if not text:
        return
    font = QFont(p.font())
    font.setPixelSize(max(11, int(f.h * 0.032)))
    font.setWeight(QFont.Weight.DemiBold)
    p.setFont(font)
    fm = p.fontMetrics()
    tw = fm.horizontalAdvance(text) + 22
    th = fm.height() + 12
    x = min(max(head.x() - tw * 0.3, f.rect.left() + 6), f.rect.right() - tw - 6)
    y = head.y() - th - f.ppm * 0.45
    rect = QRectF(x, y, tw, th)
    bg = QColor(colors["bubble"])
    bg.setAlphaF(0.93 * alpha)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(bg)
    p.drawRoundedRect(rect, th / 2, th / 2)
    tail = QPolygonF([QPointF(head.x() - 4, rect.bottom() - 1), QPointF(head.x() + 8, rect.bottom() - 1),
                      QPointF(head.x() + 1, rect.bottom() + 9)])
    p.drawPolygon(tail)
    fg = QColor(colors["bubble_text"])
    fg.setAlphaF(alpha)
    p.setPen(fg)
    p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)


def thoughts_of(decision, curiosity, intent=None):
    """Подписи желаний мозга: [(текст, сила 0..1)]. Направление - намерение за последнюю
    секунду (как стрелка «что видит»), а не одно решение: иначе подпись мигала бы
    «← иду» / «иду →» от каждого колебания."""
    if decision is None:
        return []
    way = decision.turn if intent is None else intent
    out = [("иду →" if way >= 0 else "← иду", min(1.0, abs(way))),
           ("прыгнуть?", min(1.0, decision.want_jump * 4)),
           ("интересно", min(1.0, curiosity)),
           ("сказать", min(1.0, decision.say * 6))]
    return out


def draw_thoughts(p, f, head, items, colors):
    font = QFont(p.font())
    font.setPixelSize(max(10, int(f.h * 0.024)))
    p.setFont(font)
    fm = p.fontMetrics()
    widths = [fm.horizontalAdvance(t) + 14 for t, _ in items]
    total = sum(widths) + 4 * (len(items) - 1)
    x = head.x() - total / 2
    y = head.y() - f.ppm * 0.35 - fm.height() - 8
    for (text, strength), w in zip(items, widths):
        rect = QRectF(x, y, w, fm.height() + 6)
        bg = QColor(colors["accent"])
        bg.setAlphaF(0.15 + 0.7 * strength)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(bg)
        p.drawRoundedRect(rect, 7, 7)
        fg = QColor("#ffffff")
        fg.setAlphaF(0.55 + 0.45 * strength)
        p.setPen(fg)
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
        x += w + 4


# --- кадр целиком ---

def draw_fish(p, f, world, school):
    """Рыбы: тело-эллипс, хвост-треугольник, глаз; цвет и форма - из сети рыб."""
    for fish in school.fish:
        x, y = school.position(fish)
        x = unwrap(x, f.cam_x, world.width)
        if abs(x - f.cam_x) > f.half + 1:
            continue
        L, H = fish.length * f.ppm, fish.height * f.ppm
        c = QPointF(f.X(x), f.Y(y))
        p.save()
        p.translate(c)
        if fish.dir < 0:
            p.scale(-1, 1)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qc(fish.color, 0.9))
        wag = math.sin(fish.phase * 2) * 0.25
        tail = QPolygonF([QPointF(-L * 0.4, 0), QPointF(-L * (0.7 + 0.2 * fish.tail), -H * (0.5 + wag)),
                          QPointF(-L * (0.7 + 0.2 * fish.tail), H * (0.5 - wag))])
        p.drawPolygon(tail)
        p.drawEllipse(QPointF(0, 0), L / 2, H / 2)
        p.setBrush(QColor(20, 20, 20, 220))
        p.drawEllipse(QPointF(L * 0.3, -H * 0.1), max(1.0, H * 0.12), max(1.0, H * 0.12))
        p.restore()


def _flag_cloth(p, base, ppm, t, wind, color):
    """Флаг на ветру: древко 2 м и полотнище, волна бежит по нему; ветер - сила и сторона."""
    pole_top = QPointF(base.x(), base.y() - 2.0 * ppm)
    p.setPen(QPen(QColor(90, 70, 50), max(1.5, 0.05 * ppm)))
    p.drawLine(base, pole_top)
    side = 1.0 if wind >= 0 else -1.0
    strength = min(1.0, 0.3 + abs(wind) / 3.0)
    w, h = 0.9 * ppm * strength + 0.3 * ppm, 0.5 * ppm
    top, bottom = [], []
    for k in range(9):
        u = k / 8
        wave = math.sin(t * 7.0 - u * 6.0) * 0.12 * ppm * u * (0.5 + strength)
        x = pole_top.x() + side * u * w
        top.append(QPointF(x, pole_top.y() + wave))
        bottom.append(QPointF(x, pole_top.y() + h + wave - u * (1.0 - strength) * 0.3 * ppm))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(color)
    p.drawPolygon(QPolygonF(top + bottom[::-1]))


def draw_far_mountains(p, f, world, peaks, flags, t, wind):
    """Горы издали: силуэт за ближним рельефом, тем меньше, чем дальше; снег на макушке; флаг на
    покорённой - видно издалека."""
    if not peaks:
        return
    pal = world.palette
    horizon = f.Y(f.cam_y) + 0.4 * f.ppm
    for m in peaks:
        d = (m.summit - f.cam_x + world.width / 2) % world.width - world.width / 2
        if abs(d) < m.face_len + 10:
            continue                              # он на самой горе - её видно рельефом
        # в какой стороне и как далеко: дальняя - у края кадра и ниже (замер кадром: с прежним
        # сдвигом гора в 570 м уезжала за кадр на 1950 точек)
        scale = 1.0 / (1.0 + abs(d) / 400.0)
        cx = f.rect.left() + f.w / 2 + math.copysign(f.w * 0.48 * (1.0 - math.exp(-abs(d) / 250.0)), d)
        hpx = f.h * 0.6 * (m.height / 100.0) * scale
        easy_w = hpx / 0.22 * 0.35
        face_w = hpx / 3.3 * 1.4
        left_w, right_w = (easy_w, face_w) if m.steep > 0 else (face_w, easy_w)
        if cx + right_w < f.rect.left() or cx - left_w > f.rect.right():
            continue
        top = QPointF(cx, horizon - hpx)
        poly = QPolygonF([QPointF(cx - left_w, horizon), QPointF(cx - 0.02 * hpx, top.y()),
                          QPointF(cx + 0.02 * hpx, top.y()), QPointF(cx + right_w, horizon)])
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qc(blend(pal["far"], pal["sky_bottom"], 0.25)))
        p.drawPolygon(poly)
        if m.snow > 0.35:
            k = 0.18
            snow = QPolygonF([QPointF(cx - left_w * k, top.y() + hpx * k), QPointF(cx - 0.02 * hpx, top.y()),
                              QPointF(cx + 0.02 * hpx, top.y()), QPointF(cx + right_w * k, top.y() + hpx * k)])
            p.setBrush(QColor(245, 248, 252, 230))
            p.drawPolygon(snow)
        if any(abs((fx - m.summit + world.width / 2) % world.width - world.width / 2) < 1.0 for fx in flags or ()):
            _flag_cloth(p, top, max(4.0, f.ppm * scale * 0.6), t, wind, QColor(220, 60, 50))


def draw_summit_flags(p, f, world, peaks, flags, t, wind):
    """Флаги на вершинах вблизи."""
    for fx in flags or ():
        x = unwrap(fx, f.cam_x, world.width)
        if abs(x - f.cam_x) > f.half + 2:
            continue
        _flag_cloth(p, QPointF(f.X(x), f.Y(world.height_at(x))), f.ppm, t, wind, QColor(220, 60, 50))


def draw_snow(p, f, world, peaks):
    """Снег у макушки снежной горы: белая кромка по рельефу выше (вершина - 10 м)."""
    for m in peaks or ():
        if m.snow <= 0.35:
            continue
        xs = visible_xs(f, world)
        hs = world.heights_at(xs)
        line = m.top - 10.0
        pts = [QPointF(f.X(x), f.Y(y)) for x, y in zip(xs, hs) if y > line]
        if len(pts) < 2:
            continue
        pen = QPen(QColor(245, 248, 252), max(2.0, 0.18 * f.ppm))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawPolyline(QPolygonF(pts))


def draw_dog(p, f, world, v):
    """Собака: дальние ноги темнее, туловище с пятнами, хвост, голова с ушами и мордой; в зубах -
    мяч (его рисуют игрушки); отряхивается - брызги. Вид - из сети собак (animals/dog/look.py)."""
    if v is None:
        return
    lk = v["look"]
    corners = v["corners"]
    cx = sum(c[0] for c in corners) / 4
    shift = unwrap(cx, f.cam_x, world.width) - cx
    if abs(cx + shift - f.cam_x) > f.half + 2:
        return
    def P(pt):
        return QPointF(f.X(pt[0] + shift), f.Y(pt[1]))
    L = lk.length
    lw = max(1.5, 0.07 * L * f.ppm)
    body = qc(lk.color)
    dark = qc(darker(lk.color, 0.35))
    legs = v["legs"]
    p.save()
    for i in (1, 3):                                     # дальние ноги
        hip, knee, paw, _ = legs[i]
        p.setPen(QPen(dark, lw, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.drawPolyline(QPolygonF([P(hip), P(knee), P(paw)]))
    # хвост от заднего верхнего угла
    fx = v["facing"]
    back = max(corners, key=lambda c: -fx * c[0] + c[1] * 0.3)
    wag = math.sin(f.cam_x * 0.0 + v.get("t", 0.0) * 9.0) * 0.15
    tail_len = (0.15 + 0.35 * lk.tail) * L
    up = 0.4 + 1.0 * lk.curl
    tip = (back[0] - fx * tail_len * math.cos(up + wag), back[1] + tail_len * math.sin(up + wag))
    mid = (back[0] - fx * tail_len * 0.6, back[1] + tail_len * (0.2 + 0.5 * lk.curl))
    path = QPainterPath(P(back))
    path.quadTo(P(mid), P(tip))
    p.setPen(QPen(body, lw * 0.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(path)
    # туловище
    p.setPen(QPen(dark, max(1.0, 0.012 * f.ppm)))
    p.setBrush(body)
    poly = QPainterPath()
    poly.addPolygon(QPolygonF([P(c) for c in corners]))
    p.drawPath(poly)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(qc(lk.spots, 0.85))
    a = v["angle"]
    for k in range(int(1 + 3 * lk.ears * lk.tail)):     # пятна - сколько и где, по виду
        u = -0.3 + 0.25 * k
        sx = cx + fx * u * L * math.cos(a)
        sy = sum(c[1] for c in corners) / 4 + 0.04 * L + fx * u * L * math.sin(a)
        p.drawEllipse(QPointF(f.X(sx + shift), f.Y(sy)), 0.09 * L * f.ppm, 0.06 * L * f.ppm)
    # ближние ноги
    for i in (0, 2):
        hip, knee, paw, _ = legs[i]
        p.setPen(QPen(body, lw, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.drawPolyline(QPolygonF([P(hip), P(knee), P(paw)]))
    # шея и голова
    front = max(corners, key=lambda c: fx * c[0] + c[1] * 0.3)
    hx, hy = v["head"]
    hr = 0.17 * L
    p.setPen(QPen(body, lw * 1.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    p.drawLine(P(front), P((hx, hy)))
    p.setPen(QPen(dark, max(1.0, 0.01 * f.ppm)))
    p.setBrush(body)
    p.drawEllipse(P((hx, hy)), hr * f.ppm, hr * 0.85 * f.ppm)
    p.drawEllipse(P((hx + fx * hr * 0.9, hy - hr * 0.25)), hr * 0.6 * f.ppm, hr * 0.38 * f.ppm)   # морда
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(25, 20, 20))
    p.drawEllipse(P((hx + fx * hr * 1.45, hy - hr * 0.2)), hr * 0.14 * f.ppm, hr * 0.12 * f.ppm)   # нос
    p.drawEllipse(P((hx + fx * hr * 0.35, hy + hr * 0.2)), hr * 0.12 * f.ppm, hr * 0.12 * f.ppm)   # глаз
    p.setBrush(dark)
    if lk.ears < 0.5:                                    # торчком
        ear = [(hx - fx * hr * 0.2, hy + hr * 0.6), (hx - fx * hr * 0.55, hy + hr * 1.5), (hx - fx * hr * 0.65, hy + hr * 0.4)]
        p.drawPolygon(QPolygonF([P(e) for e in ear]))
    else:                                                # висячие
        p.drawEllipse(P((hx - fx * hr * 0.45, hy - hr * 0.1)), hr * 0.3 * f.ppm, hr * 0.62 * f.ppm)
    if v.get("shake", 0.0) > 0:
        p.setBrush(QColor(170, 210, 255, 200))
        for k in range(10):
            ang = k * 0.63 + v["shake"] * 7.0
            r = (0.3 + 0.25 * ((k * 7) % 3)) * L
            p.drawEllipse(QPointF(f.X(cx + shift + r * math.cos(ang)), f.Y(hy - 0.1 + 0.6 * r * math.sin(ang))),
                          0.012 * f.ppm, 0.012 * f.ppm)
    p.restore()


def draw_toys(p, f, world, items):
    """Игрушки в мире: мяч (кружок с бликом), метка цели (флажок)."""
    for kind, x, y, r, color in items:
        x = unwrap(x, f.cam_x, world.width)
        if abs(x - f.cam_x) > f.half + 1:
            continue
        p.save()
        if kind == "ball":
            c = QPointF(f.X(x), f.Y(y))
            p.setPen(QPen(qc(darker(color, 0.4)), max(1.0, 0.012 * f.ppm)))
            p.setBrush(qc(color))
            p.drawEllipse(c, r * f.ppm, r * f.ppm)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255, 120))
            p.drawEllipse(QPointF(c.x() - r * f.ppm * 0.35, c.y() - r * f.ppm * 0.35), r * f.ppm * 0.3, r * f.ppm * 0.3)
        elif kind == "mask":
            c = QPointF(f.X(x), f.Y(y))
            p.setPen(QPen(QColor(20, 30, 60), max(1.0, 0.012 * f.ppm)))
            p.setBrush(QColor(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255), 170))
            p.drawEllipse(c, r * f.ppm, r * 0.6 * f.ppm)
        elif kind == "box":
            w, h, ang = r
            p.translate(QPointF(f.X(x), f.Y(y)))
            p.rotate(-math.degrees(ang))
            p.setPen(QPen(qc(darker(color, 0.35)), max(1.0, 0.01 * f.ppm)))
            p.setBrush(qc(color))
            p.drawRect(QRectF(-w / 2 * f.ppm, -h / 2 * f.ppm, w * f.ppm, h * f.ppm))
        elif kind == "bubble":
            c = QPointF(f.X(x), f.Y(y))
            p.setPen(QPen(QColor(255, 255, 255, 180), max(1.0, 0.012 * f.ppm)))
            g = QRadialGradient(c, r * f.ppm)
            g.setColorAt(0.0, QColor(255, 255, 255, 10))
            g.setColorAt(0.8, QColor(170, 210, 255, 50))
            g.setColorAt(1.0, QColor(255, 200, 240, 120))
            p.setBrush(g)
            p.drawEllipse(c, r * f.ppm, r * f.ppm)
        elif kind == "plane":
            p.translate(QPointF(f.X(x), f.Y(y)))
            p.rotate(-math.degrees(r))
            s = 0.25 * f.ppm
            p.setPen(QPen(QColor(120, 120, 120), 1))
            p.setBrush(qc(color))
            p.drawPolygon(QPolygonF([QPointF(s, 0), QPointF(-s, -0.25 * s), QPointF(-0.7 * s, 0), QPointF(-s, 0.2 * s)]))
        elif kind == "boomerang":
            p.translate(QPointF(f.X(x), f.Y(y)))
            p.rotate(math.degrees(r))
            s = 0.2 * f.ppm
            pen = QPen(qc(color), max(2.0, 0.05 * f.ppm))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.drawPolyline(QPolygonF([QPointF(-s, s * 0.4), QPointF(0, 0), QPointF(s, s * 0.4)]))
        elif kind == "bobber":
            c = QPointF(f.X(x), f.Y(y))
            p.setPen(QPen(QColor(40, 40, 40), 1))
            p.setBrush(qc(color))
            p.drawEllipse(c, r * f.ppm, r * f.ppm * 1.4)
        elif kind == "line":
            x2, y2 = r, color
            x2 = unwrap(x2, f.cam_x, world.width)
            p.setPen(QPen(QColor(230, 230, 230, 160), 1))
            p.drawLine(QPointF(f.X(x), f.Y(y)), QPointF(f.X(x2), f.Y(y2)))
        elif kind == "bike":
            rx, ry, fx, fy, spin, ang = r
            rx, fx = unwrap(rx, f.cam_x, world.width), unwrap(fx, f.cam_x, world.width)
            wr = 0.3 * f.ppm
            pen = QPen(qc(color), max(2.0, 0.04 * f.ppm))
            p.setPen(QPen(QColor(30, 30, 30), max(2.0, 0.03 * f.ppm)))
            p.setBrush(Qt.BrushStyle.NoBrush)
            for cx, cy in ((rx, ry), (fx, fy)):
                c = QPointF(f.X(cx), f.Y(cy))
                p.drawEllipse(c, wr, wr)
                p.drawLine(c, QPointF(c.x() + wr * math.cos(spin), c.y() + wr * math.sin(spin)))
            p.setPen(pen)
            sx, sy = (rx + fx) / 2, (ry + fy) / 2 + 0.45
            p.drawPolyline(QPolygonF([QPointF(f.X(rx), f.Y(ry)), QPointF(f.X(sx), f.Y(sy)),
                                      QPointF(f.X(fx), f.Y(fy)), QPointF(f.X(fx - 0.1 * (fx - rx)), f.Y(sy + 0.15))]))
        elif kind == "drawing":
            pts, pens, facing = r
            if len(pts) >= 2:
                xs = [x + facing * q[0] for q in pts]
                ys = [y + q[1] for q in pts]
                pad = 0.08
                p.setPen(QPen(QColor(150, 140, 120), 1))
                p.setBrush(QColor(250, 248, 240, 235))          # лист бумаги
                p.drawRect(QRectF(QPointF(f.X(min(xs) - pad), f.Y(max(ys) + pad)),
                                  QPointF(f.X(max(xs) + pad), f.Y(min(ys) - pad))))
                pen = QPen(qc(color), max(1.5, 0.02 * f.ppm))
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                p.setPen(pen)
                for k in range(len(pts) - 1):
                    if k < len(pens) and pens[k]:
                        continue
                    a, b = pts[k], pts[k + 1]
                    p.drawLine(QPointF(f.X(x + facing * a[0]), f.Y(y + a[1])),
                               QPointF(f.X(x + facing * b[0]), f.Y(y + b[1])))
        elif kind == "target":
            base = QPointF(f.X(x), f.Y(y))
            top = QPointF(f.X(x), f.Y(y + 0.7))
            p.setPen(QPen(QColor(90, 70, 50), max(1.0, 0.025 * f.ppm)))
            p.drawLine(base, top)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(qc(color))
            p.drawPolygon(QPolygonF([top, QPointF(top.x() + 0.3 * f.ppm, top.y() + 0.1 * f.ppm),
                                     QPointF(top.x(), top.y() + 0.2 * f.ppm)]))
        p.restore()


def draw_water(p, f, world, water, t):
    """Пруды: полупрозрачная вода цвета неба, линия поверхности с мелкой рябью."""
    pal = world.palette
    for pond in water.ponds:
        span = (pond.right - pond.left) % world.width
        x0 = unwrap(pond.left, f.cam_x, world.width)
        if x0 + span < f.cam_x - f.half - 1 or x0 > f.cam_x + f.half + 1:
            if not (x0 - world.width + span >= f.cam_x - f.half - 1 and x0 - world.width <= f.cam_x + f.half + 1):
                continue
            x0 -= world.width
        xs = np.linspace(x0, x0 + span, max(8, int(span * 6)))
        ys = pond.level + 0.025 * np.sin(xs * 3.1 + t * 2.0) + 0.015 * np.sin(xs * 7.3 - t * 3.1)
        top = [QPointF(f.X(x), f.Y(y)) for x, y in zip(xs, ys)]
        bottom = [QPointF(f.X(x), f.Y(world.height_at(x) - 0.05)) for x in xs[::-1]]
        deep = blend(pal["sky_bottom"], (0.1, 0.25, 0.45), 0.6)
        g = QLinearGradient(0, f.Y(pond.level), 0, f.Y(pond.bottom))
        g.setColorAt(0, qc(blend(pal["sky_bottom"], (0.35, 0.6, 0.85), 0.5), 0.55))
        g.setColorAt(1, qc(deep, 0.8))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(g)
        p.drawPolygon(QPolygonF(top + bottom))
        p.setPen(QPen(qc(lighter(pal["sky_bottom"], 0.5), 0.85), max(1.0, 0.025 * f.ppm)))
        p.drawPolyline(QPolygonF(top))


def draw_scene(p, rect, world, human, cam_x, cam_y, t, colors, speech=None, thoughts=None, guitar=None,
               guitar_x=None, chest_x=None, water=None, fish=None, toys=None, dog=None, peaks=None,
               flags=None, wind=0.0):
    f = Frame(rect, cam_x, cam_y)
    p.save()
    p.setClipRect(rect)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    draw_sky(p, f, world, t)
    pal = world.palette
    draw_clouds(p, f, world, t)
    draw_layer(p, f, world, world.far, FAR_PAR, 0.58, 0.3, qc(pal["far"]))
    draw_layer(p, f, world, world.mid, MID_PAR, 0.7, 0.22, qc(pal["mid"]))
    peaks = peaks if peaks is not None else getattr(world, "mountains", None)
    draw_far_mountains(p, f, world, peaks, flags, t, wind)
    haze = QLinearGradient(0, rect.top() + f.h * 0.45, 0, rect.top() + f.h * 0.8)
    haze.setColorAt(0, qc(pal["sky_bottom"], 0.0))
    haze.setColorAt(1, qc(pal["sky_bottom"], 0.25))
    p.fillRect(rect, haze)
    draw_ground(p, f, world, t)
    draw_snow(p, f, world, peaks)
    draw_summit_flags(p, f, world, peaks, flags, t, wind)
    draw_entities(p, f, world, t, human.px if human else None)
    if chest_x is not None:
        draw_chest(p, f, world, chest_x)
    if guitar_x is not None:
        guitar_lying(p, f, world, guitar_x)
    if dog is not None:
        dog = dict(dog, t=t)
        draw_dog(p, f, world, dog)
    if human is not None:
        # тень под ногами
        gy = world.height_at(human.px)
        lift = max(0.0, human.py - gy - human.body.leg())
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, int(60 / (1 + lift))))
        p.drawEllipse(QPointF(f.X(human.px), f.Y(gy) + 0.03 * f.ppm), 0.35 * f.ppm / (1 + 0.3 * lift),
                      0.07 * f.ppm)
        pts = draw_human(p, f, human, t, speech[1] if speech else 0.0, guitar)
        head = f.P((pts["head"][0], pts["head"][1] + human.body.head_r))
        if thoughts:
            draw_thoughts(p, f, head, thoughts, colors)
            head = QPointF(head.x(), head.y() - f.h * 0.05)
        if speech and speech[1] > 0:
            draw_bubble(p, f, head, speech[0], min(1.0, speech[1] / 0.4), colors)
    if toys:
        draw_toys(p, f, world, toys)
    if fish is not None and fish.fish:
        draw_fish(p, f, world, fish)
    if water is not None and water.ponds:
        draw_water(p, f, world, water, t)       # поверх: вброд он по колено в воде
    p.restore()
    return f


def draw_perception(p, f, per, human, colors):
    """«Что видит»: лучи к точкам рельефа из наблюдения мозга, яркость - ошибка его
    предсказателя там (где ему интересно), кольца вокруг увиденных сущностей, стрелка
    намерения за последнюю секунду. Возвращает прямоугольник панели со сводкой."""
    eye = f.P((human.points()["head"][0], human.points()["head"][1] + human.body.head_r * 0.2))
    warm = QColor(colors["warm"])
    cold = QColor(255, 255, 255)
    p.save()
    for x, y, heat in per.rays:
        c = QColor(cold)
        c.setAlphaF(0.12 + 0.2 * heat)
        p.setPen(QPen(c, 1.2, Qt.PenStyle.DashLine))
        hit = QPointF(f.X(x), f.Y(y))
        p.drawLine(eye, hit)
        dot = QColor(warm) if heat > 0.05 else QColor(cold)
        dot.setAlphaF(0.55 + 0.45 * heat)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(dot)
        r = 3.0 + 7.0 * heat
        p.drawEllipse(hit, r, r)
    for x, y, _, heat in per.entities:
        c = QColor(warm)
        c.setAlphaF(0.5 + 0.5 * heat)
        p.setPen(QPen(c, 2.0 + 2.0 * heat))
        p.setBrush(Qt.BrushStyle.NoBrush)
        rr = (0.55 + 0.25 * heat) * f.ppm
        p.drawEllipse(QPointF(f.X(x), f.Y(y) - rr * 0.6), rr, rr)
    # стрелка намерения: от таза в сторону желания, длина - сила желания
    if abs(per.intent) > 0.05:
        base = f.P((human.px, human.py - human.body.leg() * 0.15))
        tip = QPointF(base.x() + per.intent * f.ppm * 1.6, base.y())
        pen = QPen(QColor(colors["accent"]), 4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawLine(base, tip)
        s = 1 if per.intent > 0 else -1
        p.setBrush(QColor(colors["accent"]))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPolygon(QPolygonF([QPointF(tip.x() + s * 10, tip.y()), QPointF(tip.x() - s * 3, tip.y() - 8),
                                 QPointF(tip.x() - s * 3, tip.y() + 8)]))
    # сводка словами - в левом нижнем углу кадра
    font = QFont(p.font())
    font.setPixelSize(max(11, min(14, int(f.h * 0.024))))
    p.setFont(font)
    fm = p.fontMetrics()
    pad = 10
    width = min(f.w - 2 * pad, max(fm.horizontalAdvance(t) for t in per.lines) + 2 * pad)
    lines = [fm.elidedText(t, Qt.TextElideMode.ElideRight, int(width - 2 * pad)) for t in per.lines]
    height = len(lines) * fm.height() + 2 * pad - 2
    box = QRectF(f.rect.left() + pad, f.rect.bottom() - height - pad, width, height)
    bg = QColor(colors["surface"])
    bg.setAlphaF(0.82)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(bg)
    p.drawRoundedRect(box, 10, 10)
    p.setPen(QColor(colors["text"]))
    for i, t in enumerate(lines):
        p.drawText(QPointF(box.left() + pad, box.top() + pad + fm.ascent() + i * fm.height() - 1), t)
    p.restore()
    return box


RING_MIN = 640.0          # мир шире - в миниатюре ещё и всё кольцо сжатым


def ring_profile(p, rect, world, colors, marker=None, fill=None):
    """Всё кольцо мира сжатым в полосу rect: линия рельефа, земля под ней; marker - где он."""
    xs = np.linspace(0.0, world.width, max(2, int(rect.width())))
    hs = world.heights_at(xs)
    lo, hi = float(hs.min()), float(hs.max())
    span = max(hi - lo, 1.0)
    pts = [QPointF(rect.left() + i, rect.bottom() - 3 - (rect.height() - 6) * (h - lo) / span)
           for i, h in enumerate(hs)]
    poly = QPolygonF(pts + [QPointF(rect.right(), rect.bottom()), QPointF(rect.left(), rect.bottom())])
    p.save()
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(fill or colors["surface2"]))
    p.drawRect(rect)
    p.setBrush(qc(world.palette["ground"]))
    p.drawPolygon(poly)
    if marker is not None:
        mx = rect.left() + rect.width() * (marker % world.width) / world.width
        p.setPen(QPen(QColor(colors["warm"]), 2))
        p.drawLine(QPointF(mx, rect.top()), QPointF(mx, rect.bottom()))
    p.restore()


def thumbnail(world, body, width, height, colors, t=0.0):
    """Картинка мира с человечком для галереи. Мир от 640 м в кадр не влезает: внизу -
    всё кольцо сжатым, чтобы было видно хребты и котловины."""
    human = Human(body, world, x=world.width * 0.5)
    image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QColor(colors["surface"]))
    p = QPainter(image)
    from world.water import Water
    draw_scene(p, QRectF(0, 0, width, height), world, human, human.px, human.py, t, colors, water=Water(world))
    if world.width >= RING_MIN:
        strip = QRectF(0, height * 0.8, width, height * 0.2)
        ring_profile(p, strip, world, colors, marker=human.px)
    p.end()
    return image


class RingMap(QWidget):
    """Мини-карта большого мира (от 640 м) на панели: всё кольцо, где он, сундук, гитара."""

    def __init__(self, colors):
        super().__init__()
        self.colors = colors
        self.life = None
        self.setFixedHeight(44)

    def set_life(self, life):
        self.life = life
        self.setVisible(life is not None and life.world.width >= RING_MIN)
        self.update()

    def paintEvent(self, event):
        life = self.life
        if life is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(0, 0, self.width(), self.height())
        ring_profile(p, rect, life.world, self.colors, marker=life.human.px, fill=self.colors["surface2"])
        w = life.world.width
        for x, color in ((getattr(life.inv, "chest_x", None), "text"), (life.guitar_x, "accent")):
            if x is None:
                continue
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(self.colors[color]))
            p.drawEllipse(QPointF(rect.width() * (x % w) / w, 6), 3, 3)
        p.end()


class WorldView(QWidget):
    """Окно в мир: рисует жизнь (game/life.py), камера плавно идёт за человечком."""

    clicked = Signal()

    def __init__(self, colors, parent=None):
        super().__init__(parent)
        self.colors = colors
        self.life = None
        self.cam_x = self.cam_y = 0.0
        self.show_thoughts = False
        self.show_vision = False
        self.vision_rect = None        # где нарисована сводка «что видит» (для проверок)
        self.perception = None
        self.paused = False
        self.clock = 0.0
        self.setMinimumSize(420, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_life(self, life):
        self.life = life
        self.cam_x, self.cam_y = life.human.px, life.human.py
        self.update()

    def advance(self, dt):
        """Камера и часы анимации (трава, звёзды, моргание)."""
        self.clock += dt
        if self.life is None:
            return
        h = self.life.human
        k = 1.0 - math.exp(-dt / 0.35)
        self.cam_x += (h.px - self.cam_x) * k
        target_y = h.world.height_at(h.px) + h.body.leg()
        if abs(h.py - target_y) > 2.0:
            target_y = h.py                      # на стене горы, съезжает - камера за ним, а не за землёй
        if abs(target_y - self.cam_y) > 15.0 or abs(h.px - self.cam_x) > 15.0:
            self.cam_x, self.cam_y = h.px, target_y   # перенесли далеко - сразу туда
        self.cam_y += (target_y - self.cam_y) * (1.0 - math.exp(-dt / 0.6))

    def paintEvent(self, event):
        p = QPainter(self)
        rect = QRectF(0, 0, self.width(), self.height())
        if self.life is None:
            p.fillRect(rect, QColor(self.colors["surface"]))
            p.end()
            return
        life = self.life
        thoughts = thoughts_of(life.decision, life.brain.curiosity, life.intent) if self.show_thoughts else None
        guitar = None
        if getattr(life, "has_guitar", False):
            guitar = "play" if life.playing and life.attempt is not None else "back"
        lying = getattr(life, "guitar_x", None)
        inv = getattr(life, "inv", None)
        if getattr(life, "playing", False) or getattr(life, "toy", None) is not None:
            thoughts = None                     # играет - желаний «куда идти» нет
        f = draw_scene(p, rect, life.world, life.human, self.cam_x, self.cam_y, self.clock, self.colors,
                       speech=(life.last_phrase, life.speaking), thoughts=thoughts, guitar=guitar, guitar_x=lying,
                       chest_x=inv.chest_x if inv is not None else None, water=getattr(life, "water", None),
                       fish=getattr(life, "fish", None),
                       toys=life.toy_visuals() if hasattr(life, "toy_visuals") else None,
                       dog=life.dog.visuals() if getattr(life, "dog", None) is not None else None,
                       flags=life.climb.flags if hasattr(life, "climb") else None,
                       wind=life.wind.at(life.human.px, life.human.py, life.time) if hasattr(life, "wind") else 0.0)
        self.vision_rect = None
        if self.show_vision and not getattr(life, "playing", False):
            # раз в кадр, а не на каждый шаг жизни: на x8 это та же цена
            self.perception = perceive(life)
            if self.perception is not None:
                p.setRenderHint(QPainter.RenderHint.Antialiasing)
                self.vision_rect = draw_perception(p, f, self.perception, life.human, self.colors)
        if self.paused:
            p.fillRect(rect, QColor(0, 0, 0, 70))
            font = QFont(self.font())
            font.setPixelSize(28)
            font.setWeight(QFont.Weight.Bold)
            p.setFont(font)
            p.setPen(QColor(255, 255, 255, 230))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Пауза")
        p.end()


__all__ = ["WorldView", "draw_scene", "thumbnail", "QTimer"]
