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


def draw_human(p, f, human, t, speaking=0.0):
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
    arm(near, 1.0)
    draw_head(p, f, human, pts, t, speaking)
    return pts


def draw_head(p, f, human, pts, t, speaking):
    b = human.body
    m = f.ppm
    c = f.P(pts["head"])
    r = b.head_r * m
    nx, ny = pts["head"][0] - pts["neck"][0], pts["head"][1] - pts["neck"][1]
    ln = math.hypot(nx, ny) or 1.0
    up = QPointF(nx / ln, -ny / ln)                   # вверх головы, в точках экрана (y вниз)
    fwd = QPointF(-up.y() * human.facing, up.x() * human.facing)   # вперёд - по взгляду

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
    back = 1 if human.facing > 0 else -1
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


def thoughts_of(decision, curiosity):
    """Подписи желаний мозга: [(текст, сила 0..1)]."""
    if decision is None:
        return []
    out = [("иду →" if decision.turn >= 0 else "← иду", min(1.0, abs(decision.turn))),
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

def draw_scene(p, rect, world, human, cam_x, cam_y, t, colors, speech=None, thoughts=None):
    f = Frame(rect, cam_x, cam_y)
    p.save()
    p.setClipRect(rect)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    draw_sky(p, f, world, t)
    pal = world.palette
    draw_clouds(p, f, world, t)
    draw_layer(p, f, world, world.far, FAR_PAR, 0.58, 0.3, qc(pal["far"]))
    draw_layer(p, f, world, world.mid, MID_PAR, 0.7, 0.22, qc(pal["mid"]))
    haze = QLinearGradient(0, rect.top() + f.h * 0.45, 0, rect.top() + f.h * 0.8)
    haze.setColorAt(0, qc(pal["sky_bottom"], 0.0))
    haze.setColorAt(1, qc(pal["sky_bottom"], 0.25))
    p.fillRect(rect, haze)
    draw_ground(p, f, world, t)
    draw_entities(p, f, world, t, human.px if human else None)
    if human is not None:
        # тень под ногами
        gy = world.height_at(human.px)
        lift = max(0.0, human.py - gy - human.body.leg())
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, int(60 / (1 + lift))))
        p.drawEllipse(QPointF(f.X(human.px), f.Y(gy) + 0.03 * f.ppm), 0.35 * f.ppm / (1 + 0.3 * lift),
                      0.07 * f.ppm)
        pts = draw_human(p, f, human, t, speech[1] if speech else 0.0)
        head = f.P((pts["head"][0], pts["head"][1] + human.body.head_r))
        if thoughts:
            draw_thoughts(p, f, head, thoughts, colors)
            head = QPointF(head.x(), head.y() - f.h * 0.05)
        if speech and speech[1] > 0:
            draw_bubble(p, f, head, speech[0], min(1.0, speech[1] / 0.4), colors)
    p.restore()
    return f


def thumbnail(world, body, width, height, colors, t=0.0):
    """Картинка мира с человечком для галереи."""
    human = Human(body, world, x=world.width * 0.5)
    image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QColor(colors["surface"]))
    p = QPainter(image)
    draw_scene(p, QRectF(0, 0, width, height), world, human, human.px, human.py, t, colors)
    p.end()
    return image


class WorldView(QWidget):
    """Окно в мир: рисует жизнь (game/life.py), камера плавно идёт за человечком."""

    clicked = Signal()

    def __init__(self, colors, parent=None):
        super().__init__(parent)
        self.colors = colors
        self.life = None
        self.cam_x = self.cam_y = 0.0
        self.show_thoughts = False
        self.paused = False
        self.clock = 0.0
        self.setMinimumSize(420, 300)
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
        self.cam_y += (target_y - self.cam_y) * (1.0 - math.exp(-dt / 0.6))

    def paintEvent(self, event):
        p = QPainter(self)
        rect = QRectF(0, 0, self.width(), self.height())
        if self.life is None:
            p.fillRect(rect, QColor(self.colors["surface"]))
            p.end()
            return
        life = self.life
        thoughts = thoughts_of(life.decision, life.brain.curiosity) if self.show_thoughts else None
        draw_scene(p, rect, life.world, life.human, self.cam_x, self.cam_y, self.clock, self.colors,
                   speech=(life.last_phrase, life.speaking), thoughts=thoughts)
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
