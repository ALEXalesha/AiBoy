"""Тело собаки: туловище - тело pymunk, ноги - пружины опоры под бёдрами, скелет ног
рисуется по ним (два звена, колено - обратная кинематика).

Почему не ноги-шарниры с пружинами-сервоприводами (первая версия, замер в плане): награда
походки скакала от соседних чисел к соседним (у одной собаки лучшая походка - кувырок у
другой), на склоне она повисала на передних голенях, а рефлекс равновесия, достаточно сильный,
чтобы не кувыркаться, держал её на весу. Здесь - как у зверей в играх:

- **туловище** - настоящее тело: падает, кувыркается, ударяется о землю брюхом, мяч в него
  отскакивает, в воде плавает, прыгает;
- **нога в опоре** (фаза походки) - пружина с демпфером от бедра до земли под лапой: толкает
  туловище вверх, сколько сжата, и вдоль земли - к скорости шага (не больше трения); лапа
  стоит, где поставлена; ушла за размах - срывается (спотыкается);
- **нога в переносе** не держит: лапа идёт вперёд дугой высотой «подъём»; зацепилась за кочку
  (подъём мал) - запинается: толчок назад и клевок носом;
- **походка** - частота, размах, подъём, доля опоры: их решает сеть движения
  (animals/dog/motor.py). Рысь: передняя ближняя и задняя дальняя - вместе.

Сидит, лежит - опора всех ног, задние (или все) короче. В воде - плавучесть туловища и
гребки. Отряхивается после воды. Перевернулась - перекатывается на лапы.
"""
import math

import pymunk

from physics.world import GROUND, THING

DOG = 8
GROUP = 7
SWIM_SHARE = 0.55            # туловище в воде на эту долю своей высоты
SWIM_DEPTH = 0.8             # вода глубже этой доли высоты бедра - плывёт
NOSE_UP = 0.25               # рад - плывёт, задрав нос
STAND_H = 0.9                # высота бедра стоя - доля длины ноги
POSES = {"stand": (STAND_H, STAND_H), "sit": (0.95, 0.45), "lie": (0.4, 0.38)}   # высота бедра: передние, задние
MU = 1.0                     # трение лапы о землю
SLIP = 1.7                   # лапа ушла дальше размаха x SLIP - срывается
TRIP = 0.6                   # толчок назад при запинке, м/с
LEG_KP = 0.6                 # рад крена -> доля ноги короче/длиннее
LEG_KD = 0.12


class Leg:
    def __init__(self, front, offset):
        self.front = front
        self.offset = offset     # сдвиг фазы
        self.stance = True
        self.plant = None        # x, где стоит лапа
        self.paw = (0.0, 0.0)
        self.load = 0.0          # сила опоры, Н


class DogBody:
    def __init__(self, phys, x, look, facing=1):
        self.phys = phys
        self.look = look
        self.L = L = look.length
        self.H = 0.3 * L
        self.thigh = 0.28 * L + 0.04
        self.shin = 0.26 * L + 0.04
        self.leg = self.thigh + self.shin
        self.M = 4.0 + 26.0 * look.size ** 1.5
        self.facing = 1 if facing >= 0 else -1
        self.phase = 0.0
        self.gait = (2.5, 0.35, 0.5, 0.6)        # частота, размах (рад), подъём (доля ноги), доля опоры
        self.drive = 1.0                         # 0 - стоит на месте (ноги переступают на месте)
        self.pose = None
        self.wet = 0.0
        self.level = None
        self.shake = 0.0
        self.upside = 0.0
        self.paddle = 0.0
        self.swimming = False
        self.soaked = False
        self.trips = 0
        self.legs = [Leg(True, 0.0), Leg(True, math.pi), Leg(False, math.pi), Leg(False, 0.0)]
        self.torso = None
        self.build(x)
        phys.forces.append(self._forces)

    # --- сборка ---
    def build(self, x):
        L, H = self.L, self.H
        g = max(self.phys.ground(x + d) for d in (-0.45 * L, 0.0, 0.45 * L))
        torso = pymunk.Body(self.M, pymunk.moment_for_box(self.M, (L, H)))
        torso.position = (x, g + STAND_H * self.leg + H / 2)
        shape = pymunk.Poly.create_box(torso, (L, H), radius=0.02)
        shape.friction, shape.elasticity = 0.6, 0.1
        shape.filter = pymunk.ShapeFilter(group=GROUP, categories=DOG, mask=GROUND | THING)
        self.phys.space.add(torso, shape)
        self.torso, self.shape = torso, shape
        for leg in self.legs:
            leg.plant, leg.stance = None, True
            hx, hy = self.hip(leg)
            leg.paw = (hx, self.phys.ground(hx))

    def remove(self):
        if self.torso is not None and self.torso in self.phys.space.bodies:
            self.phys.space.remove(self.torso, self.shape)

    def destroy(self):
        self.remove()
        if self._forces in self.phys.forces:
            self.phys.forces.remove(self._forces)

    def place(self, x):
        """Поставить стоя в точке x (вернулась в окно физики)."""
        self.remove()
        self.build(x)

    # --- где она ---
    @property
    def x(self):
        return float(self.torso.position.x)

    @property
    def y(self):
        return float(self.torso.position.y)

    def angle(self):
        return float((self.torso.angle + math.pi) % (2 * math.pi) - math.pi)

    def vx(self):
        return float(self.torso.velocity.x)

    def hip(self, leg):
        hx = (0.42 if leg.front else -0.42) * self.L * self.facing
        return tuple(self.torso.local_to_world((hx, -self.H / 2)))

    def paws_down(self):
        return sum(1 for leg in self.legs if leg.stance and leg.load > 0)

    def head(self):
        c, s = math.cos(self.torso.angle), math.sin(self.torso.angle)
        hx, hy = self.facing * 0.58 * self.L, 0.35 * self.H + 0.06
        return self.x + c * hx - s * hy, self.y + s * hx + c * hy

    def points(self):
        """Для рисования: углы туловища и ноги [(бедро, колено, лапа, передняя?)]."""
        corners = [tuple(self.torso.local_to_world(v)) for v in self.shape.get_vertices()]
        legs = []
        for leg in self.legs:
            hip = self.hip(leg)
            legs.append((hip, self._knee(hip, leg.paw, leg.front), leg.paw, leg.front))
        return corners, legs

    def _knee(self, hip, paw, front):
        dx, dy = paw[0] - hip[0], paw[1] - hip[1]
        d = max(1e-6, min(self.thigh + self.shin - 1e-4, math.hypot(dx, dy)))
        a = (self.thigh ** 2 + d * d - self.shin ** 2) / (2 * self.thigh * d)
        ang = math.acos(max(-1.0, min(1.0, a)))
        base = math.atan2(dy, dx)
        bend = (-1 if front else 1) * self.facing        # передние - локтем назад, задние - вперёд
        k = base - bend * ang
        return hip[0] + self.thigh * math.cos(k), hip[1] + self.thigh * math.sin(k)

    # --- управление ---
    def turn(self, facing):
        f = 1 if facing >= 0 else -1
        if f != self.facing:
            self.facing = f
            for leg in self.legs:
                leg.plant = None

    def jump(self, strength):
        """Толчок туловищу вверх и вперёд (strength 0..1), если хоть две лапы на земле."""
        if self.paws_down() < 2 or self.wet > 0.2:
            return False
        v = 2.0 + 3.0 * strength
        t = self.torso
        t.velocity = (t.velocity.x + self.facing * 0.5 * v, max(t.velocity.y, 0.0) + 0.85 * v)
        return True

    def stance_speed(self):
        """Скорость, с которой лапы в опоре толкают назад: длина шага / время опоры."""
        freq, amp, lift, duty = self.gait
        return 2.0 * self.leg * math.sin(amp) * freq / max(0.3, duty) * self.drive

    def step(self, dt, level=None):
        """Раз в кадр: фаза, где лапы, вода, переворот. Силы - в _forces на каждом подшаге."""
        freq, amp, lift, duty = self.gait
        self.level = level
        # глубже, чем ей по грудь, - плывёт (лапы гребут, не стоят): иначе крупная шла по дну
        # и окуналась с головой (замер)
        # глубину меряет под передними лапами: у берега они достают дна и вытаскивают её (замер:
        # по глубине под серединой она гребла в склон берега и стояла)
        front = self.x + self.facing * 0.42 * self.L
        self.swimming = level is not None and level - self.phys.ground(front) > SWIM_DEPTH * STAND_H * self.leg
        if self.pose is None:
            self.phase = (self.phase + 2 * math.pi * freq * dt) % (2 * math.pi)
        reach = self.leg * math.sin(amp)
        for leg in self.legs:
            hip = self.hip(leg)
            ground = self.phys.ground(hip[0])
            if self.pose is not None or self.swimming:
                # сидит, лежит - все лапы под собой; плывёт - лапы гребут под водой
                leg.stance = self.pose is not None and not self.swimming
                leg.plant = hip[0] if leg.stance else None
                if leg.stance:
                    leg.paw = (hip[0], ground)
                else:
                    leg.paw = (hip[0] + self.facing * 0.25 * self.leg * math.sin(self.phase + leg.offset),
                               hip[1] - 0.6 * self.leg)
                continue
            psi = (self.phase + leg.offset) % (2 * math.pi)
            stance = psi < 2 * math.pi * duty
            if stance:
                if not leg.stance or leg.plant is None:
                    leg.plant = hip[0] + self.facing * reach * self.drive    # ставит лапу впереди
                if abs(leg.plant - hip[0]) > reach * SLIP + 0.05:
                    leg.plant = None                                        # сорвалась
                    stance = False
                else:
                    leg.paw = (leg.plant, self.phys.ground(leg.plant))
            if not stance:
                s = (psi - 2 * math.pi * duty) / (2 * math.pi * (1 - duty) + 1e-6)
                s = min(1.0, max(0.0, s))
                px = hip[0] + self.facing * reach * self.drive * (2 * s - 1)
                py = self.phys.ground(px) + lift * self.leg * math.sin(math.pi * s)
                ahead = self.phys.ground(px + self.facing * 0.08)
                if s < 0.85 and py < ahead - 0.015 and self.drive > 0:
                    self._trip()                                            # запнулась о кочку
                leg.paw = (px, min(py, hip[1]))
                leg.plant = None
            leg.stance = stance
        self._wet_state(level)
        self._upright(dt)
        if self.shake > 0:
            self.shake = max(0.0, self.shake - dt)

    def _trip(self):
        t = self.torso
        if t.velocity.x * self.facing > 0.3:
            t.velocity = (t.velocity.x - self.facing * TRIP * 0.1, t.velocity.y)
            t.angular_velocity -= self.facing * 0.3
            self.trips += 1

    def _wet_state(self, level):
        share = 0.0
        if level is not None:
            share = min(1.0, max(0.0, (level - (self.y - self.H / 2)) / self.H))
        if share > 0.3:
            self.soaked = True
        elif share <= 0.02 and self.soaked and not self.swimming:
            self.shake, self.soaked = 1.0, False                 # вылезла - отряхивается
        self.wet = share
        freq, amp, lift, duty = self.gait
        self.paddle = min(1.0, amp * freq / 1.2) * (0.6 + 0.4 * lift) if share > 0 else 0.0

    def _forces(self, h):
        """На каждом подшаге pymunk (силы он сбрасывает): опора ног, тяга, вода."""
        if self.torso is None:
            return
        t = self.torso
        k = self.M * 9.81 / (2 * 0.12 * self.leg)               # две ноги держат вес на сжатии 12 %
        c = 2.0 * math.sqrt(k * self.M / 2) * 0.6
        v_goal = self.facing * self.stance_speed()
        stance_legs = [leg for leg in self.legs if leg.stance]
        for leg in self.legs:
            leg.load = 0.0
        if abs(self.angle()) > 1.3:
            stance_legs = []                                    # на боку или на спине ноги не держат
        # равновесие - через ноги, как у зверя: нос задрался относительно склона - передние короче,
        # задние длиннее (замер: у подножия крутого склона передние пружины сжимались сильнее, и
        # она кувыркалась назад; внешний рефлекс-момент держал её на весу)
        ground_slope = math.atan2(self.phys.ground(self.x + self.L / 2) - self.phys.ground(self.x - self.L / 2), self.L)
        pitch = ((self.angle() - ground_slope + math.pi) % (2 * math.pi) - math.pi) * self.facing
        spin = t.angular_velocity * self.facing
        adjust = max(-0.35, min(0.35, LEG_KP * pitch + LEG_KD * spin)) * self.leg
        for leg in stance_legs:
            hx, hy = self.hip(leg)
            px = leg.plant if leg.plant is not None else hx
            gy = self.phys.ground(px)
            if self.pose is not None:
                rest = POSES[self.pose][0 if leg.front else 1] * self.leg
            else:
                rest = STAND_H * self.leg
            rest = max(0.3 * self.leg, min(1.0 * self.leg, rest - adjust if leg.front else rest + adjust))
            comp = rest - (hy - gy)
            if comp <= 0 or hy < gy:
                continue
            vy = t.velocity_at_world_point((hx, hy)).y
            fn = max(0.0, k * comp - c * vy)
            leg.load = fn
            if self.pose is None:
                slope = (self.phys.ground(px + 0.2) - self.phys.ground(px - 0.2)) / 0.4
                tx = 1.0 / math.sqrt(1.0 + slope * slope)
                v_tan = (t.velocity.x + t.velocity.y * slope) * tx
                want = self.M / len(stance_legs) * (v_goal * (1 if self.drive else 0) - v_tan) / 0.15
                ft = max(-MU * fn, min(MU * fn, want))
                t.apply_force_at_world_point((ft * tx, ft * slope * tx), (hx, hy))
            else:
                ft = max(-MU * fn, min(MU * fn, -self.M * t.velocity.x / 0.2 / len(stance_legs)))
                t.apply_force_at_world_point((ft, 0.0), (hx, hy))
            t.apply_force_at_world_point((0.0, fn), (hx, hy))
        t.angular_velocity *= max(0.0, 1.0 - 1.5 * h)
        if self.wet <= 0:
            return
        share = self.wet
        t.apply_force_at_world_point((0.0, self.M * 9.81 * share / SWIM_SHARE), (self.x, self.y))
        kd = max(0.0, 1.0 - 3.0 * share * h)
        t.velocity = (t.velocity.x * kd, t.velocity.y * kd)
        t.angular_velocity *= max(0.0, 1.0 - 4.0 * share * h)
        t.torque += -self.M * self.L * 3.0 * (self.angle() - self.facing * NOSE_UP) * share   # плывёт носом вверх
        t.apply_force_at_world_point((self.facing * self.paddle * self.M * 2.0 * share * self.drive, 0.0),
                                     (self.x, self.y))

    def _upright(self, dt):
        a = self.angle()
        self.upside = self.upside + dt if abs(a) > 1.3 else 0.0
        if self.upside > 1.2:
            self.torso.angular_velocity = -4.0 * a              # перекатывается на лапы
            self.torso.velocity = (self.torso.velocity.x, max(self.torso.velocity.y, 1.2))
            self.upside = 0.0
