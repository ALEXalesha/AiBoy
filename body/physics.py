"""Физика человечка: суставы тянутся к целям мозга, стопа на земле держит, таз в полёте
падает.

Ходьба здесь не написана. Человечек двигается, только если стопа стоит на земле, а мозг
отводит эту ногу назад: стопа не скользит, значит едет таз - это и есть толчок ногой об
землю. Обе стопы на земле - таз по среднему. Быстро разогнутая нога даёт скорость вверх.
Выход мозга «прыжок» - толчок вверх и чуть вперёд, по взгляду. Слишком крутой подъём -
стена. Приземление быстрее FALL_V - падение: человечек ложится и через FALL_TIME встаёт,
мозг в это время ногами не управляет.
"""
import math

import numpy as np

from body import skeleton

DT = 1.0 / 60.0
G = 9.8
TAU = 0.12            # с - за столько сустав проходит 63 % пути к цели
MAX_SPEED = 7.0       # рад/с
SLIP = 0.06           # м за шаг физики - дальше стопа проскальзывает
WALL_SLOPE = 2.5      # подъём круче (м на м) - стена
PUSH_V = 3.0          # м/с - предел скорости вверх от разгибания ног
PUSH_MIN = 1.5        # м/с - медленнее ноги только поднимают таз
PUSH_GAIN = 0.8
JUMP_V = 4.3          # м/с - толчок прыжка вверх
JUMP_VX = 1.0         # м/с - и вперёд, по взгляду
JUMP_CD = 0.5         # с между прыжками
FALL_V = 8.5          # м/с - приземление быстрее - падение
FALL_TIME = 1.6       # с лежит и встаёт
CONTACT = 0.015       # м - стопа ближе к земле - стоит
SNAP = 0.06           # м - под горку таз прилипает к земле, а не подпрыгивает
TURN = 0.35           # гистерезис разворота
AIR_DRAG = 0.995
FALLEN_POSE = skeleton.clip(np.array([0.0, 0.2, 1.4, 1.1, 0.4, 0.6, 0.5, 0.2, 0.6, 0.3]))
TILT = 1.4


class Human:
    def __init__(self, body, world, x=0.0):
        self.body, self.world = body, world
        self.angles = skeleton.rest_angles()
        self.targets = self.angles.copy()
        self.facing = 1
        self.px = float(x)
        rel = skeleton.feet(body, self.angles, 1)
        self.py = self._req(self.px, rel)
        self.vx = self.vy = 0.0
        self.contact = [True, True]
        self.anchor = [self.px + fx for fx, _ in rel]
        self.grounded = True
        self.fallen = 0.0
        self.tilt = 0.0
        self.jump_cd = 0.0
        self.odometer = 0.0
        self.air_time = 0.0

    # --- опора ---
    def _ground(self, x):
        return self.world.height_at(x)

    def _req(self, px, rel):
        """На какой высоте таз, чтобы ни одна стопа не ушла под землю."""
        return max(self._ground(px + fx) - fy for fx, fy in rel)

    def _req_points(self, px):
        pts = skeleton.points(self.body, self.angles, self.facing, (0.0, 0.0), self.tilt)
        return max(self._ground(px + x) - y for x, y in pts.values())

    def points(self):
        return skeleton.points(self.body, self.angles, self.facing, (self.px, self.py), self.tilt)

    def ground_below(self):
        return self._ground(self.px)

    # --- шаг ---
    def step(self, targets=None, turn=0.0, jump=False, dt=DT):
        events = []
        if self.fallen > 0:
            return self._lying(dt, events)
        if targets is not None:
            self.targets = skeleton.clip(np.asarray(targets, float))
        flipped = False
        if turn > TURN and self.facing < 0 or turn < -TURN and self.facing > 0:
            self.facing = -self.facing
            flipped = True
        self.jump_cd = max(0.0, self.jump_cd - dt)

        rel_old = skeleton.feet(self.body, self.angles, self.facing)
        k = 1.0 - math.exp(-dt / TAU)
        delta = np.clip((self.targets - self.angles) * k, -MAX_SPEED * dt, MAX_SPEED * dt)
        self.angles = self.angles + delta
        rel = skeleton.feet(self.body, self.angles, self.facing)
        if flipped:
            self.anchor = [self.px + fx for fx, _ in rel]
            rel_old = rel

        px_old = self.px
        if jump and self.grounded and self.jump_cd <= 0:
            self.vy = JUMP_V
            self.vx += self.facing * JUMP_VX
            self.grounded = False
            self.contact = [False, False]
            self.jump_cd = JUMP_CD
            events.append("jump")

        if self.grounded:
            self._on_ground(rel, rel_old, dt)
        else:
            self._in_air(rel, dt, events)
        self.odometer += abs(self.px - px_old)
        return events

    def _blocked(self, px, nx, rel):
        rise = self._req(nx, rel) - self._req(px, rel)
        return rise > max(0.03, WALL_SLOPE * abs(nx - px))

    def _on_ground(self, rel, rel_old, dt):
        planted = [i for i in (0, 1) if self.contact[i]] or [0, 1]
        if len(planted) == 2:
            # вес на той стопе, что ниже: вторая, приподнятая хоть на сантиметр, скользит
            need = [self._ground(self.anchor[i]) - rel[i][1] for i in (0, 1)]
            if abs(need[0] - need[1]) > 0.01:
                planted = [int(np.argmax(need))]
        want = sum(self.anchor[i] - rel[i][0] for i in planted) / len(planted)
        dx = float(np.clip(want - self.px, -SLIP, SLIP))
        if dx and self._blocked(self.px, self.px + dx, rel):
            dx = 0.0
        push = self._req(self.px, rel) - self._req(self.px, rel_old)    # разгибание ног
        self.px += dx
        self.vx = 0.6 * self.vx + 0.4 * dx / dt
        req = self._req(self.px, rel)
        vy_b = self.vy - G * dt
        py_b = self.py + vy_b * dt
        if py_b <= req or (self.vy <= 0 and self.py - req < SNAP):
            self.py = req
            # медленно разогнутая нога только поднимает таз, отрывает от земли - резкая
            self.vy = float(np.clip((push / dt - PUSH_MIN) * PUSH_GAIN, 0.0, PUSH_V))
        else:
            self.py, self.vy = py_b, vy_b
        self._touch(rel)
        self.air_time = 0.0 if self.grounded else self.air_time

    def _in_air(self, rel, dt, events):
        self.air_time += dt
        self.vx *= AIR_DRAG
        nx = self.px + self.vx * dt
        if self._blocked(self.px, nx, rel) and self._req(nx, rel) > self.py - 0.05:
            nx = self.px
            self.vx = -0.1 * self.vx
        self.px = nx
        self.vy -= G * dt
        self.py += self.vy * dt
        req = self._req(self.px, rel)
        if self.py <= req:
            if self.vy < -FALL_V:
                self.fallen = FALL_TIME
                events.append("fall")
            else:
                events.append("land")
            self.py = req
            self.vy = 0.0
            self.grounded = True
            self._touch(rel, force=True)

    def _touch(self, rel, force=False):
        gaps = [self.py + fy - self._ground(self.px + fx) for fx, fy in rel]
        self.contact = [g <= CONTACT for g in gaps]
        if force and not any(self.contact):
            self.contact[int(np.argmin(gaps))] = True
        self.grounded = any(self.contact)
        for i, (fx, _) in enumerate(rel):
            if self.contact[i]:
                self.anchor[i] = self.px + fx

    def _lying(self, dt, events):
        """Упал: ложится назад, лежит, встаёт. Мозг ногами не управляет."""
        self.fallen = max(0.0, self.fallen - dt)
        t = FALL_TIME - self.fallen
        rise = 0.5
        if self.fallen <= 0:
            self.tilt = 0.0
        elif self.fallen < rise:
            self.tilt = TILT * self.fallen / rise
        else:
            self.tilt = TILT * min(1.0, t / 0.3)
        target = FALLEN_POSE if self.fallen >= rise else skeleton.rest_angles()
        k = 1.0 - math.exp(-dt / TAU)
        self.angles = self.angles + (target - self.angles) * k
        self.targets = self.angles.copy()
        self.vx *= 0.85
        self.px += self.vx * dt
        self.vy = 0.0
        if self.fallen > 0:
            self.py = self._req_points(self.px)
            self.grounded = True
            self.contact = [False, False]
        else:
            rel = skeleton.feet(self.body, self.angles, self.facing)
            self.py = self._req(self.px, rel)
            self._touch(rel, force=True)
            events.append("up")
        return events
