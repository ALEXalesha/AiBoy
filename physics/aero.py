"""Своя аэродинамика (в pymunk её нет): пузырь, бумажный самолётик, бумеранг. Шаг 1/60 с.

- **пузырь**: почти невесом - чуть всплывает (тёплое дыхание), а ветер уносит его за
  десятые доли секунды: по горизонтали он летит быстрее, чем поднимается;
- **самолётик**: подъёмная сила и сопротивление по углу атаки (до срыва подъёмная растёт
  линейно, за срывом падает, сопротивление растёт); тангаж тянется к направлению полёта плюс
  балансировка (trim): с нулевой - планирует, с большой - задирается и срывается;
- **бумеранг**: плоская модель - вращение даёт силу поперёк скорости (как подъёмная сила
  вращающегося крыла), поэтому хороший бросок описывает круг и возвращается; вращение и
  скорость гаснут.
"""
import math

G = 9.81
RHO = 1.2


class Bubble:
    def __init__(self, x, y, r):
        self.x, self.y, self.r = x, y, r
        self.vx = self.vy = 0.0
        self.age = 0.0

    def step(self, wind, dt):
        tau = 0.15 + 3.0 * self.r              # больше - тяжелее догоняет ветер
        self.vx += (wind - self.vx) * dt / tau
        self.vy += (0.25 - 2.0 * self.vy) * dt     # всплывает медленно, к 0.12 м/с
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.age += dt


class Plane:
    S = 0.03         # м², крыло
    M = 0.006        # кг
    CL_A = 5.0       # на радиан
    STALL = 0.28     # рад

    def __init__(self, x, y, vx, vy, trim=0.0):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.trim = trim
        self.pitch = math.atan2(vy, vx)
        self.time = 0.0

    def coeffs(self, alpha):
        if abs(alpha) <= self.STALL:
            cl = self.CL_A * alpha
            cd = 0.04 + 0.08 * cl * cl
        else:
            cl = self.CL_A * self.STALL * math.copysign(1.0, alpha) * max(0.0, 1.0 - 3.0 * (abs(alpha) - self.STALL))
            cd = 0.04 + 1.2 * math.sin(abs(alpha))
        return cl, cd

    def step(self, wind, dt):
        ax, ay = self.vx - wind, self.vy
        v = math.hypot(ax, ay) or 1e-6
        path = math.atan2(ay, ax)
        # тангаж тянется к направлению полёта плюс балансировка (хвост)
        self.pitch += (path + self.trim - self.pitch) * min(1.0, 6.0 * dt)
        alpha = self.pitch - path
        cl, cd = self.coeffs(alpha)
        q = 0.5 * RHO * v * v * self.S
        lift, drag = q * cl, q * cd
        nx, ny = -ay / v, ax / v                # поперёк скорости (вверх при полёте вправо)
        fx = -drag * ax / v + lift * nx
        fy = -drag * ay / v + lift * ny - self.M * G
        self.vx += fx / self.M * dt
        self.vy += fy / self.M * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.time += dt
        return alpha


class Boomerang:
    TURN = 0.55      # сила поворота: на (рад/с вращения) x (м/с)
    SPIN_DECAY = 0.12
    DRAG = 0.12

    def __init__(self, x, y, vx, vy, spin):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.spin = spin                         # рад/с, знак - в какую сторону заворачивает
        self.time = 0.0
        self.x0, self.y0 = x, y

    def step(self, wind, dt):
        ax, ay = self.vx - wind, self.vy
        v = math.hypot(ax, ay) or 1e-6
        turn = self.TURN * self.spin / 60.0      # поперечное ускорение на м/с скорости
        nx, ny = -ay / v, ax / v
        self.vx += (turn * v * nx - self.DRAG * ax) * dt
        self.vy += (turn * v * ny - self.DRAG * ay - 0.35 * G * (1.0 - min(1.0, abs(self.spin) / 60.0))) * dt
        self.spin *= math.exp(-self.SPIN_DECAY * dt)
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.time += dt

    def home(self):
        return math.hypot(self.x - self.x0, self.y - self.y0)
