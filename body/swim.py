"""Тело в воде: как плывёт человечек от того, как он гребёт.

Это физика тела, а не умение: какие гребки хорошие, код не говорит - навык swim
(skills/swim.py) это выучивает сам. Модель боком, в 2D:
- тело почти не тонет (плотность чуть меньше воды): горизонтально голова у самой
  поверхности, стоймя (lean - наклон от горизонтали) - ниже, и толкают гребки хуже;
- руки - гребок с амплитудой arm на частоте rate: тяга, пока ладонь идёт назад (полупериод
  синуса); ноги - мах с амплитудой kick, тяга в два раза чаще, но слабее;
- молотить - не плыть: если руками часто и сильно (arm x rate выше удобного), вода уходит в
  брызги - КПД падает; согласованность ног с руками (ноги толкают, пока руки идут вперёд)
  добавляет тяги;
- голову над водой держат гребки и ноги; без них она уходит под воду;
- сопротивление квадратичное, у стоящего в воде - больше.
Шаг - 1/60 с; возвращает, где голова относительно воды и сколько сил ушло.
"""
import math
from dataclasses import dataclass

ARM_F = 9.0          # Н/кг - тяга рук при полном гребке
KICK_F = 3.5         # Н/кг - тяга ног
DRAG = 1.6           # квадратичное сопротивление (1/м)
DRAG_UP = 3.0        # стоймя - во столько больше
COMFORT = 1.2        # arm x rate выше - молотит
HEAD_BASE = 0.06     # м - голова над водой у лежащего на воде без движений
HEAD_LIFT = 0.18     # м - сколько поднимают гребки и ноги
SINK = 0.25          # м - насколько уходит голова, если стоять в воде без движений


@dataclass
class Stroke:
    arm: float = 0.0       # 0..1
    rate: float = 1.0      # Гц, 0.4..2.5
    kick: float = 0.0      # 0..1
    lean: float = 0.0      # рад, 0 - горизонтально, до 1.3 - стоймя


class SwimBody:
    def __init__(self, x, level, facing=1):
        self.x = x
        self.level = level
        self.facing = facing
        self.vx = 0.0
        self.phase = 0.0
        self.head = 0.0          # м, голова над водой (меньше нуля - под водой)
        self.effort = 0.0        # сил ушло за всё время
        self.path = 0.0          # м проплыл (с поворотами)

    def step(self, s, dt=1 / 60):
        arm = min(max(s.arm, 0.0), 1.0)
        kick = min(max(s.kick, 0.0), 1.0)
        rate = min(max(s.rate, 0.4), 2.5)
        lean = min(max(s.lean, 0.0), 1.3)
        self.phase = (self.phase + rate * dt) % 1.0
        ang = 2 * math.pi * self.phase
        flat = math.cos(lean)                                    # горизонтально - 1
        churn = 1.0 / (1.0 + 2.0 * max(0.0, arm * rate - COMFORT) ** 2)
        arm_push = arm * max(0.0, math.sin(ang)) * churn
        leg_push = kick * abs(math.sin(2 * ang)) * 0.5
        # ноги в помощь рукам: толкают, пока руки идут вперёд (вторая половина цикла)
        timing = 1.0 + 0.6 * kick * max(0.0, -math.sin(ang))
        thrust = (ARM_F * arm_push + KICK_F * leg_push) * timing * flat
        drag = DRAG * (1.0 + DRAG_UP * (1.0 - flat)) * self.vx * abs(self.vx)
        self.vx += (thrust - drag) * dt
        self.vx = max(-0.5, self.vx)
        self.x += self.facing * self.vx * dt
        self.path += abs(self.vx) * dt
        lift = HEAD_LIFT * min(1.0, 0.7 * arm * churn + 0.6 * kick)
        target = HEAD_BASE * flat - SINK * (1.0 - flat) + lift
        self.head += (target - self.head) * min(1.0, 4.0 * dt)
        self.effort += (arm * arm * rate + 0.5 * kick * kick * rate) * dt
        return self.head
