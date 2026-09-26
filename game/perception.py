"""«Что видит»: восприятие и намерение мозга - для подсказки поверх мира.

Всё берётся из того же вектора наблюдения, который получил мозг на последнем шаге
(life.last_obs), и из его же предсказателя и решения, - ничего не досчитывается заново
и не придумывается:

- лучи - 11 точек рельефа, которые он видит (высоты от земли под ним, глазами человечка);
- сущности, которые попали в наблюдение (не больше двух ближайших);
- где ему интересно - ошибка предсказателя по каждой точке рельефа и по сущностям: чем
  хуже он угадал, что там увидит, тем ярче;
- куда он идёт - намерение, усреднённое за последнюю секунду, и реальная скорость;
- короткая сводка словами.
"""
from dataclasses import dataclass, field

import numpy as np

from body.physics import JUMP_CD
from brain import observe as ob
from world.generate import KINDS

KIND_NAMES = {"tree": "дерево", "bush": "куст", "stone": "камень", "flower": "цветок", "mushroom": "гриб",
              "crystal": "кристалл"}
KIND_DAT = {"tree": "к дереву", "bush": "к кусту", "stone": "к камню", "flower": "к цветку",
            "mushroom": "к грибу", "crystal": "к кристаллу"}


@dataclass
class Perception:
    rays: list = field(default_factory=list)       # [(x, y, интерес 0..1)] - точки рельефа в мире
    entities: list = field(default_factory=list)   # [(x, y, вид, интерес 0..1)]
    intent: float = 0.0                            # -1 влево .. 1 вправо, среднее за секунду
    speed: float = 0.0                             # м/с по x
    jump_in: float = 0.0                           # с до готовности прыжка
    want_jump: float = 0.0
    lines: list = field(default_factory=list)      # сводка словами


def _interest(errs, scale):
    return float(np.clip(np.sqrt(max(errs, 0.0)) / scale, 0.0, 1.0))


def perceive(life):
    """Восприятие по последнему наблюдению мозга. Пока мозг не сделал ни шага - пусто."""
    o = life.last_obs
    if o is None:
        return None
    h = life.human
    f = h.facing
    g0 = life.obs_ground
    errs = getattr(life.brain, "last_err_vec", None)
    if errs is None or len(errs) != ob.OBS_DIM:
        errs = np.zeros(ob.OBS_DIM)
    scale = float(np.sqrt(max(life.brain.err_ema or 0.0, 1e-8))) * 3.0
    per = Perception()
    for i, off in enumerate(ob.OFFSETS):
        x = life.obs_x + f * off
        per.rays.append((x, g0 + 2.0 * float(o[i]), _interest(errs[i], scale)))
    k = ob.ENTITIES.start
    for slot in range(2):
        base = k + slot * ob.ENTITY
        if o[base + 2] < 0.5:
            continue
        kind = KINDS[int(np.argmax(o[base + 3:base + 3 + len(KINDS)]))]
        x = life.obs_x + f * float(o[base]) * ob.SEE
        y = g0 + 3.0 * float(o[base + 1])
        e = float(errs[base:base + ob.ENTITY].sum())
        per.entities.append((x, y, kind, _interest(e, scale)))
    per.intent = life.intent
    per.speed = h.vx
    per.jump_in = max(0.0, h.jump_cd)
    per.want_jump = life.decision.want_jump if life.decision else 0.0
    per.lines = words(per, life)
    return per


def side_of(dx):
    return "справа" if dx >= 0 else "слева"


def words(per, life):
    x0 = life.obs_x
    seen = []
    for x, _, kind, _ in sorted(per.entities, key=lambda e: abs(e[0] - x0)):
        seen.append(f"{KIND_NAMES[kind]} {side_of(x - x0)} {abs(x - x0):.0f} м")
    hs = [(x - x0, y - life.obs_ground) for x, y, _ in per.rays]
    for sgn in (1, -1):
        rise = [dy for dx, dy in hs if dx * sgn > 0.5]
        if rise and max(rise) > 1.0:
            seen.append(f"холм {side_of(sgn)}")
        elif rise and min(rise) < -1.0:
            seen.append(f"обрыв {side_of(sgn)}")
    if not seen:
        seen.append("ровное место")
    if abs(per.intent) < 0.25:
        want = "постоять, оглядеться"
    else:
        sgn = 1 if per.intent > 0 else -1
        want = "вправо" if sgn > 0 else "влево"
        ahead = [(abs(x - x0), kind) for x, _, kind, _ in per.entities if (x - x0) * sgn > 0]
        if ahead:
            want += ", " + KIND_DAT[min(ahead)[1]]
    if not life.human.grounded:
        jump = "в прыжке"
    elif per.jump_in > 0:
        jump = f"готов через {per.jump_in:.1f} с".replace(".", ",")
    else:
        jump = f"готов, хочет на {round(per.want_jump * 100)} %"
    return [f"видит: {', '.join(seen)}", f"хочет: {want}", f"прыжок: {jump}"]


__all__ = ["Perception", "perceive", "JUMP_CD"]
