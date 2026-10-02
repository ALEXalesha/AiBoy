"""Реестр игрушек: одна запись на игрушку. Всё остальное (выбор занятия, сундук, «Каталог»,
«Попытки», «Положи») работает по этому списку, а не по именам в коде.

У записи:
- name - имя занятия в выборе (acts/choice.py) и в файлах;
- title - как её называть на экране;
- carry - где он её носит: "back" (за спиной, гитара) или "hands" (в руках, мяч);
- half - минуты, за которые подряд уходит половина тяги (скука; «Как часто» у гитары
  сдвигает её, у других - общий множитель);
- needs - что должно быть в мире: "water" (удочка) - иначе её нельзя выбрать.

Гитара - первая игрушка реестра (1.1.0), дальше - по задачам плана 2.0.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Toy:
    name: str
    title: str
    carry: str = "hands"
    half: float = 5.0
    needs: str = ""


TOYS = {}


def register(toy):
    TOYS[toy.name] = toy
    return toy


def get(name):
    return TOYS.get(name)


def names():
    return list(TOYS)


register(Toy("guitar", "Гитара", carry="back", half=5.0))
register(Toy("ball", "Мяч", carry="hands", half=5.0))
register(Toy("blocks", "Кубики", carry="hands", half=4.5))
register(Toy("bubbles", "Пузыри", carry="hands", half=4.0))
register(Toy("plane", "Самолётик", carry="hands", half=4.5))
register(Toy("boomerang", "Бумеранг", carry="hands", half=5.0))
register(Toy("rod", "Удочка", carry="back", half=6.0, needs="water"))
register(Toy("bicycle", "Велосипед", carry="hands", half=5.5))
register(Toy("marker", "Фломастер", carry="hands", half=4.5))


# Игрушки, с которыми играют на месте (toys/inplace.py): имя -> (модуль, класс голов, класс попытки)
INPLACE = {"blocks": ("toys.blocks", "BlockHeads", "TowerAttempt"),
           "bubbles": ("toys.flying", "BubbleHeads", "BubbleAttempt"),
           "plane": ("toys.flying", "PlaneHeads", "PlaneAttempt"),
           "boomerang": ("toys.flying", "BoomerangHeads", "BoomerangAttempt"),
           "rod": ("toys.rod", "RodHeads", "RodAttempt"),
           "bicycle": ("toys.bicycle", "BikeHeads", "RideAttempt"),
           "marker": ("toys.marker", "MarkerHeads", "DrawAttempt")}
SESSIONS = {"rod": ("toys.rod", "RodSession"),        # своя сессия (не просто на месте)
            "bicycle": ("toys.bicycle", "BikeSession")}


def inplace(name):
    """(класс голов, класс попытки) игрушки на месте, или None."""
    import importlib
    spec = INPLACE.get(name)
    if spec is None:
        return None
    mod = importlib.import_module(spec[0])
    return getattr(mod, spec[1]), getattr(mod, spec[2])
