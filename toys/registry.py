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
