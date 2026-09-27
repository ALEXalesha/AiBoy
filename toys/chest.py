"""Сундук и где сейчас каждая игрушка.

Сундук стоит у места, где он появился (CHEST_AHEAD м впереди). В начале каждого мира все
игрушки - в сундуке (и гитара, которую в 1.1.0 он носил с собой: так решила спецификация
2.0 при переносе сохранений). Игрушка может быть:
- "chest" - в сундуке;
- "carried" - у него (за спиной или в руках, по реестру); одна за раз;
- число x - лежит на земле в этом месте мира (он её положил).
"""
from toys import registry

CHEST_AHEAD = 1.5          # м - где сундук от места появления
REACH = 0.8                # м - ближе - дотянулся до сундука или игрушки на земле


class Inventory:
    def __init__(self, chest_x, width, toys=None):
        self.chest_x = float(chest_x)
        self.width = float(width)
        self.where = {name: "chest" for name in (toys if toys is not None else registry.names())}

    @property
    def carried(self):
        return next((n for n, w in self.where.items() if w == "carried"), None)

    def dist(self, a, b):
        return abs((a - b + self.width / 2) % self.width - self.width / 2)

    def position(self, name):
        """Куда идти за игрушкой: к сундуку, к месту на земле или None (она у него)."""
        w = self.where.get(name)
        if w == "chest":
            return self.chest_x
        if isinstance(w, float):
            return w
        return None

    def on_ground(self):
        return {n: w for n, w in self.where.items() if isinstance(w, float)}

    def take(self, name, x):
        """Взять игрушку, стоя в x (у сундука или у неё). True - взял."""
        pos = self.position(name)
        if pos is None or self.dist(pos, x) > REACH or self.carried not in (None, name):
            return False
        self.where[name] = "carried"
        return True

    def put(self, name, x):
        """Положить игрушку на землю в x."""
        if self.where.get(name) != "carried":
            return False
        self.where[name] = float(x)
        return True

    def stow(self, name):
        """Убрать в сундук (он у сундука, или мир сменился)."""
        self.where[name] = "chest"
