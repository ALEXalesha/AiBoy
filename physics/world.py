"""Пространство pymunk одной жизни: рельеф, предметы, кисти и стопы человечка.

- **Рельеф** - статическая ломаная по высотам мира (шаг 0.25 м, как у самого мира), но
  не вся: мир до 1280 м, а нужна она только там, где что-то движется. Строится окно
  WINDOW метров вокруг центра (камеры, человечка) и пересобирается, когда центр уходит
  дальше REBUILD от середины окна. Координаты - «развёрнутые»: x растёт без кольца, высота
  берётся по модулю ширины мира, так что через шов кольца всё катится ровно.
- **Предметы** (мяч, кубики, велосипед, поплавок, собака) - динамические тела. Предмет за
  окном не падает сквозь пустоту: он замирает (выходит из пространства) и возвращается,
  когда окно снова до него дойдёт, - дешёвый шаг далёких вещей.
- **Кисти и стопы** - кинематические тела: их двигает тело человечка, pymunk считает
  только удары о предметы.
- Шаг 1/60 с делится на SUBSTEPS: сильный удар ногой (до 25 м/с) иначе проходил бы
  тонкий отрезок рельефа насквозь.
"""
import math

import pymunk

GRAVITY = -9.81
WINDOW = 80.0            # м - окно рельефа вокруг центра
REBUILD = 20.0           # м - центр ушёл дальше - окно пересобирается
SEG_R = 0.05             # м - толщина отрезков рельефа
SUBSTEPS = 4
FRICTION = 0.9
MAX_SPEED = 40.0         # м/с - предела быстрее не бывает (защита от взрыва решателя)

GROUND, THING, LIMB = 1, 2, 4   # категории столкновений


class PhysWorld:
    def __init__(self, world, center=0.0):
        self.world = world
        self.space = pymunk.Space()
        self.space.gravity = (0.0, GRAVITY)
        self.space.iterations = 20
        self.segments = []
        self.center = None
        self.things = {}          # имя -> (тело, фигуры)
        self.frozen = {}          # имя -> (позиция, угол) - предмет за окном
        self.limbs = {}
        self.forces = []          # f(dt) перед каждым подшагом: вода, ветер (pymunk сбрасывает силы после шага)
        self.build(center)

    # --- рельеф ---
    def ground(self, x):
        return self.world.height_at(x)

    def build(self, center):
        """Окно рельефа вокруг center: отрезки через шаг мира."""
        for s in self.segments:
            self.space.remove(s)
        self.segments = []
        dx = getattr(self.world, "dx", 0.25)
        n = int(WINDOW / dx)
        x0 = math.floor((center - WINDOW / 2) / dx) * dx
        pts = [(x0 + i * dx, self.ground(x0 + i * dx)) for i in range(n + 1)]
        body = self.space.static_body
        for a, b in zip(pts, pts[1:]):
            s = pymunk.Segment(body, a, b, SEG_R)
            s.friction = FRICTION
            s.elasticity = 0.4
            s.filter = pymunk.ShapeFilter(categories=GROUND)
            self.segments.append(s)
        # соседние отрезки гладко стыкуются: без этого шар цепляется за стыки
        for s1, s2 in zip(self.segments, self.segments[1:]):
            s1.set_neighbors(s1.a, s2.b)
        self.space.add(*self.segments)
        self.center = center
        self.span = (pts[0][0], pts[-1][0])

    def follow(self, center):
        """Держать окно рельефа вокруг center; предметы за окном - замирают, в окне - оживают."""
        if self.center is None or abs(center - self.center) > REBUILD:
            self.build(center)
        lo, hi = self.span[0] + 2.0, self.span[1] - 2.0
        for name, (body, shapes) in list(self.things.items()):
            if not lo <= body.position.x <= hi:
                self.frozen[name] = (tuple(body.position), body.angle, body, shapes)
                self.space.remove(body, *shapes)
                del self.things[name]
        for name, (pos, ang, body, shapes) in list(self.frozen.items()):
            if lo <= pos[0] <= hi:
                body.position, body.angle = pos, ang
                body.velocity, body.angular_velocity = (0.0, 0.0), 0.0
                self.space.add(body, *shapes)
                self.things[name] = (body, shapes)
                del self.frozen[name]

    # --- предметы ---
    def _add(self, name, body, shapes, friction=FRICTION, elasticity=0.5):
        for s in shapes:
            s.friction = friction
            s.elasticity = elasticity
            s.filter = pymunk.ShapeFilter(categories=THING)
        self.space.add(body, *shapes)
        self.things[name] = (body, shapes)
        return body

    def add_ball(self, name, x, y, radius=0.11, mass=0.4, elasticity=0.7):
        body = pymunk.Body(mass, pymunk.moment_for_circle(mass, 0, radius))
        body.position = (x, y)
        return self._add(name, body, [pymunk.Circle(body, radius)], elasticity=elasticity)

    def add_box(self, name, x, y, w, h, mass=0.3, angle=0.0):
        body = pymunk.Body(mass, pymunk.moment_for_box(mass, (w, h)))
        body.position, body.angle = (x, y), angle
        return self._add(name, body, [pymunk.Poly.create_box(body, (w, h), radius=0.005)], elasticity=0.05)

    def remove(self, name):
        if name in self.things:
            body, shapes = self.things.pop(name)
            self.space.remove(body, *shapes)
        self.frozen.pop(name, None)

    def body(self, name):
        if name in self.things:
            return self.things[name][0]
        f = self.frozen.get(name)
        return f[2] if f else None

    def position(self, name):
        if name in self.things:
            return tuple(self.things[name][0].position)
        f = self.frozen.get(name)
        return f[0] if f else None

    # --- кисти и стопы ---
    def limb(self, name, radius=0.05):
        if name not in self.limbs:
            body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
            shape = pymunk.Circle(body, radius)
            shape.friction = FRICTION
            shape.elasticity = 0.3
            shape.filter = pymunk.ShapeFilter(categories=LIMB, mask=THING)
            self.space.add(body, shape)
            self.limbs[name] = body
        return self.limbs[name]

    def move_limb(self, name, pos, dt):
        """Кисть или стопа к точке pos за dt: скорость кинематического тела - чтобы удар передал
        предмету именно её."""
        body = self.limb(name)
        x, y = body.position
        body.velocity = ((pos[0] - x) / dt, (pos[1] - y) / dt)

    # --- шаг ---
    def step(self, dt):
        h = dt / SUBSTEPS
        for _ in range(SUBSTEPS):
            for f in self.forces:
                f(h)
            self.space.step(h)
        for body, _ in self.things.values():
            vx, vy = body.velocity
            v = math.hypot(vx, vy)
            if v > MAX_SPEED:
                body.velocity = (vx * MAX_SPEED / v, vy * MAX_SPEED / v)
