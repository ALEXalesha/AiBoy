"""Мозг человечка: рекуррентная сеть, которая учится на ходу любопытством.

Каждый шаг (30 раз в секунду) по наблюдению и своему прошлому состоянию она выдаёт:
цели 10 суставов, желание идти влево или вправо, прыжок и «сказать».

Учится прямо во время жизни, актёр-критик:

- **предсказатель** по наблюдению и действию угадывает, что изменится через шаг. Его
  ошибка - награда любопытства: своё тело он скоро предсказывает хорошо, а новое место -
  нет, поэтому любопытство тянет туда, где ещё не был. Ошибка делится на её скользящее
  среднее, чтобы награда не угасала вместе с ошибкой;
- плюс 0.015 за каждый шаг на ногах (на земле, не лёжа) и минус 1 за падение;
- минус JERK * средний квадрат смены действия - за рывки: без этого мозг выучивал дёргать
  суставы и направление туда-сюда, 15 % кадров сустав менял направление движения;
- **критик** оценивает состояние, TD(0); **актёр** - гауссова политика суставов и
  направления (шум коррелирован во времени, суставы движутся плавно) и Бернулли прыжка;
- рекуррентный слой учится с усечением на один шаг: прошлое состояние для градиента -
  просто вход.

Голова «сказать» - выход той же сети, но награда в неё не идёт: любопытство от разговоров
не зависит, и обучение свело бы её к нулю или к единице.
"""
import math
from dataclasses import dataclass

import numpy as np

from body import skeleton
from brain.observe import OBS_DIM, WORLD, observe
from nn import io
from nn.layers import Dense, ReLU, Sequential, sigmoid
from nn.losses import mse
from nn.optim import Adam, clip_grads
from nn.rnn import RNNCell

HIDDEN = 64
N_ACT = len(skeleton.JOINTS) + 1            # суставы и направление
GAMMA = 0.95
SIGMA = 0.22
RHO = 0.95                                  # корреляция шума между шагами (~0.7 с)
CUR_SCALE = 0.1
JERK = 0.05                                 # цена рывка: средний квадрат смены действия за шаг
PLAY_NOISE = 0.35                           # в игре шум исследования в столько раз меньше
PLAY_LR = 0.3                               # и шаг обучения меньше: походку не расшатать
ALIVE = 0.015                               # за шаг на ногах - это «устойчивость»
FALL_PENALTY = -1.0
LR = 3e-4
LR_PRED = 1e-3
PHYSICS_PER_STEP = 2                        # физика 60 Гц, мозг 30 Гц


@dataclass
class Decision:
    targets: np.ndarray        # углы суставов, рад
    turn: float                # -1 влево .. 1 вправо
    jump: bool
    want_jump: float           # вероятность прыжка
    say: float                 # вероятность «сказать»


class Brain:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.rng = np.random.default_rng(seed + 1000)
        self.cell = RNNCell(OBS_DIM, HIDDEN, rng)
        self.pi = Dense(HIDDEN, N_ACT, rng, gain=0.3)
        rest = np.clip(skeleton.to_unit(skeleton.rest_angles()), -0.95, 0.95)
        self.pi.b[:len(rest)] = np.arctanh(rest)
        self.jump = Dense(HIDDEN, 1, rng, gain=0.3)
        self.jump.b[:] = -4.5
        self.value = Dense(HIDDEN, 1, rng, gain=0.3)
        self.talk = Dense(HIDDEN, 1, rng, gain=0.5)
        self.talk.b[:] = -2.5
        self.pred = Sequential([Dense(OBS_DIM + N_ACT + 1, 96, rng), ReLU(), Dense(96, OBS_DIM, rng, gain=0.5)])
        self.learners = [self.cell, self.pi, self.jump, self.value]
        self.opt = Adam([p for layer in self.learners for p in layer.params()], lr=LR)
        self.learn_actor = True     # False - учатся только предсказатель и критик
        self.opt_pred = Adam(self.pred.params(), lr=LR_PRED)
        self.h = np.zeros(HIDDEN, np.float32)
        self.noise = np.zeros(N_ACT, np.float32)
        self.prev = None
        self.last_action = None
        self.last_jerk = 0.0
        self.steps = 0
        self.err_ema = None
        self.last_err = 0.0
        self.last_seen_err = 0.0
        self.reward_ema = 0.0
        self.curiosity = 0.0

    def nets(self):
        return {"cell": self.cell, "pi": self.pi, "jump": self.jump, "value": self.value,
                "talk": self.talk, "pred": self.pred}

    def params(self):
        return [p for layer in self.nets().values() for p in layer.params()]

    # --- прямой проход без запоминания: для решения ---
    def _hidden(self, obs, h):
        return np.tanh(obs @ self.cell.Wx + h @ self.cell.Wh + self.cell.b)

    def _heads(self, h):
        mu = np.tanh(h @ self.pi.W + self.pi.b)
        pj = float(sigmoid(h @ self.jump.W + self.jump.b)[0])
        talk = sigmoid(h @ self.talk.W + self.talk.b)
        return mu, pj, talk

    def decide(self, obs, explore=True):
        """Решение по наблюдению из текущего состояния памяти, без обучения и без
        сдвига памяти - для проверок и подписей «мыслей»."""
        h = self._hidden(obs, self.h)
        mu, pj, talk = self._heads(h)
        return self._decision(mu, pj, talk, jump=pj > 0.5)

    @staticmethod
    def _decision(a, pj, talk, jump):
        return Decision(skeleton.from_unit(a[:-1]), float(a[-1]), bool(jump), pj, float(talk[0]))

    # --- шаг жизни ---
    def step(self, obs, extra_reward=0.0, learn=True, explore=True, noise=1.0):
        """Ход мозга: учится на прошлом переходе (если learn) и решает, что делать сейчас.
        noise - доля шума исследования: при обучении 1, в игре PLAY_NOISE, чтобы человечек
        не дёргался; градиент считается с тем шумом, что был на самом деле."""
        obs = np.asarray(obs, np.float32)
        reward = None
        if self.prev is not None:
            reward = self._curiosity(obs, learn) + extra_reward - self.last_jerk
            self.reward_ema = 0.999 * self.reward_ema + 0.001 * reward
        h_new = self._hidden(obs, self.h)
        v_new = float(h_new @ self.value.W[:, 0] + self.value.b[0])
        if learn and self.prev is not None:
            self._learn(reward, v_new)
        mu, pj, talk = self._heads(h_new)
        sigma = SIGMA * noise if explore else 0.0
        if sigma > 0:
            self.noise = RHO * self.noise + math.sqrt(1 - RHO * RHO) * self.rng.normal(size=N_ACT)
            a = np.clip(mu + sigma * self.noise, -1.0, 1.0).astype(np.float32)
            jump = self.rng.random() < pj
        else:
            a, jump = mu.astype(np.float32), (self.rng.random() < pj) if explore else pj > 0.5
        self.last_jerk = 0.0 if self.last_action is None else JERK * float(((a - self.last_action) ** 2).mean())
        self.last_action = a
        self.prev = (obs, self.h, a, float(jump), max(sigma, 1e-3))
        self.h = h_new.astype(np.float32)
        self.steps += 1
        return self._decision(a, pj, talk, jump), reward

    def _curiosity(self, obs, learn):
        p_obs, _, p_act, p_jump, _ = self.prev
        pin = np.concatenate([p_obs, p_act, [p_jump]]).astype(np.float32)[None]
        pred = self.pred.forward(pin)
        target = (obs - p_obs)[None]
        err, grad = mse(pred, target)
        self.last_err_vec = ((pred - target)[0] ** 2).astype(np.float64)
        seen = float(self.last_err_vec[WORLD].mean())
        if learn:
            self.pred.backward(grad)
            clip_grads(self.pred.params(), 1.0)
            self.opt_pred.step()
        self.last_err = err
        self.last_seen_err = seen
        # награда - ошибка в том, что он увидит (рельеф и сущности), а не в своём теле:
        # тело он и так знает, а новое место - нет. Среднее для нормировки медленное (~30 с):
        # застрял на одном месте - ошибка ниже привычной, награда падает, становится
        # скучно, и выгоднее уйти (развернуться, прыгнуть)
        self.err_ema = seen if self.err_ema is None else 0.999 * self.err_ema + 0.001 * seen
        ratio = min(seen / (self.err_ema + 1e-8), 3.0)
        self.curiosity = 0.9 * self.curiosity + 0.1 * min(ratio / 2.0, 1.0)
        return CUR_SCALE * ratio

    def _learn(self, reward, v_new):
        p_obs, p_h, p_act, p_jump, sigma = self.prev
        h = self.cell.forward(p_obs[None], p_h[None])
        mu = np.tanh(self.pi.forward(h))
        pj = sigmoid(self.jump.forward(h))
        v = self.value.forward(h)
        delta = float(reward + GAMMA * v_new - v[0, 0])
        # без шума действие - само среднее, учить актёра нечему (градиент был бы бесконечным)
        dmu = -delta * (p_act[None] - mu) / (sigma * sigma) if sigma > 0.01 else np.zeros_like(mu)
        dh = self.pi.backward(dmu * (1.0 - mu * mu))
        dh = dh + self.jump.backward(-delta * (p_jump - pj))
        dh = dh + self.value.backward(np.array([[-delta]], np.float32))
        self.cell.backward(dh.astype(np.float32))
        if not self.learn_actor:
            # актёр и общий слой заморожены: учится только оценка состояния
            for layer in (self.cell, self.pi, self.jump):
                for _, _, g in layer.params():
                    g[...] = 0
        clip_grads([p for layer in self.learners for p in layer.params()], 1.0)
        self.opt.step()

    def reset_memory(self):
        """Новый мир или новый человечек: память и прошлый переход - с нуля, веса - те же."""
        self.h[:] = 0
        self.noise[:] = 0
        self.prev = None
        self.last_action = None
        self.last_jerk = 0.0

    # --- файл ---
    def save(self, path):
        arrays = io.collect(self.nets())
        arrays["meta.h"] = self.h
        arrays["meta.numbers"] = np.array([self.steps, self.err_ema or 0.0, self.reward_ema], np.float64)
        io.save(path, arrays)

    def load(self, path):
        arrays = io.load(path)
        io.restore(self.nets(), arrays)
        self.h = arrays["meta.h"].astype(np.float32).reshape(HIDDEN)
        steps, err, rew = arrays["meta.numbers"]
        self.steps, self.err_ema, self.reward_ema = int(steps), float(err) or None, float(rew)
        self.prev = None
        for layer in self.learners + [self.pred]:
            for _, p, _ in layer.params():
                if not np.isfinite(p).all():
                    raise ValueError("в весах мозга не числа")


def run_steps(brain, human, world, steps, learn=True, explore=True, noise=1.0):
    """Жизнь без окна: мозг решает, физика двигает. Для обучения и проверок."""
    log = {"pred_err": [], "reward": [], "events": []}
    extra = 0.0
    for _ in range(steps):
        d, reward = brain.step(observe(human, world), extra, learn=learn, explore=explore, noise=noise)
        if reward is not None:
            log["pred_err"].append(brain.last_err)
            log["reward"].append(reward)
        events = []
        for k in range(PHYSICS_PER_STEP):
            events += human.step(d.targets, d.turn, d.jump and k == 0)
        extra = FALL_PENALTY if "fall" in events else (ALIVE if human.grounded and human.fallen <= 0 else 0.0)
        log["events"] += events
    return log
