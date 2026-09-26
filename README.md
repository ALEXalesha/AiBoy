<div align="center">

# AiBoy

**A living aquarium run by neural networks. One network invents a 2D world, another invents a little person, and then that person lives in the world: walks, jumps, waves his arms, goes up to trees and mushrooms, says things in a chat and hums quietly. You just watch.**

[Download for Windows](https://github.com/ALEXalesha/AiBoy/releases/latest) &nbsp;·&nbsp; [Русская версия этого файла](README.ru.md)

[![CI](https://github.com/ALEXalesha/AiBoy/actions/workflows/ci.yml/badge.svg)](https://github.com/ALEXalesha/AiBoy/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/ALEXalesha/AiBoy?color=16a34a)](https://github.com/ALEXalesha/AiBoy/releases/latest)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

<img src="docs/preview_observe.png" width="880" alt="Watching: the world, the little person with his thoughts above the head, his chat on top and the panel on the right">

</div>

> **The interface and the little person's speech are in Russian**, and so is the long write-up - [README.ru.md](README.ru.md), which this file summarises. In the screenshot, "Наблюдение / Галерея / Статистика / Настройки" are the Watch, Gallery, Statistics and Settings tabs; "Любопытство" is the curiosity chart; "Новый мир" / "Новый человечек" are "new world" / "new person".

The rule of the project: everything AiBoy does comes out of its own neural networks - terrain, sky and ground colours, mountains, clouds, trees, stones and crystals, the person himself, his movements, phrases and sounds. The code only draws, moves the physics and trains. Template "teachers" exist, but only for training at build time; a test checks the sources to make sure the game never imports them. No PyTorch: layers, backpropagation, recurrent nets and Adam are numpy (`nn/`), and every gradient is checked against a numerical one.

## Five networks

| Network | What it does | How it learned |
|---|---|---|
| **World** | from a seed: terrain, palette, mountains, clouds, entities (kind, colour, size), the world's name | CPPN with 1,637 weights; weights selected by evolution for "interestingness", no gradient |
| **Body** | from a seed: proportions, colours, hair style, name; the skeleton is always human, 15 points | the most varied of 300 random generators |
| **Brain** | 30 times a second: targets for 10 joints, which way to go, jump, "say", "sound" | learns online, while you watch, from curiosity |
| **Speech** | a phrase for the chat from the brain's state | character-level GRU, 121,001 weights, trained at build time on teacher phrases |
| **Voice** | pitch, melody, vowel and length of a sound | dense 27 → 24 → 7 plus a numpy synthesizer |

**World.** Side view. The world is a ring: the network's input is the coordinate as sines and cosines with a whole number of periods per world width, so the right edge joins the left without a seam. Evolution (48 genomes, 300 generations, new seeds every generation, 6 minutes) scores terrain spread, walkability, sky/ground contrast and entities. On 60 unseen seeds the evolved net scores 0.98 against 0.54 for a random one; 95 % of 0.5 m steps are walkable (worst world 90 %) against 54 %, and there are no unjumpable walls, against 25.6 per 100 m.

**Walking is not coded.** A foot standing on the ground does not slip; when the brain swings the standing leg back, the pelvis moves forward. That push is the only way to move. The "go left/right" output only turns him around.

**Curiosity instead of a task.** A predictor guesses what the person will *see* one step later (the terrain around and the entities); its error is the reward, divided by its running mean. His own body is easy to predict, a new place is not, so curiosity pulls him to places he has not been. Plus 0.015 per step on his feet, minus 1 for a fall. Actor-critic, recurrent layer with one-step truncation, under a millisecond per brain step with learning. The starter brain in `models/brain.npz` comes from 12 minutes of the same online learning at build time: the best of 15 snapshots, taken after 360 worlds (666,000 brain steps, 6.2 hours of life). In 10 unseen worlds it gets 62 m away from its start in a minute, against 3.0 m before training, and it keeps learning - and changing - in the game.

**Speech.** Its input is not the brain's hidden vector (that drifts while the brain learns) but a readable 27-number summary: hills and pits on each side, the nearest entity with its kind, side, distance and colour, wishes, curiosity, a recent fall, night. On 600 unseen states, 99.5 % of phrases use only the teacher's words, with 140 distinct phrases.

**Voice.** Harmonics shaped by vowel formants, notes only from the pentatonic scale, soft envelopes, peak 0.55: a fall sounds low and falling, curiosity rises, a jump is short and high. At most one sound every 5-16 seconds.

## Screens

Watch (world, chat on top, panel with curiosity chart, pause, speed x1/x2/x4, new world, new person), Gallery (every world is saved as seeds, opened again or removed after a confirmation), Statistics (counts, the brain's learning curve, most frequent words, most interesting worlds), Settings (volume, phrase frequency, default speed, thoughts above the head, dark/light theme, world size). The window reopens where it was closed.

<img src="docs/preview_worlds.png" width="880" alt="Six worlds of different seeds from the same network">

## Running it

Windows builds - an installer and a portable zip with Python, numpy and Qt inside - are on the [releases page](https://github.com/ALEXalesha/AiBoy/releases/latest).

From source (Python 3.11+): `pip install -r requirements.txt`, then `python main.py`. Headless self-check: `python main.py --selftest result.txt`. Retrain: `python -m train.world`, `train.body`, `train.speech`, `train.voice`, `train.brain`. Build: `python build.py` (needs PyInstaller, Pillow and NSIS).

## Tests

`python -m pytest` - 161 tests, about 35 seconds: gradients, world seeds and the seamless ring, body limits and symmetry, physics (the stance leg moves the body, walls, jumps, falls, no NaN), the brain (the predictor's error falls, he moves in part of the worlds), speech and voice, the trained models, and the whole window driven without a screen. A mutation check broke the code on purpose eight times and the tests went red every time.

## Licence

MIT, see [LICENSE](LICENSE).
