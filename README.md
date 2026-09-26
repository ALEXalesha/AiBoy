<div align="center">

# AiBoy

**A living aquarium run by neural networks. One network invents a 2D world, another invents a little person, and then that person lives in the world: walks, jumps, waves his arms, goes up to trees and mushrooms and says things in a chat. You just watch.**

[Download for Windows](https://github.com/ALEXalesha/AiBoy/releases/latest) &nbsp;·&nbsp; [Русская версия этого файла](README.ru.md)

[![CI](https://github.com/ALEXalesha/AiBoy/actions/workflows/ci.yml/badge.svg)](https://github.com/ALEXalesha/AiBoy/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/ALEXalesha/AiBoy?color=16a34a)](https://github.com/ALEXalesha/AiBoy/releases/latest)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

<img src="docs/preview_observe.png" width="880" alt="Watching: the world, the little person with the 'what he sees' overlay, his chat on top and the panel on the right">

</div>

> **The interface and the little person's speech are in Russian**, and so is the long write-up - [README.ru.md](README.ru.md), which this file summarises. "Наблюдение / Галерея / Статистика / Настройки" are the Watch, Gallery, Statistics and Settings tabs; "Что видит" is the "what he sees" overlay.

The rule of the project: everything AiBoy does comes out of its own neural networks - terrain, sky and ground colours, mountains, clouds, trees, stones and crystals, the person himself, his movements and his phrases. The code only draws, moves the physics and trains. A template speech "teacher" exists only for training at build time; a test checks the sources to make sure the game never imports it. No PyTorch: layers, backpropagation, recurrent nets and Adam are numpy (`nn/`), and every gradient is checked against a numerical one. There is no sound: the owner decided the game is better without it, and the voice was removed together with QtMultimedia.

## Four networks

| Network | What it does | How it learned |
|---|---|---|
| **World** | from a seed: terrain, palette, mountains, clouds, entities (kind, colour, size), the world's name | CPPN with 1,637 weights; selected by evolution for "interestingness", no gradient |
| **Body** | from a seed: proportions, colours, hair style, name; the skeleton is always human, 15 points | the most varied of 300 random generators |
| **Brain** | 30 times a second: targets for 10 joints, which way to go, jump, "say" | starter weights from an evolution strategy at build time, then learns online from curiosity |
| **Speech** | a phrase for the chat from the brain's state | character-level GRU, 121,001 weights, trained on teacher phrases |

**World.** Side view; the world is a seamless ring. A reachability graph (`world/reach.py`) encodes what the body can actually do, measured on the physics: walk up slopes to 0.5 m per metre, jump 0.8 m up and 1 m ahead. A trap is a place from which the rest of the world cannot be reached. The first build judged steps against 0.45 m per 0.5 m, a slope the legs barely manage, so the person got stuck in ravines. Now, on 60 unseen seeds, 89 % of steps are walkable (81 % in the first build, 42 % for a random network), and no world has a trap. As a safeguard, the generator lowers the terrain by 15 % at a time if a trap ever appears; that was needed in 2 of the 60 worlds.

**Walking is not coded.** A foot on the ground does not slip; swinging the standing leg back moves the pelvis. The "go left/right" output only turns him around, and only when the wish lasts 0.3 s.

**Why he kept flipping and twitching.** Measured: 577 turns a minute, and joints reversing direction in 15 % of frames. He had found a hole in the physics: turning around re-planted the feet mirrored, so turning back and forth while pushing moved him faster than steps did. Now the feet stay put when he turns, turns need a sustained wish and a 0.6 s pause, joint commands are smoothed, in-game exploration noise is 0.35 of the training noise, and jerky actions cost reward. Now, measured in 10 unseen worlds for a minute each, as in the game: 0.6 turns a minute, joints reversing in 6.7 % of frames, 90 m from the start per minute, and out of the deepest hollow within 30 s in 10 worlds of 10.

**Curiosity instead of a task.** A predictor guesses what he will *see* one step later; its error, divided by a slow running mean, is the reward. Standing still gets boring. A "pulse" (sine and cosine of the body's clock) is part of the input, a rhythm a gait can use. Twelve minutes of the in-game actor-critic alone left him shuffling on the spot, so the starter actor weights come from an evolution strategy (`train/es.py`). Episodes are scored by how much new ground he saw and whether he climbed out of the deepest hollow, minus turns, twitching, falls and jumps. The chosen compromise still jumps 21 times a minute, more often than I would like; the calmest variant (a jump costing 2 m) did not jump at all but never got out of a hollow. Numbers are in `models/brain.json`. In the game everything keeps learning, the actor at 0.3 of the training step.

**What he sees.** Toggle with V, on by default. Rays to the 11 terrain points the brain receives, brightness = the predictor's error there, rings around the entities in his input, an arrow of the last second's intent, and a plain-words summary. It is built from the same observation vector the brain got, and a test compares the numbers.

**Speech.** Its input is a readable 27-number summary, not the brain's hidden vector (that drifts while the brain learns). On 600 unseen states, 99.5 % of phrases use only the teacher's words, 140 distinct phrases.

## Screens

The game starts on a menu with a short explanation, and the current world stays frozen behind it: life pauses while the menu is open. Then Watch (world, chat on top, panel with curiosity chart, pause, speed x1/x2/x4/x8), Gallery, Statistics and Settings. Esc returns to the menu.

<img src="docs/preview_menu.png" width="880" alt="The start menu over the current world">

## Running it

Windows builds - an installer and a portable zip with Python, numpy and Qt inside - are on the [releases page](https://github.com/ALEXalesha/AiBoy/releases/latest).

From source (Python 3.11+): `pip install -r requirements.txt`, then `python main.py`. Headless self-check: `python main.py --selftest result.txt`. Retrain: `python -m train.world`, `train.body`, `train.speech`, `train.brain`. Build: `python build.py` (needs PyInstaller, Pillow and NSIS).

## Tests

`python -m pytest` - 208 tests, about a minute. They cover gradients, world seeds, the seamless ring and the no-trap law, body limits, physics (the stance leg moves the body, turning on the spot goes nowhere, smooth joints), the brain, speech, the trained models' behaviour (calm, still walking, climbs out of hollows), "no sound anywhere", and the whole window driven without a screen. A mutation check broke the code on purpose 20 times (8 in the first build, 12 for the owner's fixes); four of the new breaks first survived, the tests were strengthened, and now every one turns them red.

## Licence

MIT, see [LICENSE](LICENSE).
