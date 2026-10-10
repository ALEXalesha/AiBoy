<div align="center">

# AiBoy

**A living aquarium run by neural networks. One network invents a 2D world, another invents a little person, and then that person lives in the world: walks, jumps, waves his arms, goes up to trees and mushrooms, says things in a chat, teaches himself the guitar, takes toys out of a chest, swims, tames a dog, climbs mountains and plants a flag on the summit. He learns all of it on his own, from scratch. You just watch and listen.**

[Download for Windows](https://github.com/ALEXalesha/AiBoy/releases/latest) &nbsp;·&nbsp; [Русская версия этого файла](README.ru.md)

[![CI](https://github.com/ALEXalesha/AiBoy/actions/workflows/ci.yml/badge.svg)](https://github.com/ALEXalesha/AiBoy/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/ALEXalesha/AiBoy?color=16a34a)](https://github.com/ALEXalesha/AiBoy/releases/latest)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

<img src="docs/preview_observe.png" width="880" alt="Watching: the world, the little person with the 'what he sees' overlay, his chat on top and the panel on the right">

</div>

> **The interface and the little person's speech are in Russian**, and so is the long write-up - [README.ru.md](README.ru.md), which this file summarises. "Наблюдение / Галерея / Статистика / Настройки" are the Watch, Gallery, Statistics and Settings tabs; "Что видит" is the "what he sees" overlay.

The rule of the project: everything AiBoy does comes out of its own neural networks - terrain, sky and ground colours, mountains, clouds, trees, stones and crystals, the person himself, his movements and his phrases. The code only draws, moves the physics and trains. A template speech "teacher" exists only for training at build time; a test checks the sources to make sure the game never imports it. No PyTorch: layers, backpropagation, recurrent nets and Adam are numpy (`nn/`), and every gradient is checked against a numerical one. He has no voice: the owner decided the game is better without it. Since 1.1.0 only toys make sound: the guitar he learns to play on his own and, since 2.0, synthesised world events (hits, splashes, barking, a fanfare, birdsong).

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

## The guitar (1.1.0)

The owner is learning the guitar himself and wanted the little person to learn too, and to be able to hear him getting better.

**When he plays.** A guitar lies in the world 1.5 m from where he appears. He picks it up on his way, carries it on his back and takes it into new worlds. He sits down to play when the world has become more boring than music, not on a timer. The world's interest is a slow average of his predictor's error. Measured on different bodies, it is 0.0023-0.0031 while walking and falls to 0.0011 after 30 s of standing and 0.0004 after a minute. A setting shifts the threshold (rarely / sometimes / often). He plays at least 3 attempts in a row and spends at least 20 s in the world between sessions.

<img src="docs/preview_guitar.png" width="880" alt="He plays: the neck with his fingers and the notes above the world, the little person with a guitar on a stump">

<img src="docs/preview_guitar_close.png" width="600" alt="Close-up: sitting on a stump with the guitar, one hand at the neck, the other at the strings">

**Sound.** The strings are synthesised in numpy with Karplus-Strong, using a fractional delay in the loop, and have a guitar body. The pitch is within 0.01 cent on all 6 strings and 13 frets; the measurement itself reads a string detuned by 0.3 % as 5.19 cents, which is correct. The decay is clean, and the peak after the limiter stays at 0.95 or below. Output goes through QtMultimedia on the Windows backend, without ffmpeg. At volume 0 the audio device is never opened.

**The musician** is his own GRU (96 hidden). Every eighth note it decides whether to strike, where on the neck (one head over 78 string-and-fret places), how hard and whether to mute. The reward is A + 0.5·B + 2·C:

- **A** is how close the take is to the tune: a note-by-note alignment that penalises wrong pitches and time shifts.
- **B** is how pleasant it sounds: consonance, a steady rhythm, novelty against his last 8 phrases, and penalties for silence and mush.
- **C** is your 👍/👎. Without a rating, a small taste model guesses it, but the guess only counts once the model has earned trust: it must first guess 5 of your ratings, and after that it counts more with the number of ratings and its accuracy.

The trust rule came from a measurement: eight random ratings in 300 attempts had driven the similarity to the tune down to zero. A rating is also clipped to 5 reward spreads, so that one rating does not drown out the similarity signal for the next 20 attempts. Without ratings the clip changes nothing.

**Songbook.** Ten public-domain tunes, each split into 4-bar phrases, plus your own from the «Научить мелодии» (teach a tune) screen: play it with the mouse or the Z-M / Q-U keys, name it, and he learns it first. Every fourth attempt is free play. «Кузнечик» ("The Grasshopper") from the task is not included, because its music (Shainsky, 1972) is not in the public domain, so the measurements use «Чижик-пыжик».

<img src="docs/preview_teach.png" width="880" alt="Teach a tune: a two-octave keyboard, the recorded notes, a name">

**How fast he learns.** The first phrase of «Чижик-пыжик» (14 notes), a musician from scratch with no ratings. Median similarity A per 100-attempt window, seeds 0-2:

| Attempts | 1-100 | 201-300 | 401-500 | 901-1000 |
|---|---|---|---|---|
| similarity A | 0.13-0.22 | 0.40-0.51 | 0.63-0.73 | 0.72-0.83 |

One attempt is 8.7 s of playing. The difference becomes audible after about 100 attempts (about 15 minutes), and by attempt 500 (about 75 minutes) the phrase is recognisable. With seed 1 it was learned (A = 0.85) at attempt 1048.

Listen to the samples below. Each file plays the attempt first and then, after a second, the tune played exactly, for comparison:

- [`docs/audio/chizhik-01-first.wav`](docs/audio/chizhik-01-first.wav) - the first attempt, A = 0.14
- [`docs/audio/chizhik-02-after-300.wav`](docs/audio/chizhik-02-after-300.wav) - attempt 300, A = 0.55
- [`docs/audio/chizhik-03-learned.wav`](docs/audio/chizhik-03-learned.wav) - attempt 1048, learned, A = 0.85

**Modes.** In «Музыкант» (musician mode) the chosen note sounds and the hands only show it. In «Руками» (by hand) a note sounds only if his hands actually reach the fret and the string. The hands learn from their own experience: they hit 5 % of notes over the first 100 attempts and 41 % by attempt 600, against a geometric ceiling of 74 %.

**While he plays** life runs at x1 so every attempt can be heard; a setting keeps x2-x8, and then he plays silently. The Watch screen shows the neck with his fingers, the notes against the tune, and 👍/👎. The chat shows his music phrases; the speech network was retrained for them.

**«Попытки»** (attempts) lists every attempt with «Слушать» (listen), «Сравнить с первой» (compare with the first attempt at the same tune), 👍/👎 and a chart of how the similarity grows.

<img src="docs/preview_attempts.png" width="880" alt="Attempts: the list and a chart of similarity to Twinkle Twinkle growing from 0.1 to 0.5 over 225 attempts">

## 2.0: a bigger world

The owner wanted the little person to have more to do than the guitar, to choose for himself, and to learn everything himself. Every new skill starts as a clumsy beginner, and a learning law in the tests measures that it gets better. All numbers below are measurements; the full write-up is in the plan, `docs/superpowers/plans/2026-09-27-aiboy-2.md` (in Russian).

**Fixed from 1.1.0.**

- **Stopping the old world.** Changing the world used to leave a guitar attempt playing for up to 9 s. Now it fades out in 25 ms, and the old life takes no further step.
- **Turning.** He decides where to go with a navigator network fed by a memory of places. Over 12 minutes in each of the same 10 worlds, he dithered back and forth zero times.
- **Boredom.** The guitar gets boring now: there is one boredom for every activity. Half an hour used to be 4 guitar sessions of 27 minutes; now it is 16-22 sessions with a median of 2-3 minutes.
- **Requests.** «Положи» (put it down) is a request he may turn down.

**Fixed after the 2.0 review.**

- «Предложить» (offer) works for every toy: when he agrees, he plays with it at once.
- «Положить ...» (put down) shows for whatever he holds, and only while he holds it; in the middle of a game he is harder to talk into it.
- Walking left, the whole figure is mirrored, feet included.
- Deleting the current world in the gallery and entering it starts a new world, and it shows in the gallery.
- The bicycle no longer sinks into a steep slope (a wheel used to slip up to 0.88 m into the hill).
- He loves the guitar less: half its pull goes in 3.5 minutes instead of 5, and it pulls weaker at the same liking.

**A chest and eight toys** run on pymunk physics. Each has its own network head, journal and learning law. Median success, first attempts against later ones:

| Toy | First | Later |
|---|---|---|
| ball, kick at a target | 0.06-0.07 | 0.63-0.72 |
| blocks, height of a standing tower | 0.33-0.67 | 0.83 |
| bubbles, size without bursting | 0 | 1.0 |
| paper plane, distance / 40 m | 0.29 | 0.58 |
| boomerang, came back | 0.03 | 0.35-0.40 |
| fishing rod, caught a fish | 0.05-0.14 | 0.25-0.34 |
| bicycle, got there without falling | 0.01-0.05 | 0.87-1.0 |
| marker, beauty by his own taste | 0.76-0.83 | 0.96 |

<img src="docs/preview_toys.png" width="880" alt="Six frames: blocks, bubbles, paper plane, boomerang, bicycle, marker">

**Needs, water, swimming.** Hunger, thirst and tiredness are urgencies inside his choice of activity, not hard rules. World food tastes good or bad according to his own taste network. About half of the worlds have ponds. He learns to swim from scratch, flailing upright at first; in a 4 m pond he becomes a swimmer after about 11 minutes of play. He never drowns: when he is tired, or has been under water for 2.5 s, he is carried ashore. Fish bite at dawn and dusk.

**Worlds of 640 and 1280 m** get a large-scale relief from an evolved network. The shape of the hills repeats at 0.28, against 0.93 without that network. The old world sizes still give byte-identical worlds. Speed goes up to x16, and it is honest: when a frame can't keep up, the button says so.

**A dog** has its own pymunk body. The first version, with jointed legs, somersaulted on almost any gait, so now a torso rides on spring legs during the stance phase and the legs are drawn by inverse kinematics. It learns to walk (0.23-1.0 -> 1.21-1.22 on flat ground over 100 runs), has its own boredom and needs, and is tamed by patience and treats. A tame dog brings the ball and learns that from praise: the chance it brings the ball rose from 0.16 to 1.0 over 60 throws, and only to 0.13-0.26 without praise.

<img src="docs/preview_dog.png" width="700" alt="A tame dog running after the little person">

**Mountains** of 60-100 m have an easy route (slope 0.22) and a steep face. Every trap the terrain could form was found by measurement and fixed. When he notices a mountain, he chooses for himself:

- walk past;
- take the easy route;
- climb the face, with bare hands, a rope and hook, or an ice axe from the chest.

Climbing is learned from scratch, by turning the hand to the hold's facet and picking one of the next three holds. A fall slides him down to a ledge without harm. On a 24 m face he covered 0.03-0.06 of the way at first and 0.40-0.64 after 1,600 attempts. On his first summit he plants a flag. It stays in the world after a restart, flutters in the wind and is visible from afar.

<img src="docs/preview_mountain.png" width="880" alt="Climbing the face with an ice axe, the flag and joy on the summit, the conquered mountain with its flag from afar">

**A diving mask** lets him stay under water as long as his air lasts and watch the fish; without it the 2.5 s limit holds. His diving head learns: he saw a fish in 0.35-0.39 of dives at first and 0.43-0.69 after 800 dives.

<img src="docs/preview_dive.png" width="700" alt="A dive with the mask">

**Birds** fly with a two-control flight network (flap, tail). Holding 3.5 m, a beginner misses by 0.7-4 m and, after 300 flights, by 0.12-0.48 m. Flocks land, take off when someone comes near and learn to take off in time. They never land next to anyone, so nobody can catch a bird, and a test checks that.

<img src="docs/preview_birds.png" width="700" alt="A flock of birds above the little person at night">

**Screens 2.0:**

- «Каталог» (catalogue) shows the toys and activities, with «Предложить» (offer). Whether he agrees is up to him.
- «Попытки» (attempts) covers every activity and has a replay.
- The gallery has worlds, drawings and catch.
- Statistics show the time spent on each activity.
- Settings add «Собака» (dog), «Птицы» (birds) and the two large world sizes.

**Speech** knows 10 new states. 0.987 of his phrases use only the teacher's words, and 203 are distinct.

**Event sounds** are synthesised: [`world2-dog.wav`](docs/audio/world2-dog.wav), [`world2-mountain.wav`](docs/audio/world2-mountain.wav), [`world2-water-sky.wav`](docs/audio/world2-water-sky.wav).

<img src="docs/preview_catalog.png" width="880" alt="The catalogue: needs and backpack, toy cards with icons">

## Screens

The game starts on a menu with a short explanation, and the current world stays frozen behind it: life pauses while the menu is open. Then Watch (world, chat on top, panel with curiosity chart, pause, speed x1/x2/x4/x8), Gallery, «Попытки» (attempts), «Научить мелодии» (teach a tune), Statistics and Settings (including volume and the guitar mode). Esc returns to the menu.

<img src="docs/preview_menu.png" width="880" alt="The start menu over the current world">

## Running it

Windows builds - an installer and a portable zip with Python, numpy and Qt inside - are on the [releases page](https://github.com/ALEXalesha/AiBoy/releases/latest).

From source (Python 3.11+, numpy, PySide6, pymunk 7.3): `pip install -r requirements.txt`, then `python main.py`. Headless self-check: `python main.py --selftest result.txt`. Besides living 300 steps and speaking, it checks that a string sounds at its pitch, that one guitar attempt is played, scored and saved, and, since 2.0, pymunk physics, a 1280 m world with a mountain, a dog and a flock living for 10 s, a block tower and the event sounds. Retrain: `python -m train.world`, `train.body`, `train.speech`, `train.brain`, and for 2.0 `train.navigator`, `train.macro`, `train.water`, `train.mountains`. Build: `python build.py` (needs PyInstaller, Pillow and NSIS). Screenshots: `python tools\make_previews.py`; sound samples: `python tools\make_audio.py` (2.0 event sounds: `--events`).

## Tests

`python -m pytest` - 627 tests (334 in 1.1.0, 208 in 1.0.0); the learning laws and the 30-minute session laws make a full run take about an hour. They cover gradients, world seeds, the seamless ring and the no-trap law, body limits, physics (the stance leg moves the body, turning on the spot goes nowhere, smooth joints), the brain, speech, the trained models' behaviour (calm, still walking, climbs out of hollows), "no voice" (only the guitar uses QtMultimedia), the guitar, and the whole window driven without a screen. A mutation check broke the code on purpose 20 times (8 in the first build, 12 for the owner's fixes); four of the new breaks first survived, the tests were strengthened, and now every one turns them red. The guitar added 21 more deliberate breaks (a detuned string, no limiter, free extra notes, no silence penalty, a taste blind to similarity, a journal losing its first attempts, mood and hands rules, x8 while playing, the walking brain stepping while he plays, trusting the taste blindly, a self-check that does not save, a guitar that is not drawn): three survived at first, the tests were extended, and all 21 are caught now. The guitar tests include a learning law: over 200 attempts the median similarity of the last 20 beats the first 20 by at least 0.2. Version 2.0 added deliberate breaks for each task:

- stopping the world: 10 of 11 caught, and the survivor is equivalent;
- pymunk ground: 4 of 4;
- toys: 14 of 14;
- dog: 18 of 18;
- mountains and mask: 19 of 20, and the survivor is equivalent;
- birds: 10 of 10;
- gallery cards: 2 of 2.

## Licence

MIT, see [LICENSE](LICENSE).
