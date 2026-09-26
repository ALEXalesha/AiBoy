# AiBoy: гитара - спецификация

Date: 2026-09-26
Status: approved by Alexey in a dialogue on 26.09 (answers: "A B C", "1 and 2 selectable in the
settings, add volume there too", tunes "1 and 2", when to play "A", learning approach "1",
"lets go"). Release: AiBoy 1.1.0. The catalogue of other toys (ball and others) is a separate,
later part with its own spec.

## 1. What this is

The man gets a guitar and learns to play it himself - from noise to music, and we hear his
real attempts. "Played better" = three sources of reward at once:

- **A. Repeat a tune** from the songbook (reward for closeness to it);
- **B. Pleasant sound in itself** - consonance, rhythm, novelty (don't repeat
  himself);
- **C. Owner's rating** 👍/👎 on an attempt - the heaviest weight; a small "taste"
  model learns from the ratings and guesses the rating when nobody is around.

The project rule stays: everything he plays is output by his own network (numpy, the layers
in `nn/`). The code only synthesizes the string sound from the notes he chose, counts the
reward and trains. The tunes of the songbook are what he learns, not his output.

## 2. Sound

- **Physical model of a plucked string** (Karplus-Strong with damping and a body filter),
  numpy, 44.1 kHz. A guitar: 6 strings in standard tuning E2 A2 D3 G3 B3 E4, frets 0-12.
  A note = (string, fret, onset, force, duration until damped).
- In "hands" mode (see 5): a miss produces no note, a light touch produces a muted,
  quiet sound.
- **Sound returns only for toys.** His "voice" stays removed. QtMultimedia
  (QAudioSink, one mixer), volume in the settings 0-100% (0 = the device is not opened at
  all). Without a sound card the game stays silent and does not crash. The sound is
  synthesized in batches off the drawing thread and does not slow the window down, including at x8
  (at x8 the sound is not sped up: while he plays the guitar, life runs at 1x,
  - decision; an alternative "attempts at x8 play silently, only the journal" is in the settings).

## 3. Musician network

- A small recurrent network of its own (GRU from `nn/`, as in speech), `music/musician.py`.
  At each 1/8 of a beat it outputs: silence / strike; string (6), fret (13), force (3
  levels), sustain/damp. Input: the previous notes (the last 8), the beat phase and
  the measure, the tune to learn (the next 8 notes of the target, or zeros in free play),
  the step number inside the attempt.
- **An attempt** is a phrase of 5-15 s (4-8 measures, tempo 70-120). After the attempt - a score and
  one update step (REINFORCE with baseline over the whole phrase + entropy against
  getting stuck; Adam). In the game it learns online; there is no pre-training "from scratch" - it starts from
  random weights (a new character - a new musician; the weights live with the character in
  the gallery).
- **Reward of an attempt** `R = wB·B + wA·A + wC·C`:
  - B: consonance of simultaneous and adjacent notes (a table of intervals), steadiness
    of rhythm (onsets on the grid), novelty (distance from the last K own phrases),
    penalty for silence and for "mush" (too dense);
  - A (only when learning a tune): closeness in pitches and rhythm - DTW over the
    notes with a penalty for time shift; an exact repeat = 1;
  - C: the owner's 👍 = +1 / 👎 = -1 for this attempt; without a rating - the taste model's
    prediction (a small logistic network over the features of the attempt); weight wC is the largest.
  The weights are in the settings module, the reward components are recorded in the journal.

## 4. Songbook and «Научить мелодии»

- **10 built-in tunes**, simple and in the public domain (for example «В лесу родилась
  ёлочка», «Кузнечик», «Ода к радости», «Twinkle Twinkle», «Frère Jacques», «Во поле
  берёза стояла», «Jingle Bells», «Happy Birthday» - check the status, otherwise replace it;
  8-16 measures each, one voice), `music/songbook.py`, as notes (pitch, duration).
- **The owner's keyboard:** the «Научить мелодии» screen - piano keys (2 octaves, mouse
  and computer keyboard), record / stop / listen / name / save. Saved
  in the songbook next to the built-in ones, survives a restart.
- In "musician" mode the tune's pitches are reduced to the guitar's range (the choice of
  string/fret is his, several variants of one pitch are allowed).

## 5. Two modes (setting)

- **«Музыкант»** (default): the network chooses string/fret/strike directly; the hands on
  screen only play an animation of the chosen notes.
- **«Руками»**: the musician chooses the *intended* note, and it sounds only if his body
  actually reached it: the left hand - to the needed fret (a point on the neck), the right
  hand - to the string at the moment of the strike. The hands are driven by the same brain (arm joints),
  plus a motor reward "hit the target point". A miss means no note or a muted one. The
  musician learns to account for the hands (a note that did not sound is not heard in the reward).

## 6. When he plays (his choice)

- The guitar lies in his world (it appears near the start), he carries it on his back after he
  picks it up for the first time.
- The choice "world or guitar" follows the same rule as curiosity: compare the slow average of
  interest in the world (the predictor's error on what he sees) with the average of interest in music
  (novelty of his phrases + growth of the score). When the world has become more boring than music by
  a threshold - he sits down and plays; when music is more boring than the world - he gets up and goes. Hysteresis:
  at least 3 attempts in a row and at least 20 s in the world between sessions. The setting «Как часто
  берёт гитару» shifts the threshold (rarely / sometimes / often).
- While he plays: his curiosity in the world does not learn, life stands at 1x (see 2).

## 7. Attempts journal and window

- **Every attempt** is saved as notes + tune + reward components + the owner's rating +
  time; the sound is synthesized again when played back (little space). No more than 2000
  attempts per character; above that - keep the first 20, every tenth one, and the last 200.
- **The «Попытки» screen:** a list (date, tune, score, 👍/👎), «Слушать», «Сравнить с
  первой по этой мелодии» (plays one after the other), a progress chart per tune and
  for free play.
- **In the game while he plays:** the neck with the fingers, the notes being played in a line, the name
  of the tune, 👍/👎 buttons for the current/last attempt. A line in the chat from the speech
  network («учу «Кузнечика»», «получилось!») - as with his other thoughts.
- **Settings:** mode «Музыкант / Руками», volume, «Как часто берёт гитару».
- **Statistics:** time with the guitar, attempts, ratings 👍/👎, the best score per tune,
  the taste model's accuracy on the owner's ratings.

## 8. Tests (laws, TDD, mutation check)

- A string/fret sounds at its own pitch (FFT, error ≤ 5 cents), the decay is monotonic,
  force → loudness.
- Reward: a consonant chord > dissonant; an exact repeat of the tune = max A; a shift in
  time lowers A smoothly; silence and "mush" are penalised; novelty falls when repeating.
- Learning: on a fixed seed after N attempts on «Кузнечик» (N and the threshold - from a measurement)
  closeness A grows noticeably (the median of the last 20 > the median of the first 20 by a margin).
- Taste: on synthetic ratings with a hidden rule (for example "likes high notes")
  the model guesses ≥ 80% after 50 ratings.
- Choice: a bored world → he picks up the guitar; music bored → he puts it down; hysteresis holds.
- «Руками»: a miss does not sound, a hit does; the musician does not get a reward for a note that did not sound.
- Volume 0 - the audio device is not opened; no sound card - no crash.
- The journal: survives a restart, thinning out keeps the first/every tenth/last ones; «Сравнить»
  plays the right pair.
- The keyboard: a recorded tune = the pressed notes and durations; it is saved.
- The window laws of the project: a wrapping caption without a fixed height, everything fits in the
  minimum window with a font 10% larger, the gaps are on the `theme.py` scale.

## 9. Order and release

1. The string sound and the mixer + volume. 2. The musician and the reward (B, then A). 3. The songbook
and the keyboard. 4. The taste and 👍/👎. 5. The choice "world or guitar". 6. The journal and the «Попытки» screen.
7. «Руками». 8. Docs EN/RU, CHANGELOG, release notes v1.1.0, screenshots, the build and
`--selftest` (the guitar sounds, the attempts are counted). A commit per item.
