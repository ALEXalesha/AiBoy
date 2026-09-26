# AiBoy 2.0: a living world - specification

Date: 2026-09-27
Status: agreed with Alexey in a dialogue on 26-27.09.

His answers, in order:
- **toys:** ball, blocks, paper airplane, soap bubbles that pop with a sound, bicycle,
  boomerang, drawing marker, fishing rod; water in the world;
- **choice:** he takes whatever he likes;
- **release:** one, "with major work";
- **learning approach:** "C";
- **hardware:** CPU, no video card (option "C");
- **world sizes:** up to 640 and 1280 m;
- **rest:** he sits or lies down after walking for very long;
- **swimming:** he learns to swim;
- **food and drink:** he eats and drinks by himself;
- **the dog:** a neural dog that he tames with food from his backpack, plays with using the
  same ball, and that sits, lies, swims, jumps "and all that";
- **birds:** they learn to fly and flee from him and the dog;
- **his mood:** his mood changes the chance that a dog comes and how many birds there are;
- **speed:** add x16.

Release: AiBoy 2.0.0, after 1.1.0 (the guitar), which ships separately. Extra ideas
(skipping stones, kite, camera, net, snow) are not in this release.

## 1. What this is and the rule

His world becomes alive:
- a toy chest;
- water to drink, swim and fish in;
- food;
- rest when tired;
- a dog he can befriend;
- birds in the sky.

Everyone in it learns. He learns his toys, swimming and taming. The dog learns to move,
swim and fetch. The birds learn to fly.

The project rule stays. Everything that anybody does comes out of their own numpy networks
(`nn/`):
- movements, choices and drawings;
- the look of toys, food, fish, the dog and birds.

The code only draws, runs the physics, synthesizes the sound of physical events and trains.
No PyTorch, no GPU: CPU and real, visible attempts only. The speed-up is x2/x4/x8 and the
new **x16** (see 3).

## 2. World size

- Settings "Размер мира": 80 / 160 / 320 / **640 / 1280 m**.
- **Macro relief.** The terrain network sees waves of 80 m and shorter, so a 1280 m world
  would look repetitive. A new small CPPN `world/macro.py` adds, from the world seed, slow
  relief with waves of 160-1280 m: ranges, valleys, big basins for lakes. It is on only for
  640 and 1280, so the existing sizes and the saved worlds keep their look.
- **Everything else scales with the width:**
  - entity count;
  - the reachability graph;
  - the gallery thumbnail, which shows the whole ring squeezed and a detail;
  - the minimap strip in the panel, new, useful from 640 up.
- Only the visible part is drawn and simulated in full. Animals far from the camera run a
  cheap step (position and needs, no skeleton).
- **Laws:** no visible repeat (terrain autocorrelation at any lag of 80-640 m below a
  threshold); a frame at x8 in 1280 m stays within the 12 ms budget.

## 3. How skills are learned: common body skills + a head per activity (approach C)

**Body skills** of the person (`skills/`), each a small network with its own motor reward:

| Skill | What it does |
|---|---|
| `reach` | hand to a point (exists since 1.1.0 in `brain/reach.py`, moves here) |
| `kick` | foot to a point with a given velocity at contact |
| `throw` | swing and release: hand speed and angle at the release |
| `catch` | hand to the predicted point of a flying object; prediction by `skills/predict.py`, which learns from flights it has seen |
| `pedal` | leg cycle with a given force |
| `draw` | hand along a path |
| `swim` | arm and leg strokes in water (see 5) |

**Heads** (`toys/<toy>/policy.py`, `acts/<act>/policy.py`):
- a small MLP or GRU per toy or activity;
- a head decides *what* to do; the skills do *how*.

**Training.** REINFORCE with Gaussian noise and a baseline, as the musician in 1.1.0.
Skills and heads learn separately, so a new activity cannot break an old one. Stored per
body seed in `skills/<seed>/`, `toys/<seed>/`: a new world with the same person keeps what
he has learned.

**Speed.** The speed button gets **x16**: `SPEEDS = (1, 2, 4, 8, 16)` in the settings.
- At x4-x16 all sounds except the guitar's are skipped. The guitar keeps its 1x while
  playing.
- At x16 only what is near the camera runs full physics; far animals take the cheap step.
- If a frame cannot keep up, the real speed drops rather than the physics step growing
  (a bigger step would break contacts). The button then shows it honestly, for example
  "x16 (сейчас x11)".
For every learned skill the plan measures and records attempts to a visible improvement.

## 4. Physics

- **pymunk 7.3** (Chipmunk2D; a wheel exists for Python 3.13) for rigid bodies:
  - ball;
  - blocks;
  - bicycle (frame, wheels, joints);
  - rod float and line (a segment chain);
  - the dog's body (see 8).

  Terrain is a static polyline from the heights. It is a small new dependency, safer than a
  written solver with stacking and joints.
- **The person's body stays on `body/physics.py`.** Contact with objects goes through
  kinematic "hand" and "foot" bodies. On the bicycle the pelvis sits on the saddle and the
  feet on the pedals.
- **Own aerodynamics** (`physics/aero.py`), since pymunk has none:
  - airplane: lift and drag by the angle of attack, stall;
  - boomerang: flat 2D model; spin gives a force across the velocity, so a good throw loops
    back; spin fades;
  - bubbles: buoyancy and wind drag;
  - birds: wing lift and drag by the angle and speed of the wing (see 9).
- **Water physics** (`physics/water.py`): buoyancy by the submerged share of a body, drag,
  small surface waves from bodies and splashes.
- **Wind:** a small weather network from the world seed (`world/weather.py`) gives a base
  strength and direction plus slow gusts. Wind moves bubbles, the airplane, the boomerang,
  birds and clouds.

## 5. Water and swimming

- **Where the water is.** A new network `world/water.py` (from the world seed) decides
  whether a world has water and the level in the basins. It is trained by evolution
  (`train/water.py`) for "interesting": about 60% of worlds with water, 1-4 ponds or lakes
  of different depth. In 640-1280 m worlds the macro relief gives room for big lakes.
- **How it fills.** A basin fills up to min(spill height, network level). Water is drawn
  with a surface line, transparency and waves. No streams: in a side view they read as a
  ditch.
- **Swimming is learned from zero** with the `swim` skill (a head plus body strokes):
  - at first he flails and barely moves;
  - the reward is distance and speed on the surface, head above the water;
  - he wades where the depth is under half his leg length;
  - deeper water opens to him as his swimming improves: the reachability graph asks his
    current measured swim ability.
- **Swimming as an activity.** He chooses it himself when he likes it (see 10).
- **Safety.** He never drowns. If he is in deep water and tired, or the swim is failing, he
  floats and is carried to the nearest shore within 5 s; the chat comments on it.
- **Fish:**
  - a fish network from the pond seed gives shape and colour;
  - they swim in simple patterns;
  - they bite with a probability by time of day and bait depth.

## 6. Food, drink and rest (needs)

**Three needs**, each 0..1, in the life save:
- **hunger** grows slowly;
- **thirst** grows faster in the sun and after swimming or running;
- **tiredness** grows with walked distance, climbing, swimming and bicycle.

**Food:**
- world entities get a new kind **food** (berries on bushes, fruit on trees, mushrooms),
  and their look comes from the existing entity network;
- each food kind has a hidden taste from a food network by the world seed; some are tasty,
  some are not;
- he learns his own preferences from experience (pleasure after eating, see 10) and
  starts avoiding what he did not like;
- his backpack has a food supply and a flask; both are refilled at the chest.

**Drink:** from ponds and lakes; in a world without water, from the flask.

**Rest:**
- tired, he sits down;
- very tired, he lies down;
- lying restores fastest;
- tiredness above 0.5 makes walking a bit slower.

**Needs as urgency.** Needs are not rules; they are urgency in the choice (10). The
stronger a need, the sooner he goes to eat, drink or rest. **Law:** no need stays above 0.9
for more than 60 s of game time.

**Chat.** New state flags: "hungry", "thirsty", "tired", "tasty", "not tasty". Speech is
retrained and gets teacher phrases.

## 7. The toys

A toy chest stands near where he appears. He carries a toy in his hands or on his back.
The look of the toys comes from a toy network by the chest seed.

| Toy | What his head decides | Success (reward) | Sounds |
|---|---|---|---|
| Guitar | as in 1.1.0 | tune + own sound + 👍 | strings |
| Ball | kick / throw / catch; target, force, angle | distance, hitting a target, catch, catching his own wall bounce | kick, bounce, catch |
| Blocks | which block (sizes and colours from a net), where and at what angle | tower height or bridge span standing 3 s | clack, fall |
| Paper airplane | throw angle and force, trim | flight time and distance | swish, rustle |
| Bubbles | blow strength and duration, then catch or pop | size without bursting, popped by hand | blow, pop (pitch by size) |
| Bicycle | pedal force, body lean on climbs and descents | distance without falling, climbs taken | chain, bell, fall |
| Boomerang | throw angle, force, spin | how close it returned; a caught return counts double | whoosh, catch |
| Marker | stroke path (a small stroke GRU, a mini sketch-rnn) | beauty: his taste net + 👍 | scratch |
| Fishing rod | cast angle and force, when to pull after a bite | fish caught | cast, splash, reel |

- **Drawings** stay where they were made until the world changes. All drawings go to the
  Gallery ("Рисунки"); caught fish go there too ("Улов").
- **Sounds** are synthesized like the 1.1.0 string: impulses, filtered noise, pitch by size,
  no sample files. One mixer and the 1.1.0 volume; 0 = silent.

## 8. The neural dog

**When it appears.** A dog can appear when a world is created and can also wander in later.
- The base chance comes from the setting "Собака": never / sometimes (about 40% per world) /
  often.
- **His mood changes it** (see 10): when he is in a good mood a dog comes more likely and
  sooner; when he is sad, less likely. A dog that has not been tamed may leave after a
  while and come back. The dog network from the world seed gives:
- the look: size from small to large, colour, ears, tail;
- the temperament: shyness, playfulness, stamina.

**Body.** A four-legged skeleton in pymunk, with its own physics and animation:
- walk, run, jump;
- sit, lie;
- swim (paddling);
- shake off after water.

**Brain.** Its own small networks, also learning:
- movement (gaits, jumps over bumps, swimming): a motor head trained with its own motor
  reward, awkward at first;
- behaviour: approach or flee, follow, chase, bring back, rest, eat, drink. It works like
  the person's choice (10): its own liking, needs and fears;
- **trust in him:** a value that grows when he approaches slowly and offers food, and falls
  when he runs at it or throws something at it.

**Needs.** The dog gets hungry, thirsty and tired too. It drinks from water, eats what he
gives or what it finds, and sits or lies down, sometimes next to him.

**Taming.** It is his own activity, with a head that learns:
- he decides how fast to approach, at what distance to stop, and when to offer food from
  the backpack (a separate dog treat supply, refilled at the chest);
- the success is the dog's trust;
- once trust passes a threshold the dog is tamed: it follows him, rests near him and plays.

**Play with the same ball:**
- he throws or kicks the ball; the dog chases it and brings it back;
- at first it runs off with the ball; bringing it back is learned by the dog's behaviour
  network, with the reward "praise and a treat when brought";
- the dog swims after a ball that falls into water.

**Laws:**
- the dog never gets stuck: the reachability graph for its body;
- trust grows under patient approach and falls under running at it;
- a tamed dog brings the ball back more often after N throws than at the start.

## 9. Birds

**When they appear.** From time to time a small flock crosses the sky or lands on trees
and on the ground.
- The setting "Птицы" (off / rare / often) gives the base rate.
- **His mood changes it** (see 10): in a good mood flocks come more often and are bigger
  (up to 8); when he is sad, rarer and smaller (1-2). A bird network from the world seed gives:
- the look: size, colour, wing shape;
- the kind: small fast birds or larger gliders.

**Flight is learned.** Each bird kind has a flight network (wing flap strength and angle,
tail). It is trained with aero (4) and a reward for holding height and speed with little
effort:
- the first birds of a new world flap clumsily, lose height and land badly;
- later ones fly better;
- the flight network is saved per world.

**Reactions:**
- birds see him and the dog and try to fly away: a flee reward for distance to danger;
- if they flee in time, they learn when to take off earlier;
- the dog chases landed birds and never catches them (a hard rule of the world: nobody
  gets hurt);
- he watches them and runs after them sometimes. For him, birds are a source of interest in
  the world, so the existing curiosity reward counts them.

**Sounds:** wing flaps and short chirps, synthesized.

## 10. The choice: he does what he likes

**Activities:**
- explore the world;
- each toy;
- swim;
- tame or play with the dog;
- eat;
- drink;
- rest.

**Pleasure after every attempt or episode.** P = wS·success + wG·growth + wB·beauty +
wO·owner + wT·taste:
- success: the activity's reward normalized to 0..1;
- growth: this attempt against the median of the previous 20;
- beauty: his taste net for drawings, sound and flight (the 1.1.0 taste generalized);
- owner: 👍 +1 / 👎 −1 if given;
- taste: food pleasure, for eating only.

**Mood.** A slow average of all his P over the last minutes of game time, from −1 to 1.
- It changes how the world answers him: the dog chance (8) and the birds (9).
- It is shown in the panel as a face next to his needs.
- It is a state flag for his speech, so he can say he is in a good mood.

The mood is a sum of his own pleasures, not a network. The worlds and animals are still
networks; the mood only sets their spawn rates within the settings.

**Scoring and choosing:**
- **Liking** per activity: a slow average of P (α = 0.05), in the save.
- **Boredom** grows while he does an activity and fades while he does not.
- **Score** = liking − boredom + need urgency. Urgency is zero for play and grows with
  hunger, thirst or tiredness for eat, drink and rest.
- **Choice:** softmax(score / T) with hysteresis. The 1.1.0 "world or guitar" rule
  generalizes to all activities: at least 3 attempts and 20 s before a switch, unless a need
  is above 0.8.
- **What can be chosen:** the rod and swimming only with water in reach; the dog only when
  there is one.
- **"Offer a toy"** (catalogue) puts a toy next to him. He takes it with the same softmax
  plus a bonus, so a toy he dislikes may be refused, and the chat says so.

## 11. Screens

- **«Каталог»** (new tab):
  - toy cards: look, liking meter, attempts, best result, favourite marked, "offer";
  - activity cards for swimming and the dog;
  - his needs as three bars;
  - the backpack: food, treats, flask.
- **«Попытки»:**
  - a selector for toys and activities (swimming, taming);
  - first and latest attempt as a replay (audio for the guitar, recorded trajectory for
    the others);
  - a progress chart and the journal; thinning is the same as in 1.1.0.
- **«Статистика»:**
  - time per activity and the favourite over time;
  - catch, drawings, best tower, longest swim;
  - dog trust and fetches;
  - bird flight quality.
- **«Галерея»:** "Рисунки" and "Улов" next to worlds and people.
- **In the world:**
  - the chest;
  - the toy in hand or on the back;
  - water, fish, food, drawings;
  - the dog and birds;
  - the "what he sees" overlay also shows the flight prediction when he catches, and the
    dog's trust when he tames it.
- **Settings:** world size, "Собака", "Птицы" (off / rare / often), speed up to x16.
- **Panel:** a mood face next to the three need bars.

## 12. Saves and migration

- A 1.1.0 save opens in 2.0.0:
  - the guitar he carries goes into the chest;
  - the musician and the guitar journal stay;
  - needs start at zero.
- **New files per body seed:**
  - `skills/<seed>/*.npz`;
  - `toys/<seed>/<toy>.npz`;
  - `journals/<seed>/<activity>.json`.
- **Per world:** dog and bird networks, `worlds/<world seed>/`.
- A broken or missing file means a fresh network, not a crash.

## 13. Tests (laws, TDD, mutation check)

**Physics:**
- the ball does not fall through terrain up to the maximum kick;
- an unbalanced tower falls, a balanced one stands 3 s;
- the bicycle on flat ground rolls straight;
- the airplane with zero trim glides, a stalled one falls;
- a good boomerang throw returns within 1.5 m;
- a bubble rises slower than it drifts;
- a body in water floats by its share.

**Water and swimming:**
- water never above its spill height;
- no unreachable-deep water on his path before he can swim;
- he never drowns: back on shore within 5 s;
- swim distance per attempt grows over N attempts.

**Needs:**
- no need above 0.9 for more than 60 s;
- a food he rated bad is chosen less often after it;
- tiredness makes him sit, then lie.

**Learning:** for each skill and toy, fixed seed, N attempts, the median of the last 20
above the first 20 by a margin. N and the margin are measured in the plan.

**Dog:**
- it gets used to him under patient approach and flees when he runs at it;
- the tamed dog fetches more often after N throws;
- it never gets stuck;
- it sits and lies when tired;
- it swims.

**Birds:**
- the flight network holds height better after training than before;
- birds take off before he or the dog comes close;
- nobody catches a bird.

**Choice:**
- a toy with constant 👍 becomes the favourite;
- a boring activity is dropped;
- no rod or swimming without water;
- an offer can be refused.

**World size:**
- 640 and 1280 without visible repeat;
- the existing sizes are unchanged: their worlds are the same bytes as before.

**Sound:**
- every event has a sound;
- volume 0 opens no device;
- x8 skips toy sounds and the guitar keeps its 1x.

**Mood:** a good mood raises the dog chance and the number of birds, a bad one lowers them;
"never" and "off" in the settings stay absolute.

**Speed:**
- x16 advances game time 16 times faster when the frame keeps up;
- when it cannot, the physics step stays the same and the real speed shown is honest.

**Performance:** x8 in a 1280 m world with a dog, birds and a toy stays within the 12 ms
frame budget; the real speed at x16 in the same world is measured and recorded.

**Saves:** migration from 1.1.0; corrupted files.

**The rule:** the game does not import the teacher; looks and behaviour come from networks.

**Screens:**
- offscreen screenshots of every new screen at 1280x720 and 1920x1080, looked at by eye;
- overlap laws for the new labels and cards.

**Mutation check** across the new modules.

## 14. Order of work (one release) and release

1. Foundation:
   - pymunk terrain;
   - `skills/` (reach moves in);
   - the activity choice with needs;
   - the chest;
   - the «Каталог» screen;
   - world sizes 640/1280 with macro relief.
2. Needs: food entities, taste, backpack, rest poses.
3. Water, water physics, swimming, fish.
4. Toys:
   - ball;
   - blocks;
   - wind + bubbles;
   - airplane + boomerang;
   - fishing rod;
   - bicycle;
   - marker.
5. The dog: body, movement learning, behaviour, needs, taming, fetch with the ball.
6. Birds: flight learning, fleeing, reactions.
7. «Попытки», «Статистика», «Галерея» for everything; speech retrained.
8. Release 2.0.0:
   - previews and sound samples in `docs/audio`;
   - README EN/RU, CHANGELOG, `docs/release-notes/v2.0.0.md`;
   - build and self-check on the exe;
   - the usual cycle: Gitea, GitHub, install, old builds to the Recycle Bin.

**Parallel work.** After step 1 the independent parts (3, 4, 5, 6) can go to separate
agents in separate worktrees. Each owns its modules, and one person merges. After each
step all tests are green and the step is committed.
