# Red Alert agent

How it plays, why, and where things live. Numbers are in `config.py`.

## The game

Real time, 25 ticks/s, the server asks about 5 times a second. A move is a
batch of orders. Win by destroying every enemy building and the MCV; at the
time limit (20 min in tournaments) the tiebreak is `kills_cost - deaths_cost`,
then `assets_value`. Platform rules: https://api.altruagent-game.com/skill/redalert

## Design

- **Rules play every view** (no model on the hot path): `agent.py` reads the
  state, `brain.py` picks a Mode, `executor.py` builds one order batch.
- **Models advise off the hot path**: the advisor (`advisor.py`, Sonnet) is
  told this strategy and sees what the rules see (unit values, intel, raid
  situation, confirming); every ~6 s it may set posture, attack route,
  expansion wish, army target, raid target and turret side, each within the
  rules' limits (staging and formation are the rules'). Haiku squad leads
  (`squadlead.py`) direct the raid squad and the garrison when not in a fight.
  Advice is validated and expires; safety rules win.
- **Logs**: `matchlog.py` writes a JSONL and a summary per match to
  `matchlogs/` (samples every 15 s, mode changes, refusals, model advice).

| Module | Role |
|---|---|
| `observe.py` | Typed, defensive view of the state and game config (map, ore, costs) |
| `memory.py` | Enemies seen through the fog |
| `commander.py` | The army's one decision each view: policy, escort, look-before-commit, no quick flips |
| `brain.py` | The policy: OPEN, BUILD, ALL_IN, RETREAT, DEFEND_RUSH, DEFEND_BASE, PROTECT_LEAD |
| `buildplan.py` | What to build next and where to place it |
| `executor.py` | Production, economy, army orders, rally points, stances |
| `scout.py` | Base threats, the massing point, hunting the last buildings |
| `scouting.py` / `intel.py` | The scout corps (early circuits, pickets, base checks); reading their opening |
| `raid.py` / `guard.py` | Harvester raids (and the second prong); the expansion garrison |
| `expand.py` | Our expansions; enemy expansion sites |
| `harvest.py` | Harvesters steered to fields that still have ore |
| `micro.py` / `movement.py` | Focus fire, pull back damaged units; unstick units |
| `orders.py` | Order builders and shape checks |
| `sparring.py` | TEST ONLY: self-play sparring archetypes — rusher, turtler, macro (never in a tournament) |

## Strategy

- **Opening**: power, barracks, refinery, refinery, power, war factory (the
  pros' double refinery: refineries 1:00 / 1:38, war factory 2:40). The advisor
  waits until it stands unless a rush is read. Then, unless outgunned (units
  first), the third refinery and the service depot.
- **Defend**: a building lost to a push bigger than a small raid recalls the
  army, unless our attack is winning (a base race: it goes on, the home units
  fight). A small raid is handled by the units at home only.
- **Small raids**: a scout or two at the base draws the nearest couple of
  infantry; the concave keeps its place. In the opening, a rush is enemies
  worth 600+ at the base (89356b51: two scouts flipped both armies to defend).
- **Placement**: a spot the engine moved a building from is skipped for 3
  minutes (89356b51: one legal-looking expansion refinery spot failed three times).
- **Placement rules, from the OpenRA source** (BuildingUtils.cs, Building.cs,
  mods/ra/rules/structures.yaml): every non-empty footprint cell (bibs
  included) on Clear/Road terrain (ore is "Ore": never), no actor of any
  kind on it, no other building or bib; within 2 cells (Chebyshev, at most
  one free cell) of a cell of a building that gives buildable area (not
  turrets); its centre within 16 cells of a yard's centre (we keep 13).
  Shapes: fact 3x4 (bib row), proc _X_/xxx/X==/===, fix _+_/+++/_+_,
  tsla 1x1. Checked against 266 logged placements: 200 legal and placed,
  26 illegal and moved; most of the 30 legal-but-moved had units in the way.
- **Turrets**: the first faces the road in (scout.approach_cell: the cell 12
  from the base that is the shortest walk from the enemy spawn), as close to
  that line as a legal spot allows; the rest fan out to either side.
  Buildings grow only into the half of the base facing their bearing, and
  for turrets units in the way don't outrank direction (they're cleared off
  the spot): 08560d9b's massing army pushed the first flame tower behind the base.
- **The enemy through the fog** (brain.enemy_estimate): the most of what we
  see or remember, the most we saw in the last 3 minutes less our kills
  since, and 0.8 x (our army + our deaths - our kills). Attacks, retreats and
  the advisor's posture all weigh this, never just what is visible (08560d9b).
- **Committed attacks**: an attack goes on until the retreat rule ends it (no
  drop back to massing when the estimate dips; 82908a24 flipped 9 times).
  The fog prior scales by their known refineries / ours (floor 0.5): a
  one-refinery rusher can't field our army. An aggressive read brings
  turrets, not a standing hold (0ee93a05 held 13 minutes). Before the war
  factory, a warning gets one turret; an attack on the base gets the full set.
- **Retreat** at losses over 1.3x kills and over 25% of the army (the bigger of at the start and now). The massing army attacks
  anything worth more than a small raid within 12 cells of it.
- **Expansion MCV**: trained only with 8+ fighters and not outgunned;
  cancelled (money back) if outgunned while it trains.
- **Steadier**: an alarm takes two lost scouts; a rush read holds until 4:00;
  defending takes a 2000+ threat over 1.3x the home units and lasts 20 s past
  the last sighting; modes dwell 40 s; the advisor's posture and read change
  only when two answers agree. With no trigger: build a 3000+ army, hold the
  massing point and pickets, then push and stay committed until the retreat rule.
- **Finishing and the estimate** (b90e492f): "beaten" needs a 5000 lead (a 600
  lead sent 18 units in at 4:05); finishing waits out the retreat cooldown
  unless hunting; each past sighting of their army counts what they could have
  rebuilt since (as much as we built, scaled by refineries), so a good defense
  doesn't make them look empty. Overwhelming: split across up to 4 bases, one
  per 10 units. An outgunned expansion is called off, not retried every view.
- **After the main base** (hunter.hunting, latched: razed once we knew 6+
  buildings and know 3 or fewer while ahead; off if a real base of 7+ turns up
  or we fall behind): the whole army fans out over a 12-cell grid of the map
  (64 cells on Sidestep/Trapped, all reachable) in groups of 3, nearest
  unsearched cell first; each building cluster or MCV found gets the nearest
  10 units; the rest keep searching.
- **The hunt** (c4caa45e: a barracks survived a 100-unit hunt): up to 8
  groups of 6+, every cell where we once knew an enemy building is a search
  point, and a point counts as checked within 5 cells (about sight).
- **Fight for the base, push when maxed, finish won fights** (c7a982ac,
  Shadowfiend II: 75 units watched a refinery die 17 cells away; two 20k+
  armies idled 10 minutes at ~1.1x; a won fight was abandoned for
  protect-the-lead): the army takes on any real push near our yards and
  refineries, not only near itself; an army of 15000+ attacks at 1.05x; an
  attack that has killed 1000+ more than it lost ignores the 0.8x retreat line
  (only < 0.5x or losing trades stop it) and protect-the-lead.
- **Steady orders** (906dbb32: the big push split and went back and forth): a
  unit sent within 5 cells of the same spot in the last 20 s isn't re-sent
  (unless it stands idle away from it: stuck); a queued leg keeps only the
  units its first leg kept. Only reinforcements still at the massing point wait
  for a group; nobody is sent back from a push.
- **Cleaner fights** (5e840776: attacks retreated 6-30 s after contact and were
  chased; armies strung out, single file through Sidestep's 2-cell gap):
  squad leads off unless REDALERT_SQUAD_LEADS=on (80% of ~180 calls repeated
  the rules 2 s late, the rest overrode them); focus fire only on enemy
  fighters within ~4 cells, sticky; no lone pull-back of damaged vehicles; in
  the first 15 s of a fight only < 0.5x pulls back; stragglers wait at the
  massing point until 6 go together; an army of 12+ goes round a passage under
  5 cells wide when a wider route is at most 1.6x as long (Sidestep: north).
- **Choke towers** (screenshot potential-towers): from the terrain alone (any
  map), the narrowest points of the enemy's walking path to each of our yards
  (11 cells across or less, 4-16 cells out, on their side) are chokes; each
  base's first tower goes within 5 cells of its choke when a legal spot is
  there, else on that base's edge facing it; with no choke in range, facing
  the road in. Sidestep's middle gap is 2-3 wide, Trapped's island passage 6-7.
- **Attack a weak enemy we have seen**: a scout round of their base in the last
  45 s showing our army 2x what we saw overrides the "about our size" prior.
- **Harass and scout further** (f333a770, from the replay: both armies walked
  behind their bases for a minute to escort an MCV to a safe site; neither seat
  found any of the other's 2-3 expansions; raids hit the defended main field):
  no escort for an expansion behind our base; picket p0 roams the remote ore
  fields (18+ cells from both main bases), each rechecked every 2 minutes;
  the raid squad goes for remote fields before their main field, grows with
  the army (4-8, one per 6 fighters) and reforms after 30 s.
- **The army stays up front**: the massing point (and the rally points) no
  longer pull back into the base when an enemy is remembered nearby (a passing
  scout moved them for 90 s); the army fights what comes near and falls back
  to its turret only when outnumbered. Concave slots move off cliffs, trees
  and our buildings.
- **Attack bar 1.15 after 7:00** (was 1.2; 1.3 before, 1.05 with a 15k+ army).
- **More raiders the longer a healthy game goes** (no big army idling): with
  4+ harvesters and not outgunned, a second raid squad forms at 6:00 and one
  more every 3 minutes (up to 4), each a unit bigger and re-forming after 12 s
  instead of 30 s; together they never take more than 35% of the fighters.
  Each squad targets something 12+ cells from the others' targets.
  Outgunned or short on harvesters: the extra squads rejoin the army.
  Healthy also needs an army worth 6000+ and ore ahead (ee2ac21b raided on with
  a 9-unit army and mined-out fields); a unit is in one squad only (squads
  re-forming took each other's units); raids that lost 800+ net over the last
  few send only one squad for 2 minutes.
- **Raids only at their expansions** (52a77fb1 sent squad after squad into
  their main field to die): a raid forms only for a target at an enemy
  expansion our scouts found (18+ cells from their main base) — a harvester,
  refinery or lightly defended expansion there, or the advisor's pick there.
  No expansion found: no raid (the roaming picket checks remote fields).
- **The army holds the main choke, always**: an MCV on its way to a forward
  site gets the 2-rifle outpost as escort, not the army (52a77fb1 walked the
  whole army to the expansion for two minutes and the main lay open). No new
  attack starts while it travels.
- **Behind on army, units first**: estimate 1.3x our army (or the advisor's
  "more_army"), past 3:00: no money held for an MCV, refineries, depot or dome —
  only power, barracks, war factories and harvesters keep theirs (52a77fb1
  banked 1200-3800 for 4 minutes at 2-3x behind).
- **Expansions that always take** (52a77fb1, ee2ac21b): ore spreads from its
  mines all game and the platform drops a building asked for on ore (or on a
  unit) next to the main yard. The MCV unfolds where it stands only on a cell
  with room, no unit in the yard's way and (when one is near) one cell clear of
  the starting ore; a deploy still refused a beat later marks that cell bad and
  sends it to the nearest valid cell at once (was: three tries, then 3-7 cells
  at a time for two minutes). Expansion refineries avoid predicted ore; one that
  comes up elsewhere is asked for again at once,
  its spot skipped. Behind on army: no new refinery or MCV unless home runs dry.
- **Every order gets sent** (skill/redalert: 20 orders a batch, a batch every
  0.2 s): what doesn't fit a batch goes first in the next one (dropped if a
  newer order covers the same units, or after 1.5 s); a queued leg goes only
  with its first.
- **Hit back** (skill/redalert events): `under_attack`, `unit_lost` and
  `building_lost` events, plus anything of ours that vanished since the last
  view (events scroll out in a fight), mark spots we are being hit; for 6 s
  every idle fighter within 12 cells attack-moves there (at the nearest enemy
  seen by it).
- **Pushes are met at our line**: a push seen coming is met just behind our
  turret nearest it, else just in front of our building nearest it.
- **The guide's commands** (skill/redalert): the outpost uses the engine's
  `guard` order — it follows the MCV on its way, then stays with the
  expansion's yard (attacking threats near it first); an order the server
  reports `lost` or `rejected` clears our no-repeat memory so the plan sends it
  again. Not used on purpose: `power_toggle` (a powered-down dome or depot may
  stop unlocking siege units and tanks; we build power instead).
- **Decisive play, no guess** (3bd6e450): the enemy estimate is only what we
  know — the most we see or remember, or saw lately less half our kills since;
  no "about as big as us" prior, no rebuild guess. Trading clearly up (1500+
  net and 1.5x in the last 2 minutes) starts an attack at once, no look first;
  otherwise an attack on little seen waits for a scout's look. "Nothing seen"
  counts as nothing left only 2000+ ahead on trades.
- **A concave that fires at once**: ranks on arcs of 3.5 and 4.5 cells round a
  kill point 2 cells in front of the massing point (each within rifle range 5);
  units beyond the slots wait just behind the line. In-line units are never
  sent back to the centre (the edges of the line wandered to and fro), idle
  ones get their slot orders first.
- **Guard**: enemy fighters within 8 cells of the massing army are met by every
  idle unit in one order; they walk back to their slots after. The advisor may
  switch it off ("guard") and pick the waiting army's stance ("army_stance").
- **Intercept**: an enemy army worth 1500+ seen in our half sets the army up
  across its path, between it and the base it heads for (held 20 s after the
  last sighting), when at least as strong; else the turret line.
- **Scouts that look**: the lane scouts alternate their post with a look at the
  enemy's exit from their own side every 45 s; the extra pickets watch that exit
  from the flanks (never the centre lane). A rush is read from 600 of army seen
  before 2:30, or 1000 before 4:00, whatever their refineries; two early scouts
  lost latch a rush read until 4:00.
- **Reading a turtle** (intel.turtling): 4+ ground turrets known after 2:30
  and no push of theirs near our base for 5 minutes (a rusher's early
  pillboxes don't count while it keeps coming). Then: the dome at once, siege
  units first in the vehicle queue (about one per turret, 3-6), expansions as
  early as allowed, and no attack before 3 siege units unless the army
  overwhelms army and turrets together (2.5x). The advisor has a "turtle" read.
- **Against a turtle** (efefdca8, our agent vs the rules-only turtler: lost
  39k to 92k): the 7:05 attack (13150 into "10440", 700 actually seen) wasn't
  caught as blind because the check used the peak, whose rebuild term adds our
  own growth since a sighting — it now uses what we actually saw lately. Known
  turrets at the target count as 1.5x their price in the enemy army when we
  decide to attack (the 7:05 and 13:55 attacks went into 5-7 turrets). A raid
  target where a squad lost more than it killed is left alone for 4 minutes
  (a squad a minute died at the turtle's natural).
- **Unit strength, not price** (strength.py, from the OpenRA weapon/armor tables):
  hit points, armor class and damage per second against each class — a rifleman
  does 1875/s to infantry but 125/s to a tank, a rocket soldier 2500/s to a tank.
  `edge` (Lanchester-style) decides whether the army meets a push in front of
  our turrets (edge 1.25+) or at the turret line, standing just behind the
  turret; the army edge for attacks is scaled by the mix (0.6-1.5x, once 5+
  units a side are known); riflemen per rocket drop from 3:1 to 1:1 as their
  army turns to armor. 52a77fb1: 5000 vs ~4000 by price was about even by
  strength, met 5 cells in front of the pillboxes with 13 of 22 units.
- **Formation orders fit the batch** (52a77fb1): one order per unit overflowed
  the 20-order batch, and the walk-to-the-centre order came last and was cut
  every view — new units stood at the barracks doors through the fight. Now at
  most 6 slot orders a view, and everyone else far from the centre walks there
  in one order sent first.
- **The pros' unit mix** (9 pro replays: rifle:rocket ~3.7:1, 45-55% of combat
  value in main battle tanks in every phase — heavy for Soviets, medium for
  Allies, light tanks never): we train the strongest tank offered (3tnk, 2tnk,
  4tnk, then 1tnk only before the depot; 52a77fb1 built light tanks all game as
  England), and while tanks hold under 45% of our combat value the tank queue
  runs full and infantry wait unless there is money for both (not while
  defending or warned).
- **Ore growth model** (oregrowth.py): a Monte Carlo of the engine's seeding
  rules (each mine adds ore every 3 s along a random walk past full cells) on
  Shadowfiend II, Sidestep and Trapped: fields with a mine grow 1 cell past the
  starting ore by ~1:30-5:00, 2 by 8:00-15:00, 3 by ~17:00+; fields without a
  mine never grow. MCV deploy spots (looking a minute ahead) and expansion
  refineries keep off that ring.
- **A lost war factory is rebuilt first** (past 3:00), jumping the dome, depot,
  kennel or silo in the queue, its money kept even while defending.
- **No blind attacks**: past 7:00, an attack call whose estimate is mostly the
  "about as big as us" prior (over 2x what we saw lately, no fresh base look)
  waits up to 40 s while a scout looks where their army masses (our massing
  rule mirrored). ee2ac21b attacked 5150 into "4440" — our income had stopped,
  theirs hadn't — and met 13500.
- **Mass farther out on big maps**: the massing point is 10 cells in front of
  the base or a quarter of the way to the enemy, whichever is farther (still
  capped at 30%): Shadowfiend II 17 cells, Trapped ~14, Sidestep ~13. The army
  meets attacks sooner and the base has room to grow. Buildings for an
  expansion stay inside that expansion's yard radius.
- **An honest estimate** (f581756e: 3900 attacked "1040", really 2400-3700 in a
  defended base): no "smaller economy" discount (it halved the estimate for
  refineries we hadn't seen; a rusher spends everything on army anyway); a
  past sighting loses only half our kills since (the rest are buildings and
  trucks). "Outgunned" compares combat units only (fighters seen recently vs
  ours; no turrets, buildings or trades).
- **No coin-flip attacks late** (ef66403c: both seats attacked at ~1.0x after
  7:00 and retreated within 5-7 s; seat 0's retreat was chased into its base):
  the late attack bar is 1.2x the estimate (1.3x before 7:00), 1.0x only when
  behind at the end. The estimate is logged with every sample and mode change.
- **Even games keep expanding** (ef66403c: one expansion each until 9:30 at
  even trades, with the first three fields running dry around 9:00): with
  kills - losses within 2000 the advisor's "wait" doesn't hold the next
  expansion; expansions 120 s apart; the ore forecast fires at 240 s left
  (an expansion and its refinery take 3-4 minutes).
- **Expansion vs the advisor**: running out of ore overrides its "wait"
  ("more_army" all game blocked a second expansion); "balanced" is its
  default, "more_economy" when the ore runway is under 3 minutes. The MCV also
  needs 2000+ army value (the fighter count includes the scouts).
- **Scout rounds**: the left scout circles their base 14 cells out, the right
  7 cells out (through it) 10 s later, each round its own side, past the back
  and onto the other side; mid-game checks alternate the rings. On the three
  real maps they see 74-75% of everything within 18 cells of the enemy spawn
  (one ring of 11: 57-60%).
- **Rush defense** (6b457914): one refinery seen after 1:40 is a suspicion
  worth the warning turret on the road in (no hold); one refinery plus an army
  of 1000+ seen before 4:00 is a rush read. An outnumbered army (under 0.8x
  what is near it) falls back to our turret nearest the attackers instead of
  charging. With fewer trucks than refineries, no depot, kennel, dome or
  extra production until they are back.
- **Ore forecast** (harvest.HarvestManager): a field holds ~300 x its starting
  ore cells and each harvester working it takes ~10 a second. When the worked
  fields have under 150 s left, the next expansion may start at once (its
  refinery follows when it deploys) and no more harvesters are trained (the
  money goes to the army). The forecast flagged 8:53-11:10 in recent games,
  1.5-3 minutes before the expansions actually came.
- **Outpost** (guard.Garrison): 2 infantry at a forward expansion (nearer the
  enemy than our base) only; posted only while the army masses in calm, back
  at once on any attack, defense, retreat or nearby fight, unless their own
  expansion is attacked (then they fight there). 0ee93a05 kept 4, tanks
  first, at an expansion through every fight.
- **Advisor**: strong, generally good moves inside the game plan; its value is the read.
  enemy_style aggressive -> hold, up to 3 turrets, win on macro; macro -> a raid
  party of 6 that flanks to their economy; standard -> our plan. balance
  more_army -> no extra refineries/dome/expansion; more_economy -> expand as soon
  as the gates allow. mistakes (idle units, exposed harvesters, piling bank,
  stuck units, a losing fight) go to the log and console. Its posture, route,
  army target and expand calls stay, within their limits ("attack" only when
  the numbers support it).
- **Flanking**: none before 10:00 unless decisively ahead (dominant trades
  or 2.5x their known army): engaged early, a flank loses us the defense.
- **Economy**: trucks come with refineries (target = refineries + 2); more
  refineries over time; expand right after the service depot (~4:45), up to 3
  times, sooner when ore runs out. Money bands: broke = one queue; rich =
  spend harder.
- **Army**: infantry-heavy (about 3 riflemen per rocket soldier) from up to 5
  barracks, tanks as support from one war factory.
- **Fighting**: mass on our side of the map, attack with an edge, flank when
  the middle is guarded, target power then production, focus fire, retreat
  from losing fights (fighting back when pressed), hunt the last buildings.
- **Squads**: the scout corps, the harvester raid squad (second prong of an
  attack), home responders for small raids, garrison at the newest expansion
  (joins pushes once it's complete). The army splits one way: groups to
  points (two clusters, a semi-concave across the target, or hunt points).
- **End**: protect a real lead in the last 2 minutes (any lead in the last
  minute); when behind in the last 3 minutes, attack regardless.
- **Late game**: radar dome ~7:00; 2-4 siege units (V2 / artillery) once enemy
  defenses are known, shelling defenses first; against 3+ defenses, fewer
  riflemen per batch and more tanks (flame towers shred infantry). No tech
  centre, mammoths, aircraft or superweapons: they don't pay off in 20 minutes.

## Platform limits

No capture order (no engineer captures, no oil derricks), no mine laying, no
order to send a unit into a repair bay. Placement hints farther than about a
cell from the base are silently moved next to the construction yard.

## Lessons from matches

- A ready-but-unplaced building counts as pending (else it is built twice).
- Never skip ahead in the build order; wait at the first unmet step.
- Power plants can't wait behind a big building in the one-at-a-time queue.
- A needed harvester can't wait behind tanks in the vehicle queue.
- Cap refineries; ask for an expansion's refinery once, not every view.
- Only attacks in the opening are rushes; later, defend without dropping the economy.
- Mass (and rally new units) 10 cells in front of the base toward the enemy (3 if enemies are near), in a concave facing their base; the rules own staging, not the advisor (a2952a6d: it massed toward our ore and faced our base).
- Every combat order is an attack-move, retreats and pull-backs included: a plain move let units be shot in the back without answering. Plain moves only for the MCV and the scout.
- No all-in before 3:00 and a war factory: early sightings show a fraction of the enemy (b7fcbd4e: both seats traded their opening army at 1:05).
- Broke with no harvester: sell a spare building (dome, depot, defenses, extra production) to buy one.
- Retreat on a losing exchange, but not in the first 15 s of a fight.
- After a retreat, regrow before attacking the same defenses again (2 min at most).
- An MCV needs a 3x3 clear block; a stuck or slow MCV deploys where it stands.
- An MCV escort must never freeze the army when an attack is needed.
- With no enemy known, hunt; don't idle on the ruins of their base.
- Protect a lead only when ahead on kills - deaths.
- Turrets form an even fan facing the enemy, grown from base buildings (not from each other); no free slot, no turret.
- Harvesters leave a mined-out field for one with ore; a bare harvest order leaves them idle at home.
- Throttle *requests* for an expansion's refinery/turret, never their *placement*: a finished one always goes to the expansion.
- Keep fields staffed by refinery share, not only when home runs dry.
- Don't protect a lead against a beaten enemy (2.5x their known army): keep killing; split the army between their two nearest bases.
- When broke, the last money goes to units, not to another turret in front of the attackers.
- Depletion: a field with harvesters around but none mining for 45 s is mined out (harvesters search rather than idle).
- Once home is mined out, the newest expansion with live ore becomes the active base: new buildings and a 3-turret fan go there.
- Far ore never rules out a refinery when we're ahead: the base crawls toward it.
- No building may cover ore or blocked terrain: OpenRA refuses it and silently drops it at the main yard (why expansion refineries kept landing at home).
- Harvesting distances are walking steps, not straight lines: a cliff between a refinery and ore means its trucks go elsewhere (terrain.py; blocked cells come from the map legend).
- An expansion's refinery is placed only beside that expansion's own buildings; when the engine moves one home anyway, that spot is marked off-limits for the next one.
- An expansion counts as ours when a NEW yard appears near the planned site, even several cells off it (0561274e lost track of one and never built its refinery).
- Expand to the natural first: the nearest unmined field by walking, not the richest (b7fcbd4e walked past a natural 16 cells behind the base).
- Buildings stand only on clear ground: rough ground, beach and road/bridge are walkable but unbuildable, and a footprint there is moved home.
- Refineries spread over the fields in reach by load: a field takes one per 15 ore cells (1-3); the next goes to the field with the most room, nearest first on a tie (a2952a6d: three around the nearer, smaller field, none at the bigger one).
- Once their base is nearly razed (we knew 6+ enemy buildings, now 3 or fewer) and we lead, finish them: no protect-lead or hold; with none known the hunter aims the army and a big army sweeps 3 points at once (0561274e idled on ruins for 6 minutes).
- The advisor's army target counts only with a known enemy within 10 cells of it (b7fcbd4e: its stale target kept a 40k army 10 cells from the enemy's last building for 5 minutes).
- Placement: one candidate source (legal spots, exactly one free cell from a building that extends the buildable area), roles only score; no building on the army's lanes, the massing area or a refinery's road to its ore; a spot with units on or beside it loses (a2952a6d: legal spots on truck routes were moved home). The old "avoid"/"give up after 2 moves" nets are gone (no moves in 4 games since the centre-cell fix).
- The build radius (lobby "buildradius"): a building's centre within 13 cells of a construction yard's; beyond 16 cells every placement was moved home (audit of six games).
- Every finished Building/Defense-queue item is placed, known id or not (14a78c43: an unplaced kennel blocked the Building queue for the rest of the game, 9600 banked).
- Units stepping off a building's spot get no other order that view, from any part of the agent.
- An expansion deploys as close to its field as leaves room for the refinery (4-8 walking steps), not nearest our base (14a78c43: both naturals unfolded beside the main base).
- Hunting the last buildings: idle scouts each sweep one of the hunter's search points while the army takes what they find.
- One army: guards/garrisons claim no one during an attack and only units at home otherwise, never from the front (fd507a58). The raid squad is the one exception: the second prong.
- Raid squad: defenders before harvesters (weakest first); outmatched -> pull back fighting. While it draws their army, our main army attacks down the centre if our units outweigh theirs; once the main army is in a fight, the raid attack-moves into it (5aea2632 vs Ryanxhuang).
- The scout backs off from enemy fighters; killed past the middle before 6:00, it raises the early-push alarm for 30 s (5aea2632: died ~2:00, no warning, push at 2:17).
- No infantry cap: units spend only what the economy doesn't need (buildings, harvesters, an MCV in production plus the next build-order step are pre-funded); a read rush, an alarm or defending drops the reserve.
- Intel: one refinery but a war factory or 2nd barracks by 1:40 = a rush (pros have two refineries); it lifts the reserve, holds the army home and puts turrets up.
- Scouting: two scouts round the back of their base (left, right) in the first 3 minutes, no backing off — the buildings are worth the scout; then picket posts across the approach, 4 from 5:00 (spare units or trained: dogs where possible, a kennel after the war factory), a base check every 2.5 minutes. Posts and waypoints snap to cells we can walk to.
- Mass in a concave; attack in three groups closing on the target as a semi-concave.
- Look before committing: while trades are about even (kills - losses within 200), an attack call is re-checked 4 s later as the army edges forward to see; clearly ahead or behind, it goes at once. Once confirmed, fully committed (the dwell and retreat rules apply as usual).
- Barracks, factories and depots grow to the side away from the ore, slightly forward; the front is the army's (5aea2632: four barracks beside the massing point moved home).
- The early scout goes out along a flank, never down the middle, to a lookout over the ground in front of their base where a push masses (159f4f11's side lookout watched their natural and never warned); a dead rifle scout is replaced after 90 s, on the other flank.
- Queues: at most 2 deep per unit type, 1 unit per order: money finishes units instead of spreading over many.
- Nothing is built within 3 cells of the massing point, and if a building or rock ends up there the point moves to the nearest clear cell (159f4f11: barracks asked for on the army's spot four times, moved home each time).
- Attack into turrets only with 6 infantry per turret (15+) or 3 tanks per turret (siege x2); 2 known turrets tilt production to tanks/artillery (654f1b2e: three losing attacks into flame towers).
- Expansion MCVs go only to fields beyond 16 cells of our yards; nearer fields get refineries from the base (654f1b2e wasted a yard 8 cells from home).
- Dominant (kills >= 2x losses, lead >= 5000) with no real enemy army: finish them, no protect-lead (654f1b2e won 59k-26k on time instead of outright).
- Broke (under 600 for 45 s) while trading even or better: the next expansion goes 60 s after the last (not 150), past an advisor "wait"; army training pauses for the MCV.
- Never ask for a building where units stand: OpenRA moves it to the main yard (a1329ceb). A free spot is preferred; if ours stand on the chosen one, they step aside (fighters attack-move) and the placement waits up to 5 s, those units sitting out other orders meanwhile.
- The observation reports a building's centre cell; placement takes the top-left. Read as corners, every spot right of or below a building was 2 cells out and moved home (fd507a58 placement audit: each successful place landed at asked + (1, 1)). observe.py converts our buildings to top-left.
- OpenRA's adjacency: a new building within 1 free cell of a building that extends the buildable area; turrets don't. Power plants, barracks and domes are 2x3 (a bib row).
- Growth spots are chosen only among cells exactly 1 free cell from a building (legal by construction); stepping out along a diagonal and rounding put them 2 out (c2fec761: 14 of 26 moved home).
- An attack with no kills and nothing in sight for 40 s is parked on a stale target: forget remembered buildings around the army, ignore the advisor's target for 60 s. While finishing, the advisor never aims the army (c2fec761: 50 units sat on ruins while the last building stood 15 cells away).
- Memory follows a yard that packs up into an MCV (and an MCV that deploys); a remembered enemy MCV is a target.
- A small raid (worth 1500 or less) near a yard or harvester is attacked by units at home, the weakest raider first, even while the army is out.
- After 3:00 the army is recalled only for a threat worth 1500+ that outweighs the units at home and the turrets, and goes back out 10 s after the base is clear (feba5f26: two stragglers recalled a winning attack six times, the army parked 40-120 s each time).
- Stored ore is spent before cash: a frozen cash figure with ore piling up is an unspent bank. With 2500+ banked after 6:00: a second war factory and a 2-deep tank queue.

## Sources

- Platform guide: https://api.altruagent-game.com/skill/redalert
- https://github.com/OpenRA/OpenRA/wiki/How-to-be-a-better-OpenRA-player
- https://github.com/OpenRA/OpenRA/wiki/The-New-Players-Guide-to-OpenRA
- https://forum.openra.net/viewtopic.php?p=313702 (Multiplayer Guide 2022)
- https://sharoma.com/ral/logistics.htm
- 9 RAGL/ladder replays (`ora-replays/`, not committed): build timings,
  refinery-heavy economy, infantry-heavy armies, expansion to the natural at ~4:45.

## TEST ONLY — remove before the tournament

- `sparring.py`: in a self-play test match one seat plays a fixed archetype, without the advisor:
  - `RA_SPARRING=rusher`: plays like Yuriko (henry-vs-yuriko): one barracks, two refineries, three
    pillboxes on the approach by 2:30, war factory and depot, a third refinery ~10:00 and a second war
    factory ~11:30, no expansion; nonstop riflemen then main battle tanks; waves from ~3:00 (18+ fighters),
    each later one once the army is 1.4x the last launch (2.5-5 min after regrouping), pushing until
    down to 40% of its launch value.
  - `RA_SPARRING=turtler`: our build plus a growing turret ring (2 at 2:30 up to 8 by 12:00), army at
    home, no raids; attacks only from 15:00 with a 1.3 edge, or with an 18000 army.
  - `RA_SPARRING=macro`: refineries early (2 at 0:45, 3 at 2:30 ... 6 at 7:00), first expansion as early
    as allowed, a token army (1500) until 5:30, no raids; one big push at 12000 (or 13:00 with 1.15).
  - Every archetype still defends the usual way. `RA_SPARRING_SEAT` picks the seat (default 1); the old
    `RA_TEST_RUSH=1` means rusher. Three locks: the flag, no tournament id, and both seats of the session
    seen on this machine (marker files). Remove the module and the lines marked "sparring" to retire it.
