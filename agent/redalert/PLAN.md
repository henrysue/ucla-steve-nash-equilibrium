# Red Alert agent

How it plays and where things live. The numbers are in `config.py`; why a
rule exists is under "Lessons" (match ids in brackets).

## The game

Real time, 25 ticks/s, the server asks about 5 times a second. A move is a
batch of orders. Win by destroying every enemy building and the MCV; at the
time limit (20 min in tournaments) the tiebreak is `kills_cost - deaths_cost`,
then `assets_value`. Platform rules: https://api.altruagent-game.com/skill/redalert

## Design

- **Rules play every view** (no model on the hot path): `agent.py` reads the
  state, `commander.py` makes the army's one decision (policy in `brain.py`),
  `executor.py` builds one order batch.
- **The advisor** (`advisor.py`, a model, every ~6 s from the first second,
  off the hot path) reads the enemy from the scouting report (every enemy
  building and unit the first time our scouts see it: `intel.report`),
  predicts their posture and first push early, and may set posture, route, expansion,
  army and raid targets, turret side, the waiting army's stance and the guard
  — each inside the rules' limits. Advice is validated and expires; safety
  rules win. Each call carries its memory: its first read of the enemy (with
  evidence), its last 4 changes of read (what changed, how many answers and
  seconds the read before held), its latest answer, the read in effect, what
  happened since it was last asked (trades, losses, sightings, mode changes),
  how our recent attacks went, and our past games against this opponent
  (`opponents.py`, saved at each match end under `REDALERT_PROFILE_DIR` or
  `<REDALERT_LOG_DIR>/opponents`). Each answer gives a confidence and its
  evidence: an "aggressive" read raises turrets only at 0.6+, and at 0.8+ with
  evidence a change takes effect without waiting for a second answer; with no
  evidence from this game it is at most 0.5 sure (the opponent history is a
  hypothesis to confirm, not a fact). It sees how long ago their army was last
  seen and doesn't call an attack on a stale estimate. The
  system prompt is cached.
- **Between games** (`postgame.py`): after a tournament match, Claude Opus
  reads a digest of the match log (timeline, mode changes, first sightings,
  the advisor's reads, raid and attack results, buildings lost, the result)
  and writes the opponent's playstyle, opening, what worked and failed for
  each side, a summary and one high-level way to beat them, into their
  profile. The next game against them the advisor gets it as an addendum to
  this plan (its system prompt). By hand: `python -m agent.redalert.postgame
  <log.jsonl> --opponent NAME [--save]`. `REDALERT_ANALYST=off` switches it
  off, `REDALERT_ANALYST_MODEL` / `REDALERT_ANALYST_TIMEOUT` (90 s) tune it. Haiku squad leads (`squadlead.py`) are off unless
  `REDALERT_SQUAD_LEADS=on`.
- **Logs**: `matchlog.py` writes a JSONL and a summary per match to
  `matchlogs/` (samples every 15 s, mode changes, refusals, advice, attacked
  spots, the outbox, first sightings — kind "scouted" — with a "What the scouts
  saw first" section); the console prints each new sighting.

| Module | Role |
|---|---|
| `observe.py` | Typed, defensive view of the state and game config (map, ore, costs) |
| `memory.py` | Enemies seen through the fog |
| `commander.py` | The army's one decision each view: policy, escort, look-before-commit, no quick flips |
| `brain.py` | The policy: OPEN, BUILD (mass), ALL_IN, RETREAT, DEFEND_RUSH, DEFEND_BASE, PROTECT_LEAD |
| `strength.py` | How every fight is judged (`odds`) |
| `buildplan.py` | What to build next and where to place it |
| `executor.py` | Production, economy, the defense handler, army orders, rally points, stances |
| `scout.py` | Base threats, the massing point, the main base's turret line, the hunt |
| `scouting.py` / `intel.py` | The scout corps; reading their opening and posture |
| `raid.py` / `guard.py` | Raid squads at enemy expansions; the outpost at our forward expansion |
| `expand.py` / `oregrowth.py` | Our expansions (deploy spots clear of growing ore); enemy expansion sites |
| `harvest.py` | Harvesters steered to fields that still have ore; the ore forecast |
| `micro.py` / `movement.py` | Focus fire, siege on turrets; unsticking units |
| `orders.py` | Order builders, shape checks, splitting into batches |
| `opponents.py` / `postgame.py` | Opponent profiles across games; Claude's analysis of each opponent between games |
| `sparring.py` | TEST ONLY: self-play archetypes (never in a tournament) |

## The plan

### 1. Economy

- **Opening**: power, barracks, refinery, refinery, power, war factory (the
  pros' double refinery), its money set aside. Then the service depot (it
  unlocks the main battle tanks) and a third refinery, unless outgunned; more
  barracks only after those. Under a rush read everything goes into units —
  except a lost war factory's rebuild and the harvesters in production.
- **Harvesters**: refineries + 2, at most 3 per refinery. **Refineries** by
  game time, spread over the fields by load (at most 3 per field).
- **Expansions**: to the natural after the depot (from ~3:45), then every 2
  minutes or sooner when the ore forecast runs dry; up to 3; none after 14:00.
  The MCV is escorted by the outpost, never the army.
- **Production matches the army**: one infantry queue and one vehicle queue
  (more buildings only speed them up), and the vehicle queue carries tanks,
  harvesters, the MCV and siege units. So a second war factory once the depot
  stands (from 6:00, its price in hand) and at most 3 barracks.
- **Behind on army** (odds at most 1/1.3) or outgunned: units first — no money
  held for an MCV, refineries, the depot or the dome. A lost war factory is
  rebuilt first. Broke with no harvester: sell a spare building.

### 2. The army's make-up

Half the combat value in main battle tanks (the strongest offered, never
light tanks once better ones exist), riflemen and rockets 3:1 shifting to 1:1
as their armor grows — judged from the enemy units our scouts saw in the last
3 minutes, not just those in sight. Siege units once enemy turrets are known; against
a turtle about one per turret (3–6) before any attack.

### 3. Information

- **0–3 min**: two scouts round the back of their base: their opening. A
  scout still alive after a circuit (these, and every later base check) goes
  on to look where their army masses (the mouth of their choke) before it
  returns to its post: their army's size and make-up, and a fresh look for
  the attack's blind check.
- **Routes**: every scout trip is walked in attack-move legs of 8 steps round
  the chokes, the front of their base (15 cells, 70° either side of their way
  out) and 8 cells round every turret we know; circuits start at the side of
  their base (never across the front); posts and looks stand outside those
  places (82bdca3e: the left scout's road ran through the gap, the right one
  cut across their front, and the flank picket died five times in the gap).
- **From 5:00**: pickets on the flanks of their exit (never the centre lane),
  the lane scouts peek at their exit every 45 s, one runs round their base
  every 2.5 min, a roamer checks remote ore fields for expansions.
- **Rush read**: one refinery with an early war factory or second barracks;
  600 of army seen before 2:30 or 1000 before 4:00; or two scouts lost early.
  Turrets are not army.
- **Turtle read**: 4+ ground turrets and no push of theirs for 5 minutes →
  the dome at once, siege units, expansions early.

### 4. Judging a fight — one way

`strength.odds`: each side's damage per second against the other's armor mix
times its hit points (Lanchester); 1.0 is even and it reads like a value
ratio. Turrets fight in it. Their army is what we saw lately less half our
kills since — no guess; the part we believe in but can't see counts as a
mirror of our own mix. Every decision reads it: attack, retreat, meeting a
push, raids, the base's defense, behind/outgunned.

### 5. The army's one decision each view (first match wins)

1. **DEFEND**: a threat at the main base worth more than a small raid (1500)
   that what defends it (units near it, turrets) doesn't beat; any rush before
   3:00. Expansions never call the army.
2. **HOLD**: an early push warning, a rush read or the scout alarm: mass at
   home, turrets up; an attack under way pulls back.
3. **FINISH**: their base razed, or we dominate (2x their kills and 5000 up)
   with odds of 2.5+.
4. **PROTECT LEAD**: the last 2 minutes with a real lead.
5. **RETREAT**: our attack is losing (odds under 0.8, or losing the trades
   after the first 15 s).
6. **ATTACK** when the odds against their army plus the turrets at the target
   reach the bar: 1.3, 1.15 after 7:00, 1.05 with a 15000+ army, 1.0 behind
   at the end or on the advisor's "attack". Or at once after the main army
   clearly won a fight (1500+ net and 1.5x in 2 minutes) when at least even.
   - **Look first**: knowing little of them (under half our army seen lately,
     and no fresh look where their army masses — the mouth of the choke on
     their side; a lap round their base doesn't count), a scout looks first
     (up to 40 s); on even trades, a second look 4 s later (cb4bfc36 attacked
     on a lap and met the riflemen massed at their exit).
   - **Never**: before 3:00 and a war factory, a new attack while the MCV
     travels, within 60 s of a retreat, into a turtle without 3 siege units
     (unless the odds are 2.5+).
   - **Raid**: while our raid squads hold a real share of their army, the
     main army is judged against the rest and attacks away from the raid.
7. **MASS**: otherwise.

An attack goes on until the retreat rule ends it. No mass/attack flip within
40 s.

### 6. Execution

- **Orders**: attack-move, except retreats and a raid breaking off (a plain
  move, to get away), scouts and the MCV. At most 20 orders a batch; what
  doesn't fit goes first next batch; lost orders are sent again.
- **Defense, one handler** (`executor.respond`, `executor._army_home`):
  - **The army at home**: a fight in contact with it or the main base → it
    attacks when the odds are 1.25+, else holds the main base's turret line
    (`scout.home_line`: just behind our turret nearest them). A push seen in
    our half → it gets ready at that line facing it. Otherwise the concave
    holds the choke on their road in (a passage 7 cells or narrower, 10+ cells
    out and under 45% of the way to them): its kill point on the choke's mouth
    on our side, so the whole arc fires on them as they come through; with no
    such choke, at the massing point. Rally points stay at the massing point.
  - **Everything else** — a small group, a poke at the concave, a raid on the
    trucks, a spot we were hit at with no attacker in sight
    (`under_attack`/`unit_lost` events, vanished units) → the idle units
    within 12 cells (and, for a small group at the main base, the idle units
    at home).
  - **Expansions**: the outpost (2 infantry, the engine's guard order), one
    turret and the idle units nearby. The army never walks out to one.
- **Attack**: three groups in a semi-concave across the target; stragglers
  wait to go 6 together; no flank before 10:00 unless far ahead. The route
  goes round the chokes on the road between the bases (`terrain.choke_zone`:
  passages 7 or narrower, padded 4 cells, not the bases' own exits) when the
  way round is at most 1.6x as long, as queued attack-move waypoints 10 steps
  apart so the units can't path back through (82bdca3e's first attack went
  through Trapped's gap into their concave).
- **Raids**: only at enemy expansions our scouts found; more squads the
  longer a healthy game goes (at most 35% of the fighters); they kill
  defenders first, break off when outmatched, and join the main army's fight
  as a second prong.
- **Retreat**: a plain move to the main turret line; regroup there.
- **Micro**: focus fire on the weakest enemy fighter within 4 cells; siege
  units shell turrets.
- **Turrets**: a line on the base's front edge straight toward the enemy
  (side by side across that line, within 35° of it, never behind the base);
  when it is full, no more are built until the base grows that way.

### 7. The end

Protect a real lead in the last 2 minutes; level or behind in the last 3,
attack regardless; once their base is razed, fan out and hunt every last
building.

## Platform limits

No capture order (no engineer captures, no oil derricks), no mine laying, no
order to send a unit into a repair bay. A building asked for on ore, on a
unit, or farther than the build radius is silently moved next to the main
yard.

## Lessons from matches

**Decisions**
- Rules decide, the advisor advises: its staging fields were dropped (a2952a6d
  massed toward our ore, facing our base); its army target counts only with a
  known enemy within 10 cells (b7fcbd4e).
- No guess at their army: the fog prior and the rebuild guess both misled
  (ee2ac21b attacked 5150 into "4440" and met 13500; efefdca8 attacked 13150
  with 700 seen). Little seen → a scout looks first, from the first attack.
- Commit: an attack goes on until the retreat rule (82908a24 flipped 9 times);
  in the first 15 s of contact only being clearly outmatched pulls back
  (5e840776).
- Decisive after a won fight (3bd6e450 won 16600 to 11150 and sat 3 minutes)
  — but only the main army's fights count, not a raid's kills.
- Price lies: riflemen barely scratch tanks (52a77fb1: "1.25x" by price was
  even by strength); turrets fight (efefdca8, 654f1b2e: attacks into flame
  towers); turrets are not army (pillboxes read as a rush; a turtle's ring
  made us "behind").
- Finish a beaten enemy: no protect-lead against them (654f1b2e, 4bc1ee71);
  hunt with groups over the whole map (c4caa45e, 0561274e, 2035c6b3).
- Only threats at the main base recall the army; two stragglers recalled a
  winning attack six times (feba5f26).

**Defense and the army's position**
- The army holds the main choke; it never escorts an MCV or walks out to an
  expansion (52a77fb1 left the main open for two minutes).
- Meet a push you can't beat in the open at the turret line, not in front of
  it (52a77fb1 lost 6400 to 2200 five cells in front of its pillboxes;
  6b457914 charged 22 rushers with 9 units).
- The massing army must answer what shoots it (3bd6e450, 08560d9b); idle units
  answer, units with orders keep them.
- Retreat with a plain move: an attack-move stops to shoot every chaser.
- The concave: ranks within rifle range of a kill point 2 cells in front, at
  most 6 slot orders a view (3bd6e450's 16-cell line couldn't all fire;
  52a77fb1's per-unit orders overflowed the batch).
- Steady orders: no re-send within 5 cells for 20 s unless idle away from it
  (906dbb32); stragglers go 6 together (5e840776).

**Economy and production**
- Wait at the first unmet build-order step; power never waits behind a big
  building; a harvester never waits behind tanks (28e54539, e4caa28c).
- A ready-but-unplaced building counts as pending, and every finished one is
  placed (9a9cd98e, 14a78c43).
- Units spend only what the economy doesn't need; behind on army, units first
  (52a77fb1 banked 1200-3800 at 2-3x behind).
- Expand to the nearest field by walking (b7fcbd4e); even games keep expanding
  (ef66403c); running out of ore overrides the advisor's "wait" (c4caa45e).
- Fewer trucks than refineries: income first (6b457914).
- The strongest tank offered (52a77fb1 built light tanks all game as England).

**Placement** (OpenRA's BuildingUtils/Building rules; checked against 266
logged placements)
- Every footprint cell (bibs included) on clear ground, no actor on it, within
  one free cell of a building that extends the buildable area (not turrets),
  within 13 cells of a yard. The observation reports centre cells, placement
  takes top-left (fd507a58).
- One candidate source (`legal_spots`), roles only score; nothing on the
  army's lanes, the massing area or a refinery's road (a2952a6d, 159f4f11).
- Units on a spot step aside first (a1329ceb); a spot the engine moved a
  building from is skipped for 3 minutes (89356b51).
- Ore grows from its mines all game (oregrowth.py): MCV spots look a minute
  ahead; an expansion refinery that lands elsewhere is asked for again at once.

**Scouting**
- Out along a flank, never down the middle; round the back of their base on
  two rings (6b457914 missed a second barracks on one ring).
- Scouts look, they don't stand at corners (3bd6e450); pickets watch the
  exit from the flanks.
- Raids only at expansions our scouts found (52a77fb1 fed squads to their main
  field); a target that cost a raid is left alone for 4 minutes (efefdca8).

## Sources

- Platform guide: https://api.altruagent-game.com/skill/redalert
- https://github.com/OpenRA/OpenRA/wiki/How-to-be-a-better-OpenRA-player
- https://github.com/OpenRA/OpenRA/wiki/The-New-Players-Guide-to-OpenRA
- https://forum.openra.net/viewtopic.php?p=313702 (Multiplayer Guide 2022)
- OpenRA `mods/ra/rules` and `mods/ra/weapons` (strength.py, placement)
- 9 RAGL/ladder replays (`ora-replays/`, not committed): build timings,
  refinery-heavy economy, unit mix, expansion to the natural.

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
