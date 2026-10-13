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
  `<REDALERT_LOG_DIR>/opponents`) — in tournament matches only (a tournament
  id): practice and test games neither read nor write it, nor run the
  post-game analysis. Each answer gives a confidence and its
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
  pros' double refinery), its money set aside — but the war factory waits
  for an army worth 2000 or 2:30, whichever comes first, so the barracks
  keeps training for the push at ~3:00 (2fe84e1f and 9cada749: the barracks
  idled saving for it, and the push took the war factory); not against a
  passive enemy. The first war factory goes at the back of the base, as
  far from the enemy as it reaches (4+ cells off the ore, its exit open):
  on the flank it was the first building their push reached, lost at
  3:39-3:49 in four of five games against the rusher. Then the service depot (it
  unlocks the main battle tanks) and a third refinery, unless outgunned; more
  barracks only after those. Under a rush read everything goes into units —
  except a lost war factory's rebuild and the harvesters in production.
- **Harvesters**: refineries + 2, at most 3 per refinery. **Refineries** by
  game time, spread over the fields by load (at most 3 per field), each
  beside the first field with room whose trucks walk to its ore in 6 steps;
  with none such (the base boxed in round its fields), no refinery now
  (d2d8dc27 on Shadowfiend paid 2000 for one 10 steps from its ore).
- **Expansions**: to the natural after the depot (from ~3:45), then every 2
  minutes or sooner when the ore forecast runs dry; up to 3; none after 14:00.
  Against a passive enemy (below): from 3:00 and every minute, whatever our
  army odds or the advisor's "wait", and refineries keep coming.
  The MCV is escorted by the outpost, never the army. While it drives there
  no order sends our units onto the yard's spot (posts, concave slots, rally
  points land beside it), and anyone standing on it steps off; the MCV waits
  10 s for them before trying elsewhere (d2d8dc27: 170 s of shuffling).
- **Production matches the army**: one infantry queue and one vehicle queue
  (more buildings only speed them up), and the vehicle queue carries tanks,
  harvesters, the MCV and siege units. So a second war factory once the depot
  stands (from 6:00, its price in hand) and at most 3 barracks. Money
  floating — 1000+ beyond the build order's needs for 30 s, the queues can't
  spend the income — buys the second war factory without its price in hand,
  then a third, then barracks up to 3 (d2d8dc27 floated 1000-3000 from 11:00).
- **Behind on army** (odds at most 1/1.3, or the advisor's "more_army" past
  3:00) or outgunned: units first — no money held for an MCV, refineries,
  the depot or the dome. On "more_army" the follow-up's third refinery and
  the expansion to the natural still go; later expansions, refineries past
  the third, the kennel and the dome wait (9cada749 built them through two
  minutes of "more_army" while the rusher grew from 2350 to 10100 against
  our 5900). A lost war factory is
  rebuilt first. Broke with no harvester: sell a spare building.

### 2. The army's make-up

Half the combat value in main battle tanks (65% as the Soviets, whose heavy tank trades efficiently; 65% against a tank push not yet beaten; 70% against a passive enemy,
with the second war factory at the depot and a third once two expansions
stand: we out-build it) (the strongest offered, never
light tanks once better ones exist), riflemen and rockets 3:1 shifting to 1:1
as their armor grows — judged from the enemy units our scouts saw in the last
3 minutes, not just those in sight — and at most 2:1 against a tank timing
push (intel.tank_push: their war factory seen before 3:00, their depot before
4:00 or their main battle tanks before 5:30, held until 10:00; Yuriko's build,
yuriko-vs-abu, pushes 5-6 medium tanks at 4:30-5:30). A scout looks at their
army at 3:20 (after the early circuits); what it sees goes to the advisor as
scouting.army_report, with rules_read.tank_push. Siege units once enemy turrets are known; against
a turtle about one per turret (3–6) before any attack.

### 3. Information

- **0–3 min**: two scouts round the back of their base: their opening —
  the second 30 s after the first, and on into the centre of their base.
  They go wide: out along a flank bowed 45% of the base-to-base distance,
  in from 8 cells beyond the ring at the side of their base, the outer ring
  16 cells round it (the early pairs died in the middle lanes at 1:00-1:30).
  What they see there in the first 4 minutes (riflemen and every unit type
  counted by unit, the buildings present) goes to the advisor as
  scouting.base_report (daa0e4d5 believed 1220 at 2:51 against a real
  4300). A
  scout still alive after a circuit (these, and every later base check) goes
  on to look where their army masses (the mouth of their choke) and, if
  nobody is there, on into the front of their base (a turtle's army waits at
  home) before it returns to its post: their army's size and make-up, and a
  fresh look for the attack's blind check — which counts only once it saw
  at least half of what we believe of their army, or stood in front of their
  base (d2d8dc27 counted an empty spot and attacked 3250 on an estimate of
  250; 1da6311e counted a glimpse of 1450 of a real 9750).
- **See and turn back**: a look or a peek ends the moment the scout sees
  their fighters (9 cells) and it backs off; a look counts only if it saw
  them or stood in front of their base — an empty exit isn't a look. After
  3 scouts lost in 2 minutes, no looks or peeks for 90 s (5a4bc83f lost 58
  scouts, the left one every ~30 s, peeking into their army).
- **Never down a choke**: a scout never walks a choke corridor between the
  bases; a place inside one (their army's spot at its far mouth) is looked
  at from the nearest spot beside it reached the long way round, else not at
  all — and a look that reached nothing doesn't count (ae03adbd: scouts
  walked Sidestep's gap into their concave every 15-30 s, ~30 of them).
- **Routes**: every scout trip is walked in attack-move legs of 8 steps round
  the chokes, the front of their base (15 cells, 70° either side of their way
  out) and 8 cells round every turret we know; circuits start at the side of
  their base (never across the front); posts and looks stand outside those
  places (82bdca3e: the left scout's road ran through the gap, the right one
  cut across their front, and the flank picket died five times in the gap).
- **From 3:00** a roamer checks remote ore fields for expansions — from
  beyond each field's four extreme ore cells, since a yard can stand 12 cells
  from a big field's centre (d2d8dc27 missed one so); from 8:00 a second
  roamer joins it. Each expansion found is logged and flagged for the raids,
  which harass it while the main army stays near home.
- **From 5:00**: pickets on the flanks of their exit (never the centre lane),
  the lane scouts peek at their exit every 45 s, one runs round their base
  every 2.5 min — once 3 scouts have died on their side of the map, every 5
  min and peeks every 90 s: scouting turns to their expansions.
- **Rush read**: one refinery with an early war factory or second barracks;
  600 of army seen before 2:30 or 1000 before 4:00; or two scouts lost early.
  Turrets are not army.
- **Turtle read**: 4+ ground turrets and no push of theirs for 5 minutes →
  the dome at once, siege units, expansions early.
- **Passive read**: from 4:00, no attack on us for 4 minutes — no building
  lost, no enemy army (1000+) within 8 cells of a building of ours; their
  army at our concave and their units chasing our retreat (90 s) don't count
  → expand early and often, 70% tanks, more war factories; the advisor's
  "aggressive" read puts up no turrets and its "more_army" doesn't hold the
  refinery and MCV money back (e2bc988e: a turtle took 10
  harvesters to our 7 and out-built us while we held for a push that never
  came).

### 4. Judging a fight — one way

`strength.odds`: each side's damage per second against the other's armor mix
times its hit points (Lanchester); 1.0 is even and it reads like a value
ratio. Turrets fight in it. Their army is what we saw lately or, if more,
what they could have built since: it grows by all of what we spend on our
army — they produce at our velocity (the plan, October 13; it reads them
~1.8x high overall on the logs, ~3x when we are well ahead: the main army
defends and attacks only when clearly ahead) — scaled down by what we destroyed of their
income and production (memory.growth_scale: the lesser of the refineries —
2 harvesters each — and harvesters, and the barracks and war factories,
left against what they had; each loss fading over 3 minutes as they
rebuild unseen; never under 0.2) — and loses half our kills; it stops
growing once their main base is razed and the unseen part fades while our
main army (not raiders or scouts) stands in their main base and finds nothing
(5a4bc83f: raiders by an expansion yard faded it from 14480 to 8300). (Seen-only read 1.3-12x low on
our self-play logs; this is unbiased there with half the error.) The part we
believe in but can't see counts as a mirror of our own mix. Every decision
reads it: attack, retreat, meeting a push, raids, the base's defense,
behind/outgunned. The blind check reads only what was actually seen.

The last-seen map (lastseen.py) mirrors what we last saw where: 8-cell
squares, each stamped every view our vision covers it (and scouted when a
scout does) with the enemies in it then — an empty look clears it, a unit
seen elsewhere moves there. It goes to the advisor only
(situation.last_seen_map: the army as last seen and how old, their base's
square, the squares with enemies and their ages, the share of the map seen
in the last minute) as evidence beside the estimate; it never changes it.

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
   reach the bar: 1.3, 1.15 after 7:00, 1.2 with a 15000+ army (at parity the
   defender wins: d1615ac6 lost 6800 for 3950 at 1.05), 1.0 behind
   at the end. Trading up is no reason to go in (9cada749 went in at 1.06
   after trading 15250 for 10800, was out in 31 s, and their next push took
   the war factory, refineries and depot).
   - **Look first**: knowing little of them (under half our army seen lately,
     and no fresh look where their army masses — the mouth of the choke on
     their side; a lap round their base doesn't count), a scout looks first
     — and without a look there is no attack (only behind at the end does it
     go without: b9145367's blind attack at 4:41 left 300 of 3100); on even
     trades, a second look 4 s later (cb4bfc36 attacked on a lap and met the
     riflemen massed at their exit).
   - **Never**: before 3:00 and a war factory, a new attack while the MCV
     travels, within 60 s of a retreat, into a turtle without 3 siege units
     (unless the odds are 2.5+), nor against a tank timing push (intel.tank_push)
     until our army has beaten a push of theirs (commander.push_beaten:
     traded up), unless 2.5+ or behind at the end — defend first (1da6311e
     attacked Yuriko at 6:34 on a believed 3370 and met ~9750). Under that
     read their unseen army grows by all of what we spend on ours, not 0.6.
     While behind or against it, the army doesn't edge forward to confirm.
   - **Strike an expansion** (off: `MAIN_ARMY_STRIKES`; the main army stays
     near home and raid squads harass their expansions): a remote expansion of theirs our army beats 2x
     (its turrets and the units we know there) is attacked even short of the
     whole-army bar, if we are at least 0.75x their whole army — they must
     come out of their concave to save it; no look needed (we know what is
     there). d1615ac6's raids killed one expansion's harvesters for two
     minutes while the army sat home at a believed 0.79x (really 1.3x).
   - **Through a choke with no way round** (terrain.choke_between): 1.5x
     the bar, unless 3+ siege units come to shell their concave.
   - **The advisor's "hold"** delays an attack the rules call by at most 90 s,
     then the rules go (c1c37a10 held 13 minutes at 1.4-1.9x while a turtle
     built 28800).
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
    attacks when the odds are 1.25+ (meeting them in the open 8-14 cells off, before contact,
    with 6+ units and no choke between: in a concave — three attack-moves, the centre just
    short of their middle, the wings at their flanks, our units split by where they stand; the
    attack's march meets their army the same way when the fight there is at least even), else holds. A push seen in our half,
    or defending → it holds too. Where it holds: the concave at the choke
    while they are still out in front of it (9cada749 left it for the base
    at 3:27 and 10:29, and the push took the buildings anyway); with no
    choke, or once they are inside (nearer our base than the concave), the
    main base's turret line (`scout.home_line`: just behind our turret
    nearest them). Behind on army (odds under 1), the waiting army holds
    there too — the concave at a choke, else the turret line — not out at
    the massing point (1da6311e on Shadowfiend: caught 22 cells out at 9:29
    by a push we knew was bigger). Otherwise the concave
    holds the choke on their road in (a passage 7 cells or narrower, 10+ cells
    out and under 45% of the way to them): its kill point on the choke's mouth
    on our side, so the whole arc fires on them as they come through; with no
    such choke, at the massing point. A fight at home is judged against
    the whole push as last seen (held 20 s), not the few of it in sight
    (2fe84e1f at 3:51 went at two of a 15-unit push and lost everyone).
    Rally points are at the massing point, but where the army holds while
    their units stand there, a push is coming or we defend: new units join
    the army, not walk out one by one into theirs (2fe84e1f at 4:05).
  - **A push is never answered in packets**: a group worth more than a
    small raid, or any of a push in progress, draws no idle units — it is
    the army's fight, whole.
  - **Only our half**: the army's own fight and the idle units' answers
    count only enemies on our half of the map, and a spot we were hit at
    only by our buildings or trucks — their concave at the far mouth of a
    choke, or a scout shot there, pulls nobody out (ae03adbd: hits on dying
    scouts sent pairs down Sidestep's gap, and their concave near those
    stragglers drew the whole army in: 6250 lost in 30 s while massing).
  - **At the concave** (while the army is home): nobody chases — its slots
    fire on what comes in range, and a real push is the army's fight above
    (e2bc988e chased pokes through Sidestep's gap into their massed army
    for ten minutes and lost 10000 more than it killed).
  - **Everything else** — a small group, a raid on the
    trucks, a spot we were hit at with no attacker in sight
    (`under_attack`/`unit_lost` events, vanished units) → the idle units
    within 12 cells (and, for a small group at the main base, the idle units
    at home), just enough to win clearly — counting the units already sent at
    it for 20 s or until they get there (c1c37a10: a lone scout drew two more
    every view until the whole concave stood at the back of the base).
  - **Expansions**: the outpost (2 infantry, the engine's guard order), one
    turret and the idle units nearby. The army never walks out to one.
- **Attack**: the army marches as one on one road — the walking road (or
  the way round the chokes on the road between the bases, `terrain.choke_zone`:
  passages 7 or narrower, padded 4 cells, not the bases' own exits, when at
  most 1.6x as long) as queued attack-move waypoints 10 steps apart, so no
  unit takes a road of its own round a rock (82bdca3e went through Trapped's
  gap into their concave; b9145367 split round Trapped's rocks when half the
  army was re-aimed mid-march). It leads to their buildings or base (or the
  advisor's target), never to whichever unit of theirs is nearest
  (d1615ac6 zig-zagged between Sidestep's lanes chasing those for 90 s), and
  takes a way round a choke up to 2.5x as long from where the army stands
  (from our concave the corridor is short and every lane looks long). A new
  route from where it stands when the target moves 12+ cells or the route runs out short of it (at most every
  15 s); the advisor's target stays for the attack while anything of theirs
  stands by it, and it leads onto the enemy building (or unit) nearest the
  spot named, not the spot itself (an attack-move to an empty spot stops
  there: a4a78f76 stood in their razed base 9 cells from their dome) — until
  the front half of the army reaches it, when the rules lead on to their
  buildings for 30 s (d1615ac6 stood 60 s on a reached point by their flame
  towers, 10150 against 2350, while they rebuilt). Only within 15 cells of the target does it spread into three
  groups in a semi-concave, and only to points it walks to without a detour.
  Stragglers wait to go 6 together; no flank before 10:00 unless far ahead.
  Scouts and raid squads walk in legs too.
- **Harvester snipe** (snipe.py): from 5:00 we keep 4-6 Rangers (else light
  tanks that aren't our main tank; none for the Soviets) out of the army.
  From 5:30, while their push is in our half or 2:30 after the last run,
  they slip by plain moves round their sight lines (the chokes, the front of
  their base, their turrets: the scouts' avoid zone; failing a way round,
  round the turrets only) to the flank of their main ore field, shoot the
  nearest harvester by id until it dies, and go home — at once when their
  fighters or turrets there outmatch them, after losing half, 25 s on
  target or 2 minutes in all. For an opponent with no expansion to raid
  (Yuriko takes none before ~6:40): a harvester is 1100 of tiebreak.
- **Raids**: only at their remote expansions — any building of theirs 25+
  cells from their main base (not its edge; a building there means a yard,
  even if the first one seen is its turret) — that our scouts found; more squads the
  longer a healthy game goes (at most 35% of the fighters); they kill
  defenders first, break off when outmatched, and join the main army's fight
  as a second prong. From 8:00 a found expansion is raided in force (8+
  units): the army doesn't just mass while their economy grows. Infantry
  never walks into a known turret's reach (7 cells): it goes round to a spot
  by the expansion the turrets don't cover (their field's far edge) by a road
  out of their reach, at most 2x as long; with every side covered the raid
  waits until it can take 2 tanks or siege units, and then still needs to
  clearly win (2x) against the turrets and units there. Raids that lose
  800+ net stop new squads for 2 minutes (c1c37a10: four raids into
  pillboxes lost 7300 for 1950).
- **Retreat**: a plain move to the main turret line; regroup there.
- **Micro**: focus fire on the weakest enemy fighter within 4 cells; siege
  units shell turrets.
- **Turrets**: against a tank push not yet beaten, one anti-tank turret (the
  Allied gun turret, a tesla coil; a pillbox and a flame tower are
  anti-infantry) and the rest into tanks (65% of the army's value); the
  expansion waits until the push is beaten (unless home runs dry). The first out in front of the first war factory (beside it
  on its enemy side; where its exit lane leaves no spot there, the line),
  unless a threat or the advisor names the side; the rest a line on the
  base's front edge straight toward the enemy (side by side across that
  line, within 35° of it, never behind the base); when it is full, no more
  are built until the base grows that way.

### 6a. The advisor's vote: near the bars only

Every rules bar is a ratio (odds), so it scales with any army and map. The
advisor's posture is a vote that counts only within 15% of a bar
(`brain.advised`): its "attack" fires an action up to 15% below the bar
(starting an attack — never under 1.0x —, striking an expansion, fighting on
instead of retreating), its "hold" stops one up to 15% above it (holding an
attack, pulling back from a fight). Beyond the band the numbers decide either
way; behind at the end or overwhelming, a hold doesn't count. The advisor is
shown each bar, our number and its band (`situation.decisions`), and the
enemy buildings our attacks destroyed (`counters.enemy_buildings_destroyed`:
by type, value, yards and refineries, when the last fell — memory.py counts a
building gone as a wreck or while we look at its spot): fd154ee2's advisor
called an attack that killed 4850 for 3050 "a costly all-in loss" and held.

### 6b. No getting stuck, no flip-flopping (the October audit)

Every latch has a way out and every threshold that flips on what is in sight
has hysteresis:
- A blind attack call waits for a look at most 2 minutes, then goes only at
  1.5x the bar on what we believe. A reached advisor target hands the attack
  back to the rules. A strike that took its expansion ends there unless the
  usual bar says go on. The advisor's 90 s hold restarts only after the rules
  stop calling an attack for 10 s; a call that drops for a moment keeps its
  confirmation and look for 5 s.
- The attack's destination is their buildings, not the nearest unit in sight.
  A fight at home is joined at 1.25x and kept above 1.0x for at least 10 s.
  The concave faces in 30-degree steps. Only idle stragglers wait at home.
- Scouts: a picket that backed off stays back 20 s; the roamer and the
  expansion visit move on from a place they can't reach; unreachable places
  are tried again after a minute; hunt points never reached are given up
  after 90 s, in a fixed order.
- Raids: units already sent count toward a group by its ids or where it
  stands; an outmatched squad is released (target marked failed); every
  squad, the first too, stays within 35% of the fighters and joins the main
  fight only from near; the economy's "healthy" reading holds 60 s (outgunned
  ends it at once); a new squad's route starts fresh.
- Rules agree: one remote-expansion definition (25+ cells) for raids, the
  strike and the attack's first target; the turtle gate holds for every way
  into an attack; an attack's 15 s before the 0.8 retreat line applies count
  from first contact, not from the call (a strike at 0.75-0.8 overall used to
  retreat while still marching); a distraction attack is judged on the odds
  it started on; "a way round"
  means the same ratio for the choke bar and the march; no forward expansion
  starts mid-attack; the follow-up's refinery needs a spot its trucks walk
  from; idle units answer a group at our buildings or trucks even across the
  half line; the passive read ends when a push comes into our half or their
  army grows to 1.5x ours.

### 7. The end

The tiebreak decides an unfinished game (kills - losses, then assets), so
from 15:00: a real lead (1500+) is protected — no new attacks — and level or
behind we force trades (attack at parity); in the last minute any lead is
protected; once their base is razed — or they are simply beaten
(overwhelming odds, well ahead on trades) with 10 or fewer buildings known —
fan out and hunt every last building.

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
  field); a target that cost a raid is left alone for 4 minutes (efefdca8);
  infantry goes round turrets or waits for tanks (c1c37a10).
- A look at their army goes on to the front of their base even after it sees
  them: a turtle's pillboxes stand there, out of the scouts' usual way
  (c1c37a10 saw none of eight until the last second: no turtle read, no
  siege units until 11:47).

**The advisor** (its prompt carries these as LESSONS FROM OUR GAMES)
- Holding is not free: a turtle or macro player out-builds a waiting army.
- Pillboxes eat infantry raids; aim at the side they don't cover, or wait for
  tanks or V2s.
- Turrets not seen are still there; a quiet enemy with few units is a turtle
  until a look at their front says otherwise.
- The opponent's history is a hypothesis (c1c37a10 kept "passive macro" from
  the last game for 13 minutes).

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
  - `RA_SPARRING=yuriko`: Yuriko's current build (yuriko-vs-abu x3, won all three by 6:45-10:10):
    power, one barracks, two refineries, power, war factory, depot, power; six riflemen then nothing
    until the war factory, 12-16 infantry in all, pillboxes at 2:40 and 3:35; from 4:05 tanks nonstop
    with rocket soldiers and riflemen alternating; the push from 4:30 with 3+ tanks, reinforced until
    down to 40% of its launch value, the next at 1.3x it (or 2:30 later); a second war factory at 5:54,
    a third refinery at 6:05, no expansion.
  - `RA_SPARRING=turtler`: our build plus a growing turret ring (2 at 2:30 up to 8 by 12:00), army at
    home, no raids; attacks only from 15:00 with a 1.3 edge, or with an 18000 army — or at once when the
    rules see a 2x edge (it punishes a blunder; the macro too).
  - `RA_SPARRING=macro`: refineries early (2 at 0:45, 3 at 2:30 ... 6 at 7:00), first expansion as early
    as allowed, a token army (1500) until 5:30, no raids; one big push at 12000 (or 13:00 with 1.15).
  - Every archetype still defends the usual way. `RA_SPARRING_SEAT` picks the seat (default 1); the old
    `RA_TEST_RUSH=1` means rusher. Three locks: the flag, no tournament id, and both seats of the session
    seen on this machine (marker files). Remove the module and the lines marked "sparring" to retire it.
  - A practice disguise: `RA_SPARRING_AGAINST=YurikoRed` (comma-separated names) lets this seat play the
    `RA_SPARRING` archetype against those named opponents off our machine — still never with a
    tournament id, only when the opponent's name (the context's, or `REDALERT_OPPONENT`) matches; anyone
    unnamed or else gets our real agent. Those games aren't saved to the opponent's history nor analysed.
