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
- **Models advise off the hot path**: the advisor (`advisor.py`, Sonnet) sets
  posture, targets, staging, formation, defense side and expansion timing every
  ~6 s; Haiku squad leads (`squadlead.py`) direct the raid squad and the
  garrison every ~4 s. Advice is validated and expires; safety rules win.
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
| `raid.py` / `guard.py` | Harvester raids; home guard and expansion garrison |
| `expand.py` | Our expansions; enemy expansion sites |
| `harvest.py` | Harvesters steered to fields that still have ore |
| `micro.py` / `movement.py` | Focus fire, pull back damaged units; unstick units |
| `orders.py` | Order builders and shape checks |
| `testrush.py` | TEST ONLY: the self-play sparring rusher (remove before the tournament) |

## Strategy

- **Opening**: power, barracks, refinery, refinery, power, war factory,
  refinery, service depot (the pros' double-refinery order: refineries
  1:00 / 1:38, war factory 2:40, depot 3:50); only 5 infantry until the war factory.
- **Economy**: trucks come with refineries (target = refineries + 2); more
  refineries over time; expand right after the service depot (~4:45), up to 3
  times, sooner when ore runs out. Money bands: broke = one queue; rich =
  spend harder.
- **Army**: infantry-heavy (about 3 riflemen per rocket soldier) from up to 5
  barracks, tanks as support from one war factory.
- **Fighting**: mass on our side of the map, attack with an edge, flank when
  the middle is guarded, target power then production, focus fire, retreat
  from losing fights (fighting back when pressed), hunt the last buildings.
- **Squads**: roaming scout, harvester raid squad, home guard at the power
  plants, garrison at the newest expansion (joins pushes once it's complete).
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
- Mass (and rally new units) just in front of the base, between the harvesters and the enemy: 8 cells out from base+ore, 3 if enemies are near; never send units one by one into contested ground.
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
- At most 3 refineries per field; the next one goes to a field with room, or the base crawls toward one.
- Once their base is nearly razed (we knew 6+ enemy buildings, now 3 or fewer) and we lead, finish them: no protect-lead or hold; with none known the hunter aims the army and a big army sweeps 3 points at once (0561274e idled on ruins for 6 minutes).
- The advisor's army target counts only with a known enemy within 10 cells of it (b7fcbd4e: its stale target kept a 40k army 10 cells from the enemy's last building for 5 minutes).
- Stop buying refineries for an expansion after 2 have been moved home.
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

- `testrush.py`: in a self-play test match (no tournament id, both seats on this machine), seat 1 rushes —
  a rush build (2 refineries, 3 barracks, war factory), units without pause (infantry 3 deep, tanks 2),
  no expansion or turrets; it masses a ball and launches at 2500 (or 1500 by 3:30), pushes until the army
  is below 40% of its launch value, then re-masses — to test our scouting, early-push warning and defenses. `RA_TEST_RUSH=0` turns it off, `RA_TEST_RUSH_SEAT`
  picks the seat. Remove the module and every line marked "testrush" (agent.py, executor.py, tests).
