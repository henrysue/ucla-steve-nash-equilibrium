# Red Alert agent: review and simplification plan

Goal: one simple, robust strategy that is easy to reason about, with code
and tests that say the same thing. This plan comes out of a review of the
code as it stands (Oct 10, after the scouting/intel work) and of the last
dozen match logs.

## Where we are

- **6,400 lines in 24 modules**, the biggest being `buildplan.py` (800),
  `expand.py` (700), `executor.py` (650) and `agent.py` (550).
- **274 tuning constants** in `config.py`. Many were set once from a single
  match and never revisited.
- **305 Red Alert tests (3,200 lines, one file).** The other ~1,400 tests in
  `tests/` cover the platform client, tournament runner and VGC agent and
  are out of scope here. Many Red Alert tests are named after the match that
  prompted them and pin a rule we have since changed or replaced.
- **What works in recent games:**
  - We win most self-play and the Ryanxhuang game outright.
  - Placement is mostly legal since the coordinate fix.
  - Early defense holds, and the finishing hunt works.
- **What hurts:** rules were added one match at a time. Several mechanisms
  now overlap, some override each other, and a few interact in ways no single
  rule's author intended. Examples follow.

## The strategy, on one page (what the code should express)

1. **Economy:**
   - **Opening:** the pros' double-refinery build order, pre-funded. Units
     spend only what the economy doesn't need.
   - **Expansion:** expand to the natural, then further out.
   - **Refineries:** at most 3 per field.
   - **Spare money:** a second war factory when money piles up.
   - **Broke:** sell spare buildings when there is no income.
2. **Information:**
   - **First 3 minutes:** two scouts round the back of their base.
   - **From 5:00:** pickets across the approach, plus a base check every
     2.5 minutes.
   - **Intel:** reads their opening (rush or economy) and an estimate of
     their army.
3. **One decision each view:** the first matching state wins.
   1. **DEFEND:** a real threat at a base, more than home units and turrets
      can handle.
   2. **HOLD:** an early warning or a rush read. Mass in a concave at home
      and put turrets up.
   3. **FINISH:** their base is razed or we dominate. Destroy everything.
   4. **RETREAT:** our attack is losing. Pull back fighting, then regroup.
   5. **ATTACK:** our units clearly outweigh theirs and we can break the
      turrets in the way. When trades are about even (within 200), the
      numbers must still hold after a 4-second look; once confirmed, we
      commit fully.
   6. **MASS:** otherwise. Concave at the massing point, raid harvesters.
4. **Execution:**
   - **Every army order is an attack-move.** Attacks go in a semi-concave of
     groups.
   - **Raid squad:** fights defenders first and pulls back when outmatched.
     It is also the second prong of an attack.
   - **Home responders:** units at home kill small raids.
   - **Placement:** buildings go only on legal spots, clear of the army.

Everything below is measured against this page: does a mechanism express one
of these lines, and is it the only one doing so?

## Findings

### Corrections (bugs or likely bugs)

1. **Two owners order harvesters.**
   - **Problem:** `Expansion._harvest_orders` sends 2 trucks to each
     expansion, while `HarvestManager` rebalances all trucks by refinery
     share. In the same view they can give one truck different fields.
   - **Fix:** `HarvestManager` alone owns harvester orders, treating an
     expansion's field like any other field with refineries.
2. **Several scouts now, but code still assumes one.**
   - **Problem:** `brain._regrouped` and the advisor context still take a
     single `scout_id`, so the other scouts count as army when we check
     whether the army is back after a retreat.
   - **Fix:** pass the set of squad ids everywhere.
3. **Two massing-point overrides collide.**
   - **Problem:** the attack probe (look before committing) and the MCV
     escort each move the massing point. When both apply, the probe silently
     wins.
   - **Fix:** one function decides the massing point, with a clear priority.
4. **The commit rule and the dwell rule interact.**
   - **Problem:** an attack call that is cancelled during its 4-second
     confirm starts a 25-second "no flip" dwell, so a real chance a few
     seconds later must wait.
   - **Fix:** a cancelled confirm doesn't count as a mode change.
5. **The rush read never triggers against our test rusher.** It builds two
   refineries before massing. Either give the rusher a one-refinery build
   (a better sparring partner) or accept that self-play only tests the
   scout alarm path.
6. **Leftover safety nets from before the coordinate fix:**
   - "Avoid the spot the engine moved a refinery from".
   - "Give up after 2 moves".

   The placement audit shows near-zero relocations now. Keep the audit,
   delete the nets after two clean games.

### Simplifications

1. **Decision logic lives in two places.**
   - **Now:** `brain.choose_mode` has about 12 rules. `agent.py` then
     overrides its answer five more ways: the test rusher, the expansion
     escort, the confirm step, the anti-flip dwell, and the stall breaker
     (via advice).
   - **Change:** one decision module implementing the six states above, in
     order, with the confirm step and dwell inside it. `agent.py` only wires
     data in and orders out.
   - **Win:** one place to read when a game does something odd.
2. **Money has about 14 separate rules:**
   - the build-order reserve;
   - the money bands (low, rich, very rich);
   - queue depth;
   - the harvester target;
   - the time-based refinery schedule;
   - the per-field refinery cap;
   - the saturation refinery;
   - silos;
   - the second war factory when banked;
   - the barracks schedule and barracks money gate;
   - the dome money gate;
   - the turret money gate;
   - selling for income;
   - the broke-expansion push.

   **Change:** one budget function each view, a priority list paid from
   the funds left after the reserve: economy steps, harvesters, the next
   production building, units, turrets, tech. The rest of the code just
   asks it "can I afford X?"
3. **Four ways to split the army:**
   - the two-cluster split (`_split_attack`);
   - the hunt groups;
   - the attack semi-concave;
   - the massing concave.

   **Change:** one function that splits units into groups and gives each
   group a target and a facing. The hunt, a two-cluster attack and the
   semi-concave become three inputs to it.
4. **Five things defend the base:**
   - DEFEND mode;
   - home responders;
   - the home guard (2 riflemen parked at the power plants);
   - the expansion garrison;
   - turrets.

   The home guard predates home responders, which now do its job better.
   **Change:** drop the home guard; keep responders, the garrison and
   turrets.
5. **Seven placement mechanisms:**
   - placing by role;
   - growing from the base (`grow_spot`);
   - a refinery beside its ore (`_beside_ore`);
   - the turret fan (`perimeter_spot`);
   - the keep-clear zone and lanes;
   - stepping units aside;
   - preferring spots no unit stands on.

   **Change:** one generator of legal spots (exactly one free cell from a
   real building, buildable, clear of lanes and the army), plus a scoring
   function per role (toward the enemy, toward ore, the back, the side).
   Units still step aside before a building is placed.
6. **Three things go looking for the enemy:**
   - `scout.Hunter`, the end-game search;
   - the scout corps;
   - the raid's targets.

   **Change:** at the end, scout pickets sweep the hunter's search points;
   the hunter just ranks them.
7. **The expansion state machine is 700 lines.** Much of it handles
   failures we have since fixed at the source: the coordinate bug, lost
   yards, relocated refineries. Re-cut it to:
   1. pick a field;
   2. scout it;
   3. train an MCV;
   4. escort it;
   5. deploy;
   6. build its refinery.

   That also lets us drop the now-redundant retries.
8. **The model layers.**
   - **Advisor:** it gives about 10 fields, and the rules now override or
     ignore most of them:
     - its army target is ignored while finishing, after a stall, or
       without a known enemy near it;
     - the formation is forced to a concave;
     - its massing point is limited;
     - its "hold" is limited;
     - its attack route only applies after 5:00.
   - **Squad leads:** the rules override them in every fight.
   - **Cost:** both add latency, cost and nondeterminism.
   - **Recommendation:** run 5–10 test games with the models off and compare
     results. If the rules win as often, keep the advisor only for the match
     log commentary and remove the squad leads.
9. **Constants.** Cut `config.py` to the few dozen that matter, grouped by
   the strategy page, with a comment saying what each controls and why.
   Constants used once go back into the code beside the rule they tune.
10. **Match log.** Samples carry about 40 fields. Keep what we actually read
    in reviews:
    - mode;
    - money (cash plus stored ore);
    - harvesters per field;
    - the placement audit;
    - expansions;
    - scouts and intel;
    - the confirm and stall flags;
    - kills and losses.

    Add the enemy-army estimate and the budget decision.

### Optimizations

- **Repeated work each view:** `choke_cell` (the massing point) is computed
  about 10 times per view, and each call scans buildings. Compute it, the
  enemy-army estimate and the legal-spot set once per view and pass them
  down.
- **Placement and `perimeter_spot` scan every spot round every building.**
  Fine at 25 buildings, but `defense_orders` calls it every view. Cache it
  per view.
- **Hot-path latency:** profile one decision on a late-game view. A decision
  should take under 20 ms.

### Test-only code

`testrush.py` and its 15 marked lines stay for now: it's our sparring
partner. Remove it before the tournament (grep `testrush`). The rush read
fix (correction 5) may change its build.

## Tests: cleanup plan

The Red Alert tests become a specification of the strategy page, not a
history of matches.

1. **Split by area,** into `tests/redalert/`:
   - `test_economy.py`: build order, reserve, harvesters, refineries,
     expansion.
   - `test_placement.py`: legal spots, roles, stepping aside, the audit.
   - `test_intel.py`: scouts, pickets, the opening read, memory, the hunt.
   - `test_decisions.py`: the six states and their order, confirm, dwell,
     stall.
   - `test_army.py`: attack-move only, concave, semi-concave, raid prongs,
     responders, retreat.
   - `test_platform.py`: order shapes, the observation parser (the centre
     cell offset, legend), refusals.
   - `test_match_maps.py`: replays of real map pictures from our logs
     (sidestep, trapped). Every growth and refinery spot must be legal; scout
     circuits and posts must be reachable; the natural expansion is chosen.
     The map pictures are copied into small fixtures under
     `tests/redalert/fixtures/`.
2. **One test per behaviour on the strategy page,** written against public
   functions (decide, budget, spots, formations, scouting) rather than
   internals. That's about 100–140 tests, down from 305.
3. **Delete tests that:**
   - pin a constant's exact value;
   - test removed or superseded rules;
   - duplicate another test;
   - need private fields poked into objects.

   I'll list them before deleting them.
4. **Keep the whole Red Alert suite under 2 s** (it's 2 s today), so it runs
   after every change.

There is no git history in this folder. Before deleting anything I'll copy
the current tests aside, or better, initialise a git repository first so
every step can be undone.

## Order of work

Each step ends with the tests green, a self-play match against the rusher,
and a short comparison with the baseline log on placement moves, idle money,
early warning, kills/losses and the result.

1. **Corrections 1–4,** small and safe.
2. **The decision module** (simplification 1), with its tests rewritten as
   part of `test_decisions.py`.
3. **The budget** (simplification 2) and its tests.
4. **Formations and base defense** (simplifications 3–4).
5. **Placement** (simplification 5), plus the real-map replay tests.
6. **Scouting and hunt merge, and the expansion re-cut** (simplifications
   6–7).
7. **Model layers:** the A/B test, then keep or remove (simplification 8).
8. **Constants, match log and the per-view cache** (simplifications 9–10,
   optimizations).
9. **Remaining test cleanup:** finish moving tests and delete superseded
   ones.
10. **Before the tournament:** remove `testrush`.

## Decisions needed

1. **Tests:** OK to replace the 305 Red Alert tests with about 120 tests
   organised by area, deleting superseded ones after I list them?
2. **Models:** run the models-off A/B test, and remove the squad leads if
   the rules do as well?
3. **Version control:** initialise git in this folder before we start?
4. **Priority:** follow the order above, or start somewhere else (for
   example the budget, since money is the recurring complaint)?
