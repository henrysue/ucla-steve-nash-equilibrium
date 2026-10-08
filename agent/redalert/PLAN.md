# Red Alert Agent — Design Plan

A new agent for the `red_alert` game: **hyper-aggressive, fast-executing, and
tuned to win the 20-minute tiebreak** rather than a drawn-out late game. It
lives in `agent/redalert/` and is entirely separate from the VGC strategist.

Status: **PLAN — awaiting approval before implementation.**

---

## 1. The win condition decides everything

From `GAMES.md`:

- You win outright by destroying every enemy building **and** their MCV.
- Otherwise, at the **time limit (10–20 min, set per match)** — or after 10 min
  of mutual inactivity — a **tiebreak** decides:
  1. higher **`kills_cost - deaths_cost`** (value destroyed minus value lost),
  2. then higher **`assets_value`** (what you still own).
- Real time: 25 ticks/s, the world never waits, the runtime asks us again
  ~5×/s. **Thinking time is pure loss.**

**Implication (the whole strategy in one line):** we do **not** need to kill
everything. We need, by the clock, a **positive trade differential and an asset
lead**. That rewards: cheap-but-efficient units that trade up, relentless
pressure on the enemy army/economy, keeping our own losses cheap, and never
stalling. It explicitly does **not** reward map control, expansions,
veterancy, or tech-switch late games.

---

## 2. What we take from the strategy guides (selective extraction)

The DVOID guide and the OpenRA/C&C community guides are written for a skilled
human in an open-ended ladder game. We take the parts that serve *our* goal and
reject the rest.

**Adopt:**
- **Streamlined opening build order** (DVOID 2022), trimmed to one base:
  `Deploy MCV → Power Plant → Barracks → Refinery → War Factory → (Power/2nd
  Refinery as economy allows)`.
- **"Mass cheap units over static base defense"** — DVOID's own point that extra
  rifles defend as well as a static defense. We skip base-defense micro.
- **Rifle:Rocket ratio ≈ 5:1** (8:1 very early) for a cheap, self-sufficient
  army with anti-armor/anti-air coverage.
- **"Attack is the best defense."** The guide's thesis is ours.
- **Harvest the *nearest* ore** (DVOID: order ore trucks to the closest ore).
- **Coarse formation: tanks in front, infantry behind** when attack-moving —
  as a one-shot grouping, not continuous concave micro.
- **Scout to read intentions early.**

**Reject (too slow / wrong goal for a 20-min aggro agent):**
- Multiple expansions, 7 barracks, Con-yard walking, second-MCV expansions.
- Veterancy preservation, repair-before-death micro.
- Engineer+Thief refinery captures, oil-derrick capture phase, Grenadier drops.
- V2/Artillery hold-fire sniping, Mig indirect-route runs, Iron Curtain /
  Demo-truck stalemate breakers, concrete walls + mines.
- Elaborate concave formations, reserve flanking, air micro.

---

## 3. Architecture: hybrid (fast executor + thinking brain)

Two layers, so routine play is instant and only genuine *decisions* cost a
model call.

### 3a. Deterministic executor (the hot path — no model call)
A fast state machine that runs on **every** view and handles everything
routine, emitting orders directly:
- **Build-order progression**: deploy MCV; queue/place the next building when
  the current one is ready and we can afford it; add power when
  `power_drained` approaches `power_provided`.
- **Economy**: keep harvesters harvesting the nearest ore; (optionally) one
  extra harvester/refinery for income; never float excessive cash — convert to
  army.
- **Production**: train rifles/rockets at the Barracks and tanks at the War
  Factory at the target ratio, up to an affordability cap.
- **Scouting**: send the two scouts out once (see §5).
- **Standing attack cadence**: when the army is at/above the current attack
  threshold, issue the server's ready-made `legal_actions.attack_now`
  (optionally retargeted) at the enemy base/nearest enemy.

The executor is pure functions over the observation. It is the "actions
executed as fast as possible" half, and it is what plays if the brain is slow,
errors, or is on cooldown — so **a model failure never stalls the game** (same
hard rule as the VGC agent).

### 3b. Thinking brain (the slow path — a throttled model call)
The LLM is **not** called every view. It is called only when a **decision**
actually arises, and its output sets a small, structured **policy/mode** the
executor then executes quickly. The brain decides things like:
- **Cheese response**: scout reports enemy MCV/units inbound or an unexpectedly
  empty/forward enemy base → switch to `DEFEND_RUSH` mode (spam cheapest attack
  units, rally them home, hit the cheese as it arrives).
- **Attack vs. hold**: given relative army value / positioning, is a push
  favorable? → `ALL_IN` vs. `BUILD` vs. `HARASS`.
- **Composition shift**: infantry-heavy early → tank-heavy once the War Factory
  is online and economy supports it.
- **Endgame-by-clock**: with a trade/asset lead and the clock low, switch to
  `PROTECT_LEAD` (preserve assets, avoid bad trades, deny the enemy cheap
  kills) to lock the tiebreak.

**Throttling (keeps it fast + cheap):** the brain is invoked at most once every
N seconds (e.g. 3–5s) **and** immediately on a *trigger event* (scout sees
cheese, `under_attack` event, War Factory finishes, army crosses a size
threshold, clock enters the final window). Between calls the executor just runs
the last policy. This mirrors the example agent's `MAX_CALLS`/backoff cost
guard. A model error → keep the current mode, back off, keep executing.

### 3c. Modes (the brain's output; the executor's input)
A tiny enum the brain sets and the executor obeys:
- `OPEN` — run the build order, start economy, send scouts.
- `BUILD` — macro up the deathball (default mid-game).
- `DEFEND_RUSH` — cheese detected: mass cheapest units, rally home, kill the rush.
- `ALL_IN` — commit the deathball to an attack-move on the enemy base.
- `HARASS` — send a detachment at enemy harvesters/economy (cheap trade-ups).
- `PROTECT_LEAD` — clock low and ahead: preserve assets, avoid bad trades.

Transitions are the brain's job; execution is the executor's. This is the clean
seam that makes it both "fast" and "thinking."

---

## 4. Phased game plan (timeline)

Costs/build-times are read at runtime from `available_production` /
`game_config.rules` — **never hard-coded** (the server is authoritative). The
item ids below (`powr`, `tent`/`barr`, `proc`, `weap`, `e1`, `e3`, light/medium
tank) are confirmed against the example agent and resolved from
`available_production` each match.

**Phase 0 — Opening (0:00–~1:30):**
Deploy MCV → `fact`. Build `powr` → `barr`/`tent` → `proc` → `weap`, placing
each as it readies, next to the base. Start the harvester on the nearest ore.
Begin training rifles immediately once the Barracks is up (they are our
anti-cheese insurance).

**Phase 1 — Scout + anti-cheese read (~0:30–~3:00):**
As soon as a cheap unit exists, execute §5 scouting. Brain evaluates the scout
report; if cheese → `DEFEND_RUSH`, else `BUILD`.

**Phase 2 — Infantry bridge (~1:00–~4:00):**
Pump rifles (+rockets ~8:1 → 5:1) for a cheap early army that both defends and
can punish a passive opponent. This is where a cheese-prone or passive LLM
opponent loses if it over-commits or stalls.

**Phase 3 — Tank-centric deathball (~3:00–~15:00):**
Once `weap` is online and economy supports it, make **tanks the core** of the
deathball (rockets mixed for anti-armor/air, rifles as cheap screen). Tanks
trade *up* against infantry, which is exactly what wins the
`kills_cost - deaths_cost` tiebreak. **20 minutes is ample** — tanks roll by
~3–4 min and reach deathball numbers by ~6–8 min, leaving 12+ min to apply
pressure.

**Phase 4 — Pressure & protect the lead (~8:00–20:00):**
Attack-move the deathball at the enemy (tanks front, infantry back). Trade
favorably, retreat from clearly-losing fights (don't feed the enemy cheap
`kills_cost`). As the clock enters its final window with a lead →
`PROTECT_LEAD`: stop taking bad trades, keep assets alive, coast to the
tiebreak.

---

## 5. Scouting & cheese defense (your step 2)

**Two scouts, sent once, cheap:**
1. **Infantry at a choke point** — park one rifle on the natural approach to our
   base (map geometry from `map.spawns` / `map.bounds`), stance `defend`/`hold`,
   as a tripwire + view range. Replaced if killed (cheap).
2. **Light vehicle** (jeep/APC/light tank, whatever is cheapest/fastest from
   `available_production`) toward the **enemy spawn guess**
   (`choose_enemy_spawn`) for speed and information.

**What counts as cheese (deterministic signals the executor flags for the brain):**
- Enemy **MCV or combat units detected near our base / on our side of the map**
  early (base-move or unit rush).
- Enemy base **not where expected / looks packed up** (MCV rush / base-move).
- An early `under_attack` event before we have an army.

**Reaction (brain → `DEFEND_RUSH`):** build the **cheapest attack units as fast
as possible** (rifles, +rockets vs. vehicles), rally them **home**, and kill the
rush as it arrives (defender's advantage + our view-range tripwire). No static
defense micro. Once the cheese is dead or doesn't materialize → back to `BUILD`.
Because most RTS LLM bots are weak to cheese, this early read is also where *we*
can punish *their* over-extension — so `DEFEND_RUSH` naturally flows into a
counter-attack when the enemy army is spent.

---

## 6. Robustness (the agent must never stall or forfeit)

Mirrors the VGC agent's hard rule — a model/parse problem costs decision
quality, never a legal move:
- **Executor is the floor.** If the brain errors, times out, or is on cooldown,
  the executor keeps running the last mode. The game never waits on the model.
- **`WAIT` when truly nothing to do**, exactly as the example agent does.
- **Order hygiene** reused from the example agent: validate every order's shape
  before sending; feed `on_action_result` refusals back; never resend a batch
  refused as a whole; drop an order refused repeatedly for the same reason;
  don't over-queue buildings; re-read unit ids every view (ids change/die).
- **Idempotent build order**: progression keyed off the *observed* state
  (what's built/queued), not a script counter — so a refused/retried order or a
  killed building self-corrects.
- **Cost guard**: cap total model calls per match (~a few hundred, far below the
  example's 2400 since we call far less often), with error backoff.

---

## 7. Proposed module layout (`agent/redalert/`)

```
agent/redalert/
  __init__.py
  agent.py          # create_agent(); RedAlertAgent.choose_action / on_action_result
  executor.py       # deterministic state machine -> order batches (hot path)
  brain.py          # throttled LLM call -> Mode + targets (slow path)
  modes.py          # Mode enum + the policy each mode implies
  observe.py        # typed reads of observation: economy, army, buildings, scouts, cheese signals
  orders.py         # order builders + shape validation (adapted from examples/llm/redalert.py)
  buildplan.py      # the build order as data, progressed off observed state
  scout.py          # scout dispatch + cheese detection
  config.py         # item ids resolved from available_production; thresholds, ratios, timings
```

Reuses, not reinvents, the example agent's proven order-schema handling,
prompt-sizing, and refusal-feedback machinery.

---

## 8. Testing approach (offline, like VGC)

The sandbox has no network for deps and no live RA server, so (as with VGC)
verification is **deterministic offline tests** against synthesized
observations:
- Build-order progression picks the right next building given a state.
- Economy keeps harvesters on nearest ore; adds power when drained.
- Cheese detector fires on MCV-near-base / early-under-attack and *not* on
  normal states (no false positives — a false cheese alarm would wreck tempo).
- Mode→orders mapping (e.g. `DEFEND_RUSH` trains cheapest units + rallies home;
  `ALL_IN` emits an attack-move at the enemy base).
- Robustness: brain error/timeout → executor still returns a legal batch;
  nothing-to-do → `WAIT`; refused orders aren't resent.
- Order shape validation (adapted from the example agent's tests).

A live match remains the real test (unverifiable here) — called out explicitly,
as with VGC.

---

## 9. Open questions / assumptions to confirm

1. **Tank timing (answered):** infantry-first, tanks the deathball core by
   mid-game; 20 min is ample. Assumed OK unless you want infantry-heavier for
   even more cost-efficiency vs. weak bots.
2. **One base only** (no expansions) — assumed, per "simple & robust." One extra
   refinery for income is the most macro we'll do.
3. **Brain cadence** (every ~3–5s + on triggers) — a starting value, tunable.
4. **Faction-agnostic**: build order uses faction-resolved ids
   (`tent` vs `barr`, Allied vs Soviet tank ids) read from
   `available_production`, so it plays either faction the match assigns.
```
