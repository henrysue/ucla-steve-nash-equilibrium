"""Item ids, build order, ratios, and thresholds — tuned for a fast aggro game.

Two rules keep this robust:

- **The server is authoritative about costs and availability.** Nothing here
  hard-codes a credit cost or build time; those are read at runtime from the
  observation's ``available_production`` and ``game_config.rules``. The ids
  below are only *candidates*: we use whichever a given faction/match actually
  offers (``resolve_*`` picks the first available), so the agent plays either
  faction without change.
- **Thresholds are starting values, deliberately conservative on the one axis
  that can lose us tempo (false cheese alarms) and escalating on the one that
  can lose us the game (never attacking).**
"""

from __future__ import annotations

# -- building ids (OpenRA Red Alert). Candidates, resolved against what the
#    match actually offers; faction variants listed together. ------------------
MCV = "mcv"
CONSTRUCTION_YARD = "fact"
POWER_PLANT = "powr"
ADV_POWER_PLANT = "apwr"
BARRACKS = ("tent", "barr")       # Allied tent / Soviet barracks
REFINERY = "proc"
WAR_FACTORY = "weap"

# -- unit ids ----------------------------------------------------------------
RIFLE = "e1"                       # cheap infantry, anti-infantry + filler/screen
ROCKET = "e3"                      # anti-armor / anti-air
# Light/medium tanks differ by faction; resolve from available_production.
LIGHT_TANKS = ("1tnk", "2tnk")     # Allied light / medium
HEAVY_TANKS = ("3tnk", "4tnk")     # Soviet heavy / mammoth
SCOUT_VEHICLES = ("jeep", "apc", "1tnk", "2tnk")  # fast/cheap, in preference order
HARVESTER = ("harv",)

# -- the opening build order (one base, no expansions). Progressed off the
#    OBSERVED state, not a counter, so a refused/retried step self-corrects.
#    Power is NOT in this list: it is a reactive interrupt (see executor). -----
BUILD_ORDER = (
    POWER_PLANT,
    BARRACKS,
    REFINERY,
    WAR_FACTORY,
    REFINERY,          # one extra refinery for income; the only "macro" we do
)

# -- army composition --------------------------------------------------------
RIFLE_TO_ROCKET_EARLY = 8          # 8:1 while poor
RIFLE_TO_ROCKET = 5                # 5:1 once we have credits
ROCKET_RATIO_CREDIT_SWITCH = 2500  # cash above which we move to the richer ratio

# -- power safety: keep this much headroom before queuing non-power ----------
POWER_MARGIN = 25                  # provided - drained must stay above this

# -- cheese detection (conservative: a false alarm wrecks our own tempo) ------
# Flip to DEFEND_RUSH only when enemy COMBAT value seen near our base crosses
# this, not on a lone scout. Distance is in cells from our base_center.
CHEESE_NEAR_BASE_CELLS = 18
CHEESE_MIN_ENEMY_UNITS = 2         # at least two threatening units near base
CHEESE_EXPIRE_SECONDS = 25.0       # drop back to BUILD if the threat fizzles

# -- attack decision ---------------------------------------------------------
# Commit the deathball only with a real edge; HARASS/BUILD otherwise.
ATTACK_ARMY_EDGE = 1.3             # our army value >= 1.3x the enemy value we can see
MIN_ATTACK_ARMY_VALUE = 1500       # don't "all-in" with a token force
# Anti-passivity escalation: past this game time, lower the bar so we DO apply
# pressure (being ahead at the tiebreak requires acting).
ESCALATE_AFTER_SECONDS = 420.0     # 7:00
ESCALATED_ARMY_EDGE = 1.0

# -- protect-the-lead window: near the limit and ahead, stop bad trades ------
PROTECT_LEAD_REMAINING_SECONDS = 180.0   # final 3 minutes

# -- brain throttle (off the hot path; the executor never waits on it) -------
BRAIN_MIN_INTERVAL_SECONDS = 4.0
MAX_BRAIN_CALLS = 300              # cost guard for a 20-minute match

# -- economy -----------------------------------------------------------------
TARGET_HARVESTERS = 2              # the refinery's own + one extra; no expansions


def resolve(candidates, available: set[str]) -> str | None:
    """First candidate id that the match actually offers (faction-agnostic)."""
    for item in candidates:
        if item in available:
            return item
    return None
