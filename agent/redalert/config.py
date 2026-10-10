"""Every tunable number and id for the Red Alert agent, grouped by topic.

The server is authoritative: costs and availability are read at runtime
(``available_production``, ``game_config.rules``); the ids here are candidates
resolved against what the match offers (:func:`resolve`), so either faction
works. Rules reference: https://api.altruagent-game.com/skill/redalert.
Why a value is what it is lives in PLAN.md ("Lessons from matches").
"""

from __future__ import annotations

# -- ids (faction variants together) ----------------------------------------------------
MCV = "mcv"
CONSTRUCTION_YARD = "fact"
POWER_PLANT = "powr"
ADV_POWER_PLANT = "apwr"
BARRACKS = ("tent", "barr")        # Allied / Soviet
REFINERY = "proc"
SILO = "silo"
WAR_FACTORY = "weap"
SERVICE_DEPOT = "fix"              # unlocks 2tnk/3tnk and the MCV
RADAR_DOME = "dome"                # unlocks siege units
RIFLE = "e1"
ROCKET = "e3"
LIGHT_TANKS = ("1tnk", "2tnk")     # Allied light / medium
HEAVY_TANKS = ("3tnk", "4tnk")     # Soviet heavy / mammoth
HARVESTER = ("harv",)
SIEGE_UNITS = ("v2rl", "arty")     # Soviet V2 / Allied artillery: outrange defenses
SCOUT_VEHICLES = ("jeep", "ftrk")
SCOUT_FALLBACK = "e1"              # a rifleman scouts until a vehicle exists
NON_COMBAT = ("e6", "medi", "mech", "spy", "thf", "lst")
VEHICLE_TYPES = ("1tnk", "2tnk", "3tnk", "4tnk", "jeep", "apc", "ftrk", "arty", "v2rl", "ttnk", "ctnk", "stnk")
DEFENSES = ("pbox", "ftur", "gun", "tsla")   # cheapest first
ALL_BUILDING_TYPES = frozenset({
    "fact", "powr", "apwr", "tent", "barr", "proc", "weap", "fix", "silo", "dome", "hpad", "afld",
    "spen", "syrd", "atek", "stek", "gap", "sam", "agun", "kenn", "iron", "pdox", "mslo",
    "pbox", "hbox", "gun", "ftur", "tsla", "fake", "brik", "sbag", "fenc",
})
DEFAULT_UNIT_COST = 300            # when the rules give no cost
KNOWN_COSTS = {                    # the skill guide's prices, when the rules give none
    "powr": 300, "tent": 500, "barr": 500, "proc": 1400, "weap": 2000, "fix": 1200, "dome": 1000,
    "harv": 1100, "mcv": 2500, "kenn": 200, "e1": 100, "e3": 300, "dog": 200,
}

# -- opening: the double-refinery build order (a repeated id = one more of it) -----------
BUILD_ORDER = (
    POWER_PLANT, BARRACKS, REFINERY, REFINERY,   # economy + infantry
    POWER_PLANT, WAR_FACTORY, REFINERY,          # power + war factory
    SERVICE_DEPOT,                               # tech (and the expansion MCV)
)
POWER_MARGIN = 25                  # build power when provided - drained falls below this
SILO_FILL_RATIO = 0.8              # build a silo when stored ore passes this share of capacity

# -- economy: harvesters, refineries, money bands -------------------------------------------
EXTRA_HARVESTERS = 2               # harvester target = refineries + this (refineries bring their own)
MAX_HARVESTERS_PER_REFINERY = 3
MAX_HARVESTERS = 12
MAX_REFINERIES = 8
MAX_REFINERIES_PER_FIELD = 3       # more on one field just queue for its ore (b7fcbd4e: 5 on one field)
REFINERY_SCHEDULE = ((300.0, 4), (390.0, 5), (480.0, 6), (570.0, 7), (660.0, 8))  # (game s, count)
REFINERY_ADD_FUNDS = 800
ECONOMY_ITEMS = ("powr", "apwr", "proc", "weap", "harv", "mcv")  # army training pauses while these build...
MIN_DEFENDERS = 6                  # ...if at least this many fighters remain
# Money = cash + stored ore. Broke: one unit queue, one deep. Rich / very rich: spend harder.
FUNDS_LOW = 600
FUNDS_RICH = 2500
FUNDS_VERY_RICH = 6000
FUNDS_VERY_RICH_DEPTH = 2          # queue depth when very rich (never more than 2: money finishes units)
MAX_QUEUE_DEPTH = 2                # units queued per type at most, trained one at a time

# -- production: infantry-heavy army from many barracks ---------------------------------------
RIFLE_TO_ROCKET_EARLY = 4          # riflemen per rocket soldier while poor
RIFLE_TO_ROCKET = 3                # ...and once money passes the switch
ROCKET_RATIO_CREDIT_SWITCH = 2500
BARRACKS_SCHEDULE = ((240.0, 3), (270.0, 4), (330.0, 5))  # after the service depot
MAX_BARRACKS = 5
BARRACKS_ADD_FUNDS = 600
MAX_WAR_FACTORIES = 1              # tanks are support
MAX_WAR_FACTORIES_VERY_RICH = 2

# -- end game: radar dome and siege units against turtled bases ---------------------------------
DOME_AFTER_SECONDS = 420.0         # pros build the dome ~7:00-8:30
DOME_ADD_FUNDS = 1000
ENEMY_DEFENSE_TYPES = ("pbox", "hbox", "gun", "ftur", "tsla", "agun", "sam")
SIEGE_MIN_ENEMY_DEFENSES = 1       # siege units only once we know of enemy defenses
SIEGE_MAX = 4
SIEGE_TARGET_RADIUS_CELLS = 10     # siege units fire on known defenses this close first
TURTLE_DEFENSES = 2                # this many known enemy defenses: lean on tanks over infantry
GROUND_DEFENSE_TYPES = ("pbox", "hbox", "gun", "ftur", "tsla")   # turrets that shoot ground units
DEFENSE_CLUSTER_CELLS = 12         # turrets this close to the attack target defend it
INFANTRY_PER_DEFENSE = 6           # attack defended ground with this many infantry per turret...
INFANTRY_MASS_MIN = 15             # ...and at least this many...
ARMOR_PER_DEFENSE = 3              # ...or this many tanks per turret (siege units count double)

# -- building placement ------------------------------------------------------------------------
REFINERY_ORE_SEARCH_CELLS = 16     # ore this close steers refineries (and keeps power off its side)
ORE_SIDE_AVOIDANCE = 0.5           # how hard the factory/depot bearing turns off the ore side
BUILDING_GAP_CELLS = 1             # free cells between buildings (OpenRA allows little more)
LANE_WIDTH_CELLS = 3               # kept-clear road from the base toward the staging point...
LANE_KEEP_CLEAR_CELLS = 14         # ...for this many cells
EXIT_CLEAR_ROWS = 2                # rows kept clear in front of barracks / war factories
# (width, height), top-left anchored; the rules give no sizes. Power plants,
# barracks and the dome carry a bib row below (2x3): placements touching it
# fail (placement audit, match fd507a58).
FOOTPRINTS = {
    "fact": (3, 3), "powr": (2, 3), "apwr": (3, 3), "tent": (2, 3), "barr": (2, 3),
    "proc": (3, 3), "weap": (3, 3), "fix": (3, 3), "silo": (1, 1), "dome": (2, 3), "kenn": (1, 1),
    "pbox": (1, 1), "gun": (1, 1), "ftur": (1, 1), "tsla": (1, 2),
}
DEFAULT_FOOTPRINT = (2, 2)

# -- defenses and repairs ----------------------------------------------------------------------
STANDING_DEFENSES = 1              # once the war factory is up
MAX_DEFENSES = 3                   # only while under attack...
DEFENSE_ADD_FUNDS = 600            # ...and each beyond the first only with this much money
DEFENSE_FAN_DEGREES = (0, -25, 25, -50, 50)  # turret slots, relative to the enemy's direction
DEFENSE_MAIN_RADIUS_CELLS = 16     # defenses this close to the base count toward its caps
DEFENSE_MIN_GAP_CELLS = 3          # a slot is taken if a defense is this close
DEFENSE_SIDE_MAX_DEGREES = 60.0    # the advisor's defense side must face the enemy within this
REPAIR_TYPES = ("fact", "powr", "apwr", "proc", "weap")
REPAIR_BELOW_HP = 50

# -- threats at the base ---------------------------------------------------------------------
CHEESE_NEAR_BASE_CELLS = 18        # enemies this close to any of our yards are a threat
CHEESE_MIN_ENEMY_UNITS = 2
CHEESE_EXPIRE_SECONDS = 25.0       # after the threat was last seen
CHEESE_WINDOW_SECONDS = 180.0      # a threat before this is a rush (rifle spam); later, DEFEND_BASE
BASE_DEFEND_MIN_VALUE = 1500       # later, the army comes home only for a threat worth this...
BASE_DEFEND_HOME_RATIO = 1.0       # ...and more than this x what already defends the base (units at home, turrets)
BASE_DEFEND_EXPIRE_SECONDS = 10.0  # ...and goes back out this soon after the base is clear
UNDER_ATTACK_FRESH_TICKS = 250     # an under_attack event counts for 10 s

# -- attacking -----------------------------------------------------------------------------------
ATTACK_ARMY_EDGE = 1.3             # our army / known enemy army needed to attack...
ESCALATED_ARMY_EDGE = 1.0          # ...lowered after ESCALATE_AFTER_SECONDS
ESCALATE_AFTER_SECONDS = 420.0
MIN_ATTACK_ARMY_VALUE = 1500       # never attack with less
EARLY_ATTACK_SECONDS = 180.0       # before this (and a war factory), attack only when overwhelming
MODE_MIN_DWELL_SECONDS = 25.0      # no BUILD <-> ALL_IN flip faster than this
MODEL_HOLD_MAX_EDGE = 2.5          # the model's "hold" can't stop an attack at this edge or more
TARGET_PRIORITY = (("powr", "apwr"), ("fact", "weap", "tent", "barr"), ("proc",))  # first to last
ATTACK_FOCUS_RADIUS_CELLS = 10     # priority applies among buildings this close to the nearest
STAGE_FORWARD_CELLS = 8            # mass (and rally new units) this far in front of base+ore, toward the enemy...
STAGE_FALLBACK_CELLS = 3           # ...or this far if enemies are near that spot...
STAGE_FRACTION = 0.30              # ...never farther than this share of the way to the enemy
STAGE_MAX_FRACTION = 0.40          # the advisor's stage point must be on our side of this
STAGE_DANGER_RADIUS_CELLS = 10
STAGING_RADIUS_CELLS = 4           # idle units already this close to the stage are left alone
STAGE_CLEAR_CELLS = 3              # no building this close to the massing point: the army stands there...
STAGE_MOVE_SEARCH_CELLS = 6        # ...and if one is (or the ground is blocked), the point moves this far at most
PLACE_CLEAR_WAIT_SECONDS = 5.0     # our units on a building's spot are moved off; placement waits this long at most
FLANK_OFFSET_FRACTION = 0.30       # flank waypoint: halfway, offset sideways by this share
FLANK_DANGER_RADIUS_CELLS = 12     # pick the side with fewest known enemies this close
FLANK_ARRIVED_CELLS = 6
FLANK_MAX_SECONDS = 60.0
FLANK_AFTER_SECONDS = 300.0        # no flanking before 5:00: hold the middle, keep map control
HUNT_CHECKED_CELLS = 8             # a search point is checked once a unit gets this close
FINISH_MAX_BUILDINGS = 3           # this few enemy buildings known (and we lead): finish them, no holding back
FINISH_KILL_RATIO = 2.0            # ...or kills at least this x losses...
FINISH_MIN_LEAD = 5000             # ...and kills - losses at least this: dominant, finish them
ADVICE_TARGET_SUPPORT_CELLS = 10   # the advisor's army target counts only with a known enemy this close to it
SELL_FOR_INCOME_SECONDS = 20.0     # broke with no harvester: sell a spare building this often
HUNT_GROUPS = 3                    # the hunt sweeps this many search points at once...
HUNT_GROUP_MIN_UNITS = 6           # ...each group at least this many units
SPLIT_MIN_UNITS = 8                # overwhelming: split the army between the 2 nearest enemy clusters
ENEMY_CLUSTER_CELLS = 10           # enemy buildings this close together are one cluster

# -- retreating ------------------------------------------------------------------------------------
RETREAT_ARMY_EDGE = 0.8            # retreat below this edge while attacking...
TRADE_WINDOW_SECONDS = 30.0        # ...or when, since the attack began (at most this far back),
TRADE_LOSS_MIN = 1000              # we lost at least this,
TRADE_LOSS_ARMY_SHARE = 0.15       # and this share of the attacking army,
TRADE_LOSS_RATIO = 2.0             # and more than this times what we killed,
MIN_ENGAGEMENT_SECONDS = 15.0      # after at least this long in the fight
REGROUPED_SHARE = 0.7              # retreat ends when this share is back...
REGROUP_RADIUS_CELLS = 8           # ...within this of the staging point
RETREAT_COOLDOWN_SECONDS = 60.0    # no new attack this soon after a retreat
REATTACK_ARMY_GROWTH = 1.3         # ...nor before the army regrows this much...
RETREAT_REGROW_MAX_SECONDS = 120.0 # ...unless this long has passed
RETURN_FIRE_RADIUS_CELLS = 4       # retreating units this close to an enemy fight back

# -- end of the match (tiebreak: kills - deaths, then assets) -----------------------------------
PROTECT_LEAD_REMAINING_SECONDS = 120.0  # final 2 minutes with a lead of PROTECT_LEAD_MIN_MARGIN...
PROTECT_LEAD_MIN_MARGIN = 1500
PROTECT_LEAD_LAST_SECONDS = 60.0        # ...or the final minute with any lead
CATCH_UP_REMAINING_SECONDS = 180.0      # behind or level this late: attack regardless

# -- squads: scout, raids, home guard, garrison ---------------------------------------------------
SCOUT_INSET_CELLS = 8              # patrol ring this far inside the map edge
SCOUT_MIN_LEG_SECONDS = 2.0
EARLY_SCOUT_SECONDS = 300.0        # before 5:00 the scout shuttles: their base <-> our approach
EARLY_SCOUT_STANDOFF_CELLS = 12    # watch the ground in front of their base (where a push masses) from this far out...
EARLY_SCOUT_SIDE_CELLS = 4         # ...and this far to the flank's side of their approach
SCOUT_FLANK_FRACTION = 0.30        # the scout's path bows out sideways this share of the base-to-base distance
SCOUT_REPLACE_SECONDS = 90.0       # a dead rifle scout is replaced only after this (vehicles at once)
SQUAD_CLAIM_RADIUS_CELLS = 14      # guards and garrisons take only units this close to home, never from the field
EARLY_PUSH_WINDOW_SECONDS = 360.0  # an early push: before 6:00, known enemy army of at least...
EARLY_PUSH_MIN_VALUE = 1200        # ...this value
EARLY_PUSH_RATIO = 0.8             # ...and this share of ours: hold at home, no attacks, turrets up
ENEMY_ORE_RADIUS_CELLS = 20        # their ore field: ore this close to their spawn
ENEMY_EXPANSION_WATCH_AFTER_SECONDS = 210.0
ENEMY_EXPANSION_SCOUT_SITES = 2
RAID_START_SECONDS = 150.0
RAID_SQUAD_SIZE = 4
RAID_MIN_SIZE = 3
RAID_UNIT_TYPES = ("e3", "e1", "jeep", "ftrk", "1tnk")   # rockets first: harvesters are armored
RAID_COOLDOWN_SECONDS = 60.0
RAID_EXPANSION_MAX_DEFENSES = 1    # raids skip targets with more defenses than this
RAID_THREAT_RADIUS_CELLS = 7       # enemy fighters this close to the raid squad are defending: fight them first...
RAID_OUTMATCH_RATIO = 1.0          # ...unless worth more than this x the squad: then it pulls back
DISTRACTION_SECONDS = 30.0         # the raid has enemy fighters on it (or just pulled back): they are distracted...
DISTRACTION_MIN_VALUE = 600        # ...when worth at least this
MAIN_FIGHT_RADIUS_CELLS = 9        # enemies this close to the main army: it is in a fight (the raid joins as a second prong)
SCOUT_FLEE_CELLS = 6               # the scout backs off from enemy fighters this close
SCOUT_ALARM_SECONDS = 30.0         # a scout killed far forward early is an early-push alarm this long
PRODUCTION_FORWARD_BIAS = 0.4      # barracks/factories grow sideways, this much toward the enemy
# -- scouting and intel (scouting.py, intel.py) --
SCOUT_UNIT_TYPES = ("dog", "e1")  # scouts: dogs (Soviet kennel) where we have them, else riflemen
KENNEL = "kenn"                   # one kennel after the war factory, for dog scouts (Soviet only)
EARLY_SCOUT_END_SECONDS = 180.0    # two scouts (left, right) round the back of their base in the first 3 minutes
MID_PICKET_SECONDS = 300.0         # then picket posts across the approach...
MID_PICKETS = 4                    # ...this many...
PICKET_ALONG = 0.45                # ...this far toward the enemy...
PICKET_SPREAD = 0.35               # ...spread this share of the base-to-base distance to each side
BASE_CHECK_SECONDS = 150.0         # a picket runs round their base again this often
CIRCUIT_RADIUS_CELLS = 11          # the round of their base: this far from their spawn...
CIRCUIT_DEGREES = (80, 125, 170)   # ...at these angles from the side facing us, round to the back
CIRCUIT_ARRIVED_CELLS = 3
SCOUT_SNAP_CELLS = 6               # a post or waypoint walled in by terrain moves to a reachable cell this close
SCOUT_WALK_STEPS = 250             # (walking steps searched from our base)
INTEL_MIN_BUILDINGS = 3            # their opening is read once we know this many of their buildings
RUSH_READ_SECONDS = 100.0          # one refinery but a war factory / 2nd barracks by now: a rush
# -- committing to an attack --
CONFIRM_SECONDS = 4.0              # an attack call is re-checked this long later, the army edging forward to see...
CONFIRM_TRADE_BAND = 200           # ...only while kills - losses are this close to even (clearly ahead/behind: go at once)
PROBE_FRACTION = 0.3               # ...to this share of the way from the massing point to the target
ATTACK_GROUPS = 3                  # the army attacks in this many groups, a semi-concave...
ATTACK_GROUP_SPREAD_CELLS = 5      # ...the side groups this far either side of the target
ATTACK_GROUP_MIN_UNITS = 6         # ...once the army has this many units
GARRISON_SIZE = 4
GARRISON_TYPES = ("3tnk", "2tnk", "1tnk", "e3", "ftrk", "jeep", "e1")   # tanks first
GARRISON_LEASH_CELLS = 12          # a garrison stays (and is led) within this of its expansion

# -- expansion -----------------------------------------------------------------------------------
EXPAND_AFTER_SECONDS = 225.0       # earliest start ("now" from the advisor: EXPAND_EARLIEST_SECONDS)
EXPAND_EARLIEST_SECONDS = 180.0
EXPAND_LAST_START_SECONDS = 840.0  # none after this: it can't pay off
EXPAND_MIN_HARVESTERS = 3
EXPAND_MIN_REFINERIES = 2
EXPAND_MIN_ARMY_VALUE = 0
EXPAND_MAX_EXPANSIONS = 3
EXPAND_MAX_FAILURES = 2
EXPAND_AGAIN_AFTER_SECONDS = 150.0          # next expansion this long after the last...
EXPAND_DEPLETED_IDLE_HARVESTERS = 2         # ...or sooner when this many harvesters idle
EXPAND_DEPLETED_IDLE_SECONDS = 30.0         # for this long
EXPAND_STRAPPED_FUNDS = 600                 # ...or sooner when money stays under this...
EXPAND_STRAPPED_SECONDS = 45.0              # ...for this long...
EXPAND_EVEN_TRADES = 0.9                    # ...while trades are about even or better (kills >= this x losses)...
EXPAND_AGAIN_STRAPPED_SECONDS = 60.0        # ...this long after the last
EXPAND_MIN_DISTANCE_CELLS = 10     # a different ore field from the main one (the natural is ~16 away)
ENEMY_EXPANSION_MIN_CELLS = 18     # enemy buildings/fields this far from their main base = an expansion
EXPAND_MIN_FIELD_CELLS = 2         # an expansion field has at least this many ore cells...
EXPAND_MIN_FIELD_SHARE = 0.25      # ...and this share of the biggest field's (not a straggler)
EXPAND_SEARCH_WALK_STEPS = 90      # fields ranked by walking steps from our base, this far
EXPAND_CLUSTER_CELLS = 8           # ore grouped on this grid into fields
EXPAND_DEPLOY_OFFSET_CELLS = 4     # deploy this far short of the ore
EXPAND_DANGER_RADIUS_CELLS = 12    # no known enemy this close to the site
EXPAND_SEEN_RADIUS_CELLS = 8       # a unit this close has seen the site
EXPAND_SCOUT_VALID_SECONDS = 90.0
EXPAND_SCOUT_TIMEOUT_SECONDS = 120.0
EXPAND_TRAIN_TIMEOUT_SECONDS = 150.0
EXPAND_DEPLOY_RADIUS_CELLS = 2
EXPAND_DEPLOY_RETRY_SECONDS = 3.0
EXPAND_MOVE_TIMEOUT_SECONDS = 90.0 # then deploy where it stands...
EXPAND_GIVE_UP_SECONDS = 30.0      # ...then abandon it
EXPAND_STUCK_RADIUS_CELLS = 8      # stopped this close after EXPAND_STUCK_AFTER_SECONDS = arrived
EXPAND_STUCK_AFTER_SECONDS = 10.0
EXPAND_ROOM_SEARCH_CELLS = 5       # search radius for a 3x3 spot to unfold
EXPAND_REFINERY_RADIUS_CELLS = 10  # a refinery this close to the new ore serves it
EXPAND_REFINERY_RETRY_SECONDS = 90.0
EXPAND_COOLDOWN_SECONDS = 60.0
EXPANSION_DEFENSES = 1
ACTIVE_BASE_DEFENSES = 3           # turrets at the expansion that takes over once home is mined out
REFINERY_ORE_WALK_STEPS = 6        # a refinery must be this few walking steps from its ore...
EXPAND_SITE_WALK_STEPS = 8         # ...an expansion deploy spot from its field's ore...
FIELD_WALK_STEPS = 14              # ...and a refinery's trucks work ore this few steps away
REFINERY_MOVED_CELLS = 2           # an expansion refinery standing farther than this from its spot was moved
REFINERY_PLACE_WAIT_SECONDS = 15.0 # ...judged on the first refinery to appear this soon after the order
EXPAND_REFINERY_MAX_MOVES = 2      # then stop buying refineries for that expansion (each one lands at home)
FIELD_NEAR_CELLS = 10              # a harvester this close to a field's ore is working it
FIELD_IDLE_SECONDS = 5.0           # idle this long there: the field is depleted...
FIELD_RECHECK_SECONDS = 120.0      # ...until re-checked (ore mines regrow)
FIELD_WORK_CELLS = 3               # a busy harvester this close to a field's ore is mining it
FIELD_STALE_SECONDS = 45.0         # harvesters around a field, none on its ore this long: depleted
REBALANCE_PER_VIEW = 2             # harvesters moved off a depleted field per view
HARVESTER_SETTLE_SECONDS = 30.0    # a moved harvester stays put at least this long

# -- units: micro, movement, stances --------------------------------------------------------------
FOCUS_RADIUS_CELLS = 6             # focus fire on the weakest enemy this close
PULL_BACK_HP = 35                  # vehicles below this (%) leave attacks
STUCK_SECONDS = 10.0               # an ordered unit that hasn't moved this long gets a stop
ENGAGED_RADIUS_CELLS = 7           # ...unless an enemy is this close
FORMATION_SPACING_CELLS = 1.5      # concave slots
FORMATION_CURVE = 0.15
FORMATION_MAX_UNITS = 12
STANCE_HOLD = "defend"             # waiting and guarding units
STANCE_PUSH = "attack_anything"    # attacking and defending the base
RALLY_TOLERANCE_CELLS = 2

# -- enemy memory ----------------------------------------------------------------------------------
ENEMY_UNIT_MEMORY_SECONDS = 90.0   # forget an unseen enemy unit after this
MEMORY_CLEAR_RADIUS_CELLS = 4      # or when we stand on its last cell and it isn't there
ATTACK_STALL_SECONDS = 40.0        # attacking with no kills and nothing in sight this long: a stale target
TRANSFORM_RADIUS_CELLS = 4         # an enemy MCV this close to a remembered yard: the yard packed up
HOME_RAID_RADIUS_CELLS = 10        # enemies this close to a yard or harvester of ours are raiding us...
SMALL_RAID_VALUE = 1500            # ...and if worth at most this, units at home deal with them
RESPONSE_VALUE_RATIO = 2.0         # send at least this x the raiders' value...
RESPONSE_MIN_UNITS = 2             # ...and at least this many units
SECOND_FACTORY_SECONDS = 360.0     # a second war factory after this once FUNDS_RICH is banked
STALL_FORGET_CELLS = 10            # ...forget remembered buildings this close to the army...
STALL_IGNORE_ADVICE_SECONDS = 60.0 # ...and ignore the advisor's army target this long (the hunter leads)

# -- models (off the hot path; never waited on) -------------------------------------------------------
ADVISOR_INTERVAL_SECONDS = 6.0
ADVISOR_TIMEOUT_SECONDS = 10.0
ADVISOR_MAX_CALLS = 250
ADVICE_TTL_SECONDS = 20.0          # game seconds
SQUAD_LEAD_INTERVAL_SECONDS = 4.0
SQUAD_LEAD_TIMEOUT_SECONDS = 6.0
SQUAD_LEAD_MAX_CALLS = 300         # per squad
SQUAD_LEAD_TTL_SECONDS = 8.0       # game seconds
SQUAD_LEAD_SIGHT_CELLS = 12

# -- order hygiene ---------------------------------------------------------------------------------------
ARMY_REISSUE_SECONDS = 3.0         # an unchanged army order isn't resent sooner
REFUSAL_BLOCK_SECONDS = 4.0        # a refused order waits this long before a retry
PENDING_PRODUCTION_SECONDS = 1.5   # a build/train isn't resent before the state shows it


def resolve(candidates, available: set[str]) -> str | None:
    """First candidate id that the match actually offers (faction-agnostic)."""
    for item in candidates:
        if item in available:
            return item
    return None
