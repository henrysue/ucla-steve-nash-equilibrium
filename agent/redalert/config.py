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
# The strongest main battle tank offered, never the light tank when a better
# one is unlocked (52a77fb1, England: light tanks all game against medium
# tanks). Pro replays: heavy/medium tanks in every game, light tanks never.
TANK_PREFERENCE = ("3tnk", "2tnk", "4tnk", "1tnk")
MAIN_BATTLE_TANKS = ("3tnk", "2tnk", "4tnk")
TANK_VALUE_SHARE = 0.5             # pros keep 45-55% of combat value in tanks: below this, tanks come first
ENEMY_MIX_SECONDS = 180.0          # their army's make-up: the unit types our scouts saw this recently
HARVESTER = ("harv",)
SIEGE_UNITS = ("v2rl", "arty")     # Soviet V2 / Allied artillery: outrange defenses
SCOUT_VEHICLES = ("jeep", "ftrk")
NON_COMBAT = ("e6", "medi", "mech", "spy", "thf", "lst")
VEHICLE_TYPES = ("1tnk", "2tnk", "3tnk", "4tnk", "jeep", "apc", "ftrk", "arty", "v2rl", "ttnk", "ctnk", "stnk")
TURRETS = ("pbox", "hbox", "gun", "ftur", "tsla")   # ground turrets, either side: they fight in battles (strength.py),
                                   # guard attack targets, and siege units shell them (anti-air does none of it)
BUILD_TURRETS = ("pbox", "ftur", "gun", "tsla")   # what we build: the first one offered (cheapest first)
ALL_BUILDING_TYPES = frozenset({
    "fact", "powr", "apwr", "tent", "barr", "proc", "weap", "fix", "silo", "dome", "hpad", "afld",
    "spen", "syrd", "atek", "stek", "gap", "sam", "agun", "kenn", "iron", "pdox", "mslo",
    "pbox", "hbox", "gun", "ftur", "tsla", "fake", "brik", "sbag", "fenc",
})
DEFAULT_UNIT_COST = 300            # when the rules give no cost
KNOWN_COSTS = {                    # the skill guide's prices, when the rules give none
    "powr": 300, "tent": 500, "barr": 500, "proc": 1400, "weap": 2000, "fix": 1200, "dome": 1000,
    "harv": 1100, "mcv": 2500, "kenn": 200, "e1": 100, "e3": 300, "dog": 200,
    # units (OpenRA mods/ra/rules): the strength model and fights price them too
    "e2": 150, "e4": 300, "1tnk": 700, "2tnk": 850, "3tnk": 1150, "4tnk": 2000, "jeep": 500, "apc": 850,
    "ftrk": 600, "arty": 850, "v2rl": 900,
    "pbox": 600, "hbox": 750, "gun": 800, "ftur": 600, "tsla": 1200, "agun": 800, "sam": 700,
}

# -- opening: the pros' double refinery into the war factory (a repeated id = one more of it) --
# The replays: refineries at 1:00 / 1:38, war factory 2:40. The advisor waits
# until it stands (unless a rush is read).
BUILD_ORDER = (
    POWER_PLANT, BARRACKS, REFINERY, REFINERY,   # economy + infantry
    POWER_PLANT, WAR_FACTORY,                    # power + war factory
)
# Then, unless outgunned (units first): the third refinery and the service
# depot (tanks and the expansion MCV).
FOLLOW_UP = (SERVICE_DEPOT, REFINERY)   # the depot first: it unlocks the main battle tanks (and the MCV)
WAR_FACTORY_REBUILD_AFTER_SECONDS = 180.0   # past the opening, a lost (or missing) war factory is rebuilt first...
OPTIONAL_BUILDINGS = ("dome", "fix", "kenn", "silo")   # ...cancelling any of these ahead of it in the Building queue
POWER_MARGIN = 25                  # build power when provided - drained falls below this
SILO_FILL_RATIO = 0.8              # build a silo when stored ore passes this share of capacity

# -- economy: harvesters, refineries, money bands -------------------------------------------
EXTRA_HARVESTERS = 2               # harvester target = refineries + this (refineries bring their own)
MAX_HARVESTERS_PER_REFINERY = 3
MAX_HARVESTERS = 12
MAX_REFINERIES = 8
MAX_REFINERIES_PER_FIELD = 3       # more on one field just queue for its ore (b7fcbd4e: 5 on one field)
ORE_CELLS_PER_REFINERY = 15        # a field supports one refinery per this many ore cells (1..MAX per field)
FAILED_SPOT_SECONDS = 180.0        # a spot the engine moved a building from is skipped this long
BUILD_RADIUS_CELLS = 13            # a building's centre within this of a construction yard's (the lobby's build radius)
REFINERY_SCHEDULE = ((300.0, 4), (390.0, 5), (480.0, 6), (570.0, 7), (660.0, 8))  # (game s, count)
REFINERY_ADD_FUNDS = 800
MIN_DEFENDERS = 6                  # fighters always left with the army (raids, pickets)
BEHIND_ARMY_RATIO = 1.3            # their estimated army this many times ours (or the advisor's "more_army")...
BEHIND_AFTER_SECONDS = 180.0       # ...past the opening: units first — no money held for an MCV, refineries or tech
PRODUCTION_ITEMS = ("powr", "apwr", "tent", "barr", "weap")   # ...but power and production buildings keep theirs
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
BARRACKS_SCHEDULE = ((240.0, 2), (330.0, 3))  # after the service depot (each extra one only speeds the one infantry queue)
MAX_BARRACKS = 3
BARRACKS_ADD_FUNDS = 600
MAX_WAR_FACTORIES = 3              # the second once the service depot stands, the third only while money floats
FLOAT_FUNDS = 1000                 # money beyond what the build order needs (buildplan.build_reserve)...
FLOAT_SECONDS = 30.0               # ...held this long: the queues can't spend the income, more production

# -- end game: radar dome and siege units against turtled bases ---------------------------------
DOME_AFTER_SECONDS = 420.0         # pros build the dome ~7:00-8:30
DOME_ADD_FUNDS = 1000
SIEGE_MIN_ENEMY_DEFENSES = 1       # siege units only once we know of enemy defenses
SIEGE_MAX = 4
SIEGE_TARGET_RADIUS_CELLS = 10     # siege units fire on known defenses this close first
TURTLE_DEFENSES = 2                # this many known enemy defenses: lean on tanks over infantry
# The turtle read (intel.EnemyIntel.turtling): this many ground turrets known,
# past TURTLE_READ_SECONDS, and no push of theirs near our base for
# TURTLE_QUIET_SECONDS (a rusher builds pillboxes too, but keeps coming).
TURTLE_READ_DEFENSES = 4
TURTLE_READ_SECONDS = 150.0
TURTLE_QUIET_SECONDS = 300.0
# A passive enemy: past PASSIVE_READ_SECONDS, no push at our main base and no
# building of ours lost for PASSIVE_QUIET_SECONDS (their units chasing our
# retreat for PASSIVE_CHASE_SECONDS after it don't count). It doesn't come
# out: we expand early and often and build more tanks (e2bc988e: a turtle
# out-expanded us, 10 harvesters to 7, and out-built us behind its pillboxes).
PASSIVE_READ_SECONDS = 240.0
PASSIVE_QUIET_SECONDS = 240.0
PASSIVE_CHASE_SECONDS = 90.0
PASSIVE_PUSH_CELLS = 8             # their fighters (PUSH_SEEN_VALUE) this close to a building of ours (not a turret): an attack
EXPAND_AGAIN_PASSIVE_SECONDS = 60.0  # against a passive enemy the next expansion this soon after the last
TANK_VALUE_SHARE_PASSIVE = 0.7       # ...and this share of our combat value in tanks
PUSH_NEAR_BASE_CELLS = 20          # enemy fighters worth PUSH_SEEN_VALUE this close to a yard of ours = a push
PUSH_SEEN_VALUE = 1000
SIEGE_MAX_TURTLE = 6               # against a turtle: about one siege unit per turret, up to this...
SIEGE_ATTACK_MIN = 3               # ...and no attack before this many (unless overwhelming)
DEFENSE_CLUSTER_CELLS = 12         # turrets this close to the attack target defend it
RAID_FAILED_TARGET_SECONDS = 240.0  # a target where a raid lost more than it killed is left alone this long

# -- building placement ------------------------------------------------------------------------
REFINERY_ORE_SEARCH_CELLS = 16     # ore this close steers refineries (and keeps power off its side)
BUILDING_GAP_CELLS = 1             # free cells between buildings (OpenRA allows little more)
LANE_WIDTH_CELLS = 3               # kept-clear road from the base toward the staging point...
LANE_KEEP_CLEAR_CELLS = 14         # ...for this many cells
EXIT_CLEAR_ROWS = 2                # rows kept clear in front of barracks / war factories
# (width, height), top-left anchored, from OpenRA's mods/ra/rules/structures.yaml
# (the platform's rules give no sizes). Bib rows ("=") are part of the
# footprint: they must be on buildable ground and block other buildings.
#   fact xxX xxx XxX ===   proc _X_ xxx X== ===   fix _+_ +++ _+_
#   powr/barr/tent/dome xx xx ==   apwr/weap xxx xxx ===   turrets, silo, kenn x
FOOTPRINTS = {
    "fact": (3, 4), "powr": (2, 3), "apwr": (3, 3), "tent": (2, 3), "barr": (2, 3),
    "proc": (3, 4), "weap": (3, 3), "fix": (3, 3), "silo": (1, 1), "dome": (2, 3), "kenn": (1, 1),
    "pbox": (1, 1), "gun": (1, 1), "ftur": (1, 1), "tsla": (1, 1),
}
DEFAULT_FOOTPRINT = (2, 2)
# Footprint cells that aren't part of the building ("_"): not checked for
# ground, and gaps to neighbours are measured from the other cells (OpenRA's
# IsCloseEnoughToBase uses Tiles(), which skips them).
EMPTY_CELLS = {"proc": ((0, 0), (2, 0)), "fix": ((0, 0), (2, 0), (0, 2), (2, 2))}
# The reported centre cell's offset from the top-left: (dims - 1) / 2, half a
# cell up for buildings with LocalCenterOffset 0,-512 (the bib carriers),
# rounded (Building.CenterOffset). Elsewhere (w // 2, h // 2).
CENTRE_OFFSETS = {"proc": (1, 1), "fact": (1, 1)}

# -- defenses and repairs ----------------------------------------------------------------------
STANDING_DEFENSES = 1              # once the war factory is up
MAX_DEFENSES = 3                   # only while under attack...
DEFENSE_ADD_FUNDS = 600            # ...and each beyond the first only with this much money
DEFENSE_LINE_OFFSETS = (0, -3, 3, -6, 6)   # turret slots: cells across the line from the base straight to the enemy
DEFENSE_CONE_DEGREES = 35.0        # turrets stand within this of that line (when a spot is free there)
APPROACH_RING_CELLS = 12           # the approach: the cell this far from the base that is the shortest walk from the enemy
APPROACH_MAX_STEPS = 400
CHOKE_MAX_WIDTH = 11               # a choke: the enemy's walking path this narrow (cells across, cliffs to cliffs)...
CHOKE_MIN_CELLS = 4                # ...between this far...
CHOKE_MAX_CELLS = 16               # ...and this far from one of our yards (building range): its first tower watches it
WATCH_SPOT_CELLS = 5               # the tower goes at the choke when a legal spot is this close, else on the base edge facing it
DEFENSE_SIDE_WEIGHT = 2.0          # turret spots: each cell off the slot's line costs this many cells forward
DEFENSE_UNIT_SLACK = 2.0           # ...and units in the way only decide among spots within this of the best
DEFENSE_MAIN_RADIUS_CELLS = 16     # defenses this close to the base count toward its caps
DEFENSE_MIN_GAP_CELLS = 3          # a slot is taken if a defense is this close
DEFENSE_SIDE_MAX_DEGREES = 30.0    # the advisor's defense side must face the enemy within this
REPAIR_TYPES = ("fact", "powr", "apwr", "proc", "weap")
REPAIR_BELOW_HP = 50

# -- threats at the base ---------------------------------------------------------------------
CHEESE_NEAR_BASE_CELLS = 18        # enemies this close to any of our yards are a threat
CHEESE_MIN_ENEMY_UNITS = 2
CHEESE_EXPIRE_SECONDS = 25.0       # after the threat was last seen
CHEESE_WINDOW_SECONDS = 180.0      # a threat before this is a rush (rifle spam); later, DEFEND_BASE
OUTGUNNED_MIN_VALUE = 1500         # enemy units worth this and more than ours: no MCV or extra refinery (units first)
BASE_DEFEND_HOME_ODDS = 1.0        # later, the army comes home only for a threat at the main base worth more than
                                   # SMALL_RAID_VALUE that what defends it (units near it, turrets) doesn't beat...
BASE_DEFEND_EXPIRE_SECONDS = 20.0  # ...and goes back out this long after the base is clear (no bouncing)
UNDER_ATTACK_FRESH_TICKS = 250     # an under_attack event counts for 10 s

# -- attacking -----------------------------------------------------------------------------------
ENEMY_PEAK_SECONDS = 180.0         # their army: the most we saw of it this recently, less half what we killed since
SCOUT_FRESH_SECONDS = 45.0         # a scout round of their base this recent is a fresh look (not blind)
LOOK_BASE_FRONT_CELLS = 6          # a look that finds their exit empty goes on to this far in front of their spawn
KILL_ARMY_SHARE = 0.5              # a past sighting loses this share of our kills since (the rest were buildings and trucks)
# What they could have built since: their army grows by this share of what we
# spend on ours (army value + losses), unseen. On our self-play logs since the
# "no guess" estimate (545fd344 .. d2d8dc27) it read 1.3-12x low, biased
# -0.6 to -1.0 in log terms; with 0.6 the bias is about 0 and the error halved.
ENEMY_GROWTH_SHARE = 0.6
# ...but never more than their economy allows: their ore fields over the
# fields we work, times what we spend — at most ENEMY_HOME_FIELDS at home
# (two starting patches and the natural) until we see a remote expansion,
# then one more field each (every building goes up by a yard).
ENEMY_HOME_FIELDS = 3
EXPANSION_LOOK_AFTER_SECONDS = 240.0   # a new enemy expansion found after this sends a scout to look at their army
BELIEF_LOOK_CELLS = 10             # this many of our army units (BELIEF_LOOK_UNITS) this close to their base...
BELIEF_LOOK_UNITS = 3
BELIEF_DECAY_SECONDS = 30.0        # ...and the unseen part of their army fades over this (it isn't there)
ATTACK_ARMY_EDGE = 1.3             # our army / estimated enemy army (brain.enemy_estimate) needed to attack...
ESCALATED_ARMY_EDGE = 1.15         # ...lowered after ESCALATE_AFTER_SECONDS (1.0 attacked at parity and retreated within seconds: ef66403c)...
CATCH_UP_ARMY_EDGE = 1.0           # ...and to this when behind at the end
BIG_ARMY_VALUE = 15000             # an army this big pushes at...
BIG_ARMY_EDGE = 1.2                # ...this edge (c7a982ac: two 20k+ armies idled 10 minutes at ~1.1; but
                                   # d1615ac6 attacked 25150 into 24700 at 1.05 and lost 6800 for 3950: at
                                   # parity the defender's concave, turrets and short reinforcement win)
TRADE_WIN_MIN = 1000               # an attack that has killed this much more than it lost keeps going
ESCALATE_AFTER_SECONDS = 420.0
MIN_ATTACK_ARMY_VALUE = 1500       # never attack with less...
PUSH_ARMY_VALUE = 3000             # ...and push only with a real army built (unless behind at the end)
EARLY_ATTACK_SECONDS = 180.0       # before this (and a war factory), attack only when overwhelming
MODE_MIN_DWELL_SECONDS = 40.0      # no BUILD <-> ALL_IN flip faster than this
OVERWHELMING_ODDS = 2.5            # odds this good: finish them; the model's "hold" can't stop it; a turtle may be attacked
ADVISOR_ATTACK_EDGE = 1.0          # the advisor's "attack" posture starts an attack at these odds (never blind)
ADVISOR_HOLD_MAX_SECONDS = 90.0    # its "hold" delays an attack the rules call by at most this, then the rules go
TARGET_PRIORITY = (("powr", "apwr"), ("fact", "weap", "tent", "barr"), ("proc",))  # first to last
ATTACK_FOCUS_RADIUS_CELLS = 10     # priority applies among buildings this close to the nearest
STAGE_FORWARD_CELLS = 10           # mass (and rally new units) at least this far in front of the base, toward the enemy...
STAGE_SHARE = 0.25                 # ...or this share of the way to the enemy base if farther (big maps: Shadowfiend II 17 cells)...
STAGE_FRACTION = 0.30              # ...never farther than this share of the way to the enemy
CHOKE_STAGE_MAX_WIDTH = 7          # a passage this narrow on the enemy's road in is a choke the concave holds...
CHOKE_STAGE_MAX_FRACTION = 0.45    # ...when it lies at most this share of the way to them (and STAGE_FORWARD_CELLS+ out):
CHOKE_MOUTH_MAX_STEPS = 4          # the kill point goes where the way opens up past it (at most this many steps on)
STAGE_ENGAGE_CELLS = 12            # enemies this close to the army or the main base: the army's fight (executor.army_contact)
ENGAGE_MIN_EDGE = 1.25             # the army's odds (strength.odds) under this: a fight at the main base is held at the turret line
MATCHUP_MIN_UNITS = 5              # this many enemy fighters known before their mix moves our rifle:rocket ratio
ARMOR_HEAVY_SHARE = 0.6            # their hit points this armored: one rocket soldier per rifleman
OUTBOX_MAX_AGE_SECONDS = 1.5       # orders that didn't fit a batch go first in the next one, while this fresh
RETALIATE_SECONDS = 6.0            # a spot of ours attacked (event) or lost this recently, no attacker in sight...
RESPOND_RADIUS_CELLS = 12          # ...and any enemy group draw idle fighters this close (executor.respond):
                                   # all of them at a poke on the concave, else just enough to win...
RESPOND_ODDS = 2.0                 # ...these odds (strength.odds)...
RESPONSE_MIN_UNITS = 2             # ...with at least this many (a hit with nobody in sight: this many)
RESPONSE_HOLD_SECONDS = 20.0       # units sent at a group count as its answer this long (or till they get there)
GROUP_CELLS = 6                    # enemy fighters this close to each other are one group
UNSEEN_LEAD_MIN = 2000             # nothing of their army seen counts as nothing left only this far ahead on trades
CONCAVE_ORDERS_PER_VIEW = 6        # formation slot orders per view (the batch holds 20; the rest walk to the centre)
STAGING_RADIUS_CELLS = 4           # idle units already this close to the stage are left alone
STAGE_CLEAR_CELLS = 3              # no building this close to the massing point: the army stands there...
STAGE_MOVE_SEARCH_CELLS = 6        # ...and if one is (or the ground is blocked), the point moves this far at most
PLACE_CLEAR_WAIT_SECONDS = 5.0     # our units on a building's spot are moved off; placement waits this long at most
FLANK_OFFSET_FRACTION = 0.30       # flank waypoint: halfway, offset sideways by this share
FLANK_DANGER_RADIUS_CELLS = 12     # pick the side with fewest known enemies this close
FLANK_ARRIVED_CELLS = 6
FLANK_MAX_SECONDS = 90.0           # the route's waypoints are dropped after this: straight at the target
FLANK_AFTER_SECONDS = 600.0        # no flanking before 10:00 (unless decisively ahead): engaged early, a flank loses us the defense
HUNT_CHECKED_CELLS = 5             # a search point is checked once a unit gets this close (about its sight)
FINISH_MAX_BUILDINGS = 3           # this few enemy buildings known (and we lead): finish them, no holding back
FINISH_KILL_RATIO = 2.0            # ...or kills at least this x losses...
FINISH_MIN_LEAD = 5000             # ...and kills - losses at least this: dominant, finish them
ADVICE_TARGET_SUPPORT_CELLS = 10   # the advisor's army target counts only with a known enemy this close to it
ADVICE_TARGET_REACHED_CELLS = 6    # the army this close to the advisor's target: reached, the rules lead on...
ADVICE_REACHED_IGNORE_SECONDS = 30.0   # ...for this long (d1615ac6 stood 60 s on a reached point by their towers)
SELL_FOR_INCOME_SECONDS = 20.0     # broke with no harvester: sell a spare building this often
HUNT_GROUPS = 8                    # the hunt sweeps up to this many search points at once...
HUNT_GRID_CELLS = 12               # once their main base is razed: search cells this far apart over the whole map...
FAN_GROUP_UNITS = 3                # ...the army fanned out in groups of this many, one per cell...
FOUND_SQUAD_UNITS = 10             # ...and the nearest this many go and kill each building cluster found
HUNT_GROUP_MIN_UNITS = 6           # ...each group at least this many units
SPLIT_MIN_UNITS = 8                # overwhelming: split the army between the nearest enemy clusters...
SPLIT_UNITS_PER_CLUSTER = 10       # ...one per this many units...
SPLIT_MAX_CLUSTERS = 4             # ...up to this many (b90e492f razed four expansions two at a time)
ENEMY_CLUSTER_CELLS = 10           # enemy buildings this close together are one cluster

# -- retreating ------------------------------------------------------------------------------------
RETREAT_ARMY_EDGE = 0.8            # retreat below this edge while attacking...
RETREAT_EARLY_EDGE = 0.5           # ...but in the first MIN_ENGAGEMENT_SECONDS only below this (contact reveals their army)
REINFORCE_GAP_CELLS = 14           # in an attack, units this far behind the front wait at the massing point...
REINFORCE_GROUP_UNITS = 6          # ...until this many can go together (no trickling through a choke one by one)
REINFORCE_HOLD_CELLS = 8           # only units still this close to the massing point wait: none is ever sent back
DETOUR_MAX_RATIO = 1.6             # an attack goes round the chokes between the bases when that is at most this much longer
STRIKE_LOCAL_ODDS = 2.0           # the army strikes a remote expansion of theirs it beats this clearly (its turrets and units)...
STRIKE_MIN_EDGE = 0.75             # ...while not worse than this against their whole army (they must come out to it)
MARCH_DETOUR_MAX_RATIO = 2.5       # ...measured from where the army stands (at our concave by the choke's mouth the
                                   # corridor is short and every lane long: d1615ac6 went down Sidestep's gap)
CHOKE_ATTACK_FACTOR = 1.5          # with no way round, an attack through the choke needs this x the usual edge (their concave
                                   # fires on a column coming out of it), unless it brings SIEGE_ATTACK_MIN siege units
SCOUT_CHOKE_ENTRY_CELLS = 6        # a scout walks into a choke corridor only this close to where it is going
CHOKE_AVOID_CELLS = 4              # the choke zone (terrain.choke_zone): this far round each narrow cell on the road...
CHOKE_AVOID_BASE_CELLS = 13        # ...at least this far from either base (a base's own exit isn't avoided)
ARMY_LEG_CELLS = 10                # the attack's route as waypoints this many steps apart (it can't stray back)
ROUTE_RETARGET_CELLS = 12          # the attack's target moved this far: a new route from where the army is...
ROUTE_REPLAN_SECONDS = 15.0        # ...at most this often
ATTACK_SPREAD_NEAR_CELLS = 15      # the army spreads into its groups only this close to the target...
ATTACK_POINT_MAX_DETOUR = 1.3      # ...and only to points it walks to without a detour round terrain
TRADE_LEAD_WINDOW_SECONDS = 120.0  # traded up within this long...
TRADE_LEAD_MIN = 1500              # ...by this much (kills - losses)...
TRADE_LEAD_RATIO = 1.5             # ...killing this x what we lost: attack now, decisively
BLIND_SEEN_SHARE = 0.5             # no fresh look and less than this x our army seen lately: look before attacking
TRADE_WINDOW_SECONDS = 30.0        # ...or when, since the attack began (at most this far back),
TRADE_LOSS_MIN = 1000              # we lost at least this,
TRADE_LOSS_ARMY_SHARE = 0.25       # and this share of the army (the bigger of at the start and now),
TRADE_LOSS_RATIO = 1.3             # and more than this times what we killed (08560d9b bled out at 1.33),
MIN_ENGAGEMENT_SECONDS = 15.0      # after at least this long in the fight
REGROUPED_SHARE = 0.7              # retreat ends when this share is back...
REGROUP_RADIUS_CELLS = 8           # ...within this of the staging point
RETREAT_COOLDOWN_SECONDS = 60.0    # no new attack this soon after a retreat
REATTACK_ARMY_GROWTH = 1.3         # ...nor before the army regrows this much...
RETREAT_REGROW_MAX_SECONDS = 120.0 # ...unless this long has passed

# -- end of the match (tiebreak: kills - deaths, then assets) -----------------------------------
PROTECT_LEAD_REMAINING_SECONDS = 120.0  # final 2 minutes with a lead of PROTECT_LEAD_MIN_MARGIN...
PROTECT_LEAD_MIN_MARGIN = 1500
PROTECT_LEAD_LAST_SECONDS = 60.0        # ...or the final minute with any lead
CATCH_UP_REMAINING_SECONDS = 180.0      # behind or level this late: attack regardless

# -- squads: scout, raids, home guard, garrison ---------------------------------------------------
SCOUT_INSET_CELLS = 8              # patrol ring this far inside the map edge
SCOUT_FLANK_FRACTION = 0.30        # the scout's path bows out sideways this share of the base-to-base distance
SQUAD_CLAIM_RADIUS_CELLS = 14      # guards and garrisons take only units this close to home, never from the field
EARLY_PUSH_WINDOW_SECONDS = 360.0  # an early push: before 6:00, known enemy army of at least...
EARLY_PUSH_MIN_VALUE = 1500        # ...this value
EARLY_PUSH_RATIO = 1.0             # ...and our odds against it no better than this: hold at home, no attacks, turrets up
ENEMY_ORE_RADIUS_CELLS = 20        # their ore field: ore this close to their spawn
ENEMY_EXPANSION_WATCH_AFTER_SECONDS = 210.0
ENEMY_EXPANSION_SCOUT_SITES = 2
RAID_START_SECONDS = 150.0
RAID_SQUAD_SIZE = 4
RAID_SQUAD_SIZE_MACRO = 6          # against a macro-heavy enemy (the advisor's read): a bigger party...
RAID_FLANK_MIN_CELLS = 20          # ...that goes round the flank when its target is this far
RAID_MIN_SIZE = 3
RAID_UNIT_TYPES = ("e3", "e1", "jeep", "ftrk", "1tnk")   # rockets first: harvesters are armored
RAID_COOLDOWN_SECONDS = 30.0
RAID_SQUAD_MAX = 8                 # the squad grows with the army: one per RAID_ARMY_SHARE fighters, up to this
RAID_ARMY_SHARE = 6
# Healthy macro (HEALTHY_HARVESTERS+ harvesters, not outgunned): the longer the
# game goes, the more raid squads hunt for an edge instead of a big army idling.
HEALTHY_HARVESTERS = 4
RAID_EXTRA_MIN_ARMY_VALUE = 6000   # ...a real army behind them (ee2ac21b: 13 raiders beside a 9-unit army)...
RAID_LOSING_NET = 800              # the last few raids lost this much more than they killed...
RAID_BACKOFF_SECONDS = 120.0       # ...one squad only for this long
RAID_EXTRA_START_SECONDS = 360.0   # from 6:00 a second squad...
RAID_EXTRA_EVERY_SECONDS = 180.0   # ...one more every 3 minutes (and each squad one unit bigger)...
RAID_MAX_SQUADS = 4                # ...up to this many at once
RAID_MAX_ARMY_SHARE = 0.35         # the extra squads never take the raiders past this share of our fighters
RAID_LATE_COOLDOWN_SECONDS = 12.0  # healthy and past 6:00: a squad re-forms this soon
RAID_SPREAD_CELLS = 12             # each squad's target this far from the other squads' targets
RAID_JOIN_FIGHT_CELLS = 25         # an extra squad joins the main army's fight only from this close
REMOTE_SITE_RECHECK_SECONDS = 120.0  # a remote ore field checked by a picket or the raid is skipped this long
FIELD_WATCH_OUT_CELLS = 3          # a field is checked from beyond its 4 extreme ore cells, this far out...
FIELD_WATCH_SEEN_CELLS = 5         # ...each seen once any unit of ours has been this close (about its sight)
EXPANSION_ROAM_SECONDS = 480.0     # from 8:00 a second roamer checks the remote fields for their expansions...
RAID_EXPANSION_SECONDS = 480.0     # ...and a raid at a found expansion goes in force:
RAID_EXPANSION_SQUAD = 8           # at least this many (army and MIN_DEFENDERS allowing)
RAID_REMOTE_MIN_CELLS = 25         # raids only at a yard or refinery of theirs this far from their main base (a remote expansion)
RAID_NO_TARGET_GRACE_SECONDS = 8.0 # a raid keeps going through this long without a target (it flickers as units come and go)
RAID_TARGET_ODDS = 2.0             # a raid goes only where the squad clearly beats the units known there (strength.odds)
RAID_TURRET_CLEAR_CELLS = 7        # infantry raids keep this far from known enemy turrets (a pillbox reaches ~6)...
RAID_ROUTE_MAX_RATIO = 2.0         # ...walking round them when the way round is at most this much longer...
RAID_ARMOR_UNITS = 2               # ...else the raid waits for this many tanks or siege units to take along
RAID_THREAT_RADIUS_CELLS = 7       # enemy fighters this close to the raid squad are defending: fight them first...
RAID_OUTMATCH_ODDS = 1.0           # ...unless the squad's odds against them are under this: then it breaks off
DISTRACTION_SECONDS = 30.0         # the raid has enemy fighters on it (or just pulled back): they are distracted...
DISTRACTION_MIN_VALUE = 600        # ...when worth at least this...
DISTRACTION_ARMY_SHARE = 0.4       # ...and this share of their army: then the main army is judged against the rest
MAIN_FIGHT_RADIUS_CELLS = 9        # enemies this close to the main army: it is in a fight (the raid joins as a second prong)
SCOUT_FLEE_CELLS = 6               # the scout backs off from enemy fighters this close
SCOUT_SEEN_CELLS = 9               # a scout on a look or a peek turns back once it sees their fighters this close
SCOUT_BACKOFF_LOSSES = 3           # this many scouts lost...
SCOUT_BACKOFF_WINDOW_SECONDS = 120.0   # ...this recently: no looks or peeks...
SCOUT_BACKOFF_SECONDS = 90.0       # ...for this long after the last (5a4bc83f lost 58 scouts, the left one every ~30 s)
SCOUT_ALARM_SECONDS = 60.0         # two scouts killed far forward early within this: an early-push alarm
SCOUT_ALARM_LOSSES = 2             # (one dead scout is just a dead scout)
RUSH_HOLD_SECONDS = 240.0          # a rush read holds the army home until this (a rush arrives by then)
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
CIRCUIT_OUTER_CELLS = 14           # the rounds of their base: the left scout this far from their spawn...
CIRCUIT_INNER_CELLS = 7            # ...the right scout this far (through the base): a scout sees ~5 cells
CIRCUIT_DEGREES = (90, 125, 160, 195, 230)   # from its side round the back onto the other side: never across the front
SCOUT_LEG_CELLS = 8                # scouts walk in legs this long, round the chokes and the front of their base...
SCOUT_FRONT_CELLS = 15             # ...the front: this far from their spawn...
SCOUT_FRONT_DEGREES = 70.0         # ...this far either side of their way out toward us...
SCOUT_TURRET_CELLS = 8             # ...and this far round every enemy turret we know
SCOUT_MAX_DETOUR = 3.0             # the way round may be this much longer than the straight road
CIRCUIT_STAGGER_SECONDS = 10.0     # the right scout sets out this long after the left
CIRCUIT_ARRIVED_CELLS = 3
SCOUT_SNAP_CELLS = 6               # a post or waypoint walled in by terrain moves to a reachable cell this close
SCOUT_WALK_STEPS = 250             # (walking steps searched from our base)
INTEL_MIN_BUILDINGS = 3            # their opening is read once we know this many of their buildings
RUSH_READ_SECONDS = 100.0          # one refinery but a war factory / 2nd barracks by now: a rush...
RUSH_ARMY_SEEN_VALUE = 1000        # ...or an army this big seen before RUSH_HOLD_SECONDS (any refinery count)...
RUSH_EARLY_ARMY_SECONDS = 150.0    # ...or RUSH_EARLY_ARMY_VALUE seen before this (3bd6e450: 600 seen at 2:04,
RUSH_EARLY_ARMY_VALUE = 600        #  the wave hit at 2:50, the read came at 4:07)
SCOUT_PEEK_SECONDS = 45.0          # the lane scouts alternate their post with a look at the enemy's exit this often
SCOUT_PEEK_BACK_CELLS = 6          # the look stops this short of their massing point (it sees ~5 cells)...
SCOUT_PEEK_SIDE_CELLS = 6          # ...and this far to its own side (never down the centre lane)
INTERCEPT_MIN_VALUE = 1500         # an enemy army this big seen in our half: the army sets up across its path...
INTERCEPT_HOLD_SECONDS = 20.0      # the intercept point holds this long after the last sighting (no dithering)
# -- committing to an attack --
CONFIRM_SECONDS = 4.0              # an attack call is re-checked this long later, the army edging forward to see...
CONFIRM_BLIND_MAX_SECONDS = 40.0   # an attack call knowing little of their army (BLIND_SEEN_SHARE) waits up to this for a look
CONFIRM_TRADE_BAND = 200           # ...only while kills - losses are this close to even (clearly ahead/behind: go at once)
PROBE_FRACTION = 0.3               # ...to this share of the way from the massing point to the target
ATTACK_GROUPS = 3                  # the army attacks in this many groups, a semi-concave...
ATTACK_GROUP_SPREAD_CELLS = 5      # ...the side groups this far either side of the target
ATTACK_GROUP_MIN_UNITS = 6         # ...once the army has this many units
GARRISON_SIZE = 2                  # a small outpost at a forward expansion only...
GARRISON_TYPES = ("e1", "e3")      # ...of infantry: the tanks stay with the army
GARRISON_LEASH_CELLS = 12          # a garrison stays (and is led) within this of its expansion

# -- expansion -----------------------------------------------------------------------------------
EXPAND_AFTER_SECONDS = 225.0       # earliest start ("now" from the advisor: EXPAND_EARLIEST_SECONDS)
EXPAND_EARLIEST_SECONDS = 180.0
EXPAND_LAST_START_SECONDS = 840.0  # none after this: it can't pay off
EXPAND_MIN_HARVESTERS = 3
EXPAND_MIN_REFINERIES = 2
EXPAND_MIN_FIGHTERS = 8            # no expansion MCV with fewer fighters (08560d9b trained one with 3, and with 5)...
EXPAND_MCV_ARMY_VALUE = 2000       # ...or less army than this (the count includes the scouts: c4caa45e flipped at 8)
EXPAND_MAX_EXPANSIONS = 3
EXPAND_MAX_FAILURES = 2
EXPAND_AGAIN_AFTER_SECONDS = 120.0          # next expansion this long after the last...
EXPAND_EVEN_TRADE_BAND = 2000               # ...and with kills - losses within this, the advisor's "wait" doesn't hold it
# The ore forecast (harvest.HarvestManager): the observation has no live ore,
# so a field holds about ORE_VALUE_PER_CELL x its ore cells at the start (ore
# is 25 a unit, up to 12 a cell: OpenRA mods/ra/rules) and each harvester
# working it takes about HARVEST_RATE a second (500 a load, ~40 s a trip).
ORE_VALUE_PER_CELL = 300.0
HARVEST_RATE = 10.0
EXPAND_LEAD_SECONDS = 240.0                 # ...or sooner when the worked fields have less ore left than this (an expansion and its refinery take ~3-4 min)
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
EXPAND_CLEAR_WAIT_SECONDS = 10.0   # our units on the yard's spot get this long to step off before the MCV tries elsewhere
EXPAND_KEEP_OUT_CELLS = 1          # while the MCV drives there, nobody of ours is sent within this of the yard's cells
EXPAND_MOVE_TIMEOUT_SECONDS = 90.0 # then deploy where it stands...
EXPAND_GIVE_UP_SECONDS = 30.0      # ...then abandon it
EXPAND_STUCK_RADIUS_CELLS = 8      # stopped this close after EXPAND_STUCK_AFTER_SECONDS = arrived
EXPAND_STUCK_AFTER_SECONDS = 10.0
EXPAND_ROOM_SEARCH_CELLS = 8       # search radius for a spot to unfold (nearest first)
ORE_GROWTH = ((90.0, 1), (480.0, 2), (1020.0, 3))   # (game s, cells): how far fields with a mine may have grown
                                                    # past their starting ore (oregrowth.py: a Monte Carlo of the engine)
ORE_GROWTH_LOOKAHEAD_SECONDS = 60.0  # an MCV's spot is judged by the ore a minute from now (it is still on the road)
EXPAND_REFINERY_RADIUS_CELLS = 10  # a refinery this close to the new ore serves it
EXPAND_REFINERY_RETRY_SECONDS = 90.0
EXPAND_COOLDOWN_SECONDS = 60.0
EXPANSION_DEFENSES = 1
ACTIVE_BASE_DEFENSES = 3           # turrets at the expansion that takes over once home is mined out
REFINERY_ORE_WALK_STEPS = 6        # a refinery must be this few walking steps from its ore...
EXPAND_SITE_WALK_STEPS = 8         # ...an expansion deploy spot from its field's ore...
EXPAND_SITE_MIN_STEPS = 4          # ...and at least this far from it (room for the refinery between)
FIELD_WALK_STEPS = 14              # ...and a refinery's trucks work ore this few steps away
FIELD_NEAR_CELLS = 10              # a harvester this close to a field's ore is working it
FIELD_IDLE_SECONDS = 5.0           # idle this long there: the field is depleted...
FIELD_RECHECK_SECONDS = 120.0      # ...until re-checked (ore mines regrow)
FIELD_WORK_CELLS = 3               # a busy harvester this close to a field's ore is mining it
FIELD_STALE_SECONDS = 45.0         # harvesters around a field, none on its ore this long: depleted
REBALANCE_PER_VIEW = 2             # harvesters moved off a depleted field per view
HARVESTER_SETTLE_SECONDS = 30.0    # a moved harvester stays put at least this long

# -- units: micro, movement, stances --------------------------------------------------------------
FOCUS_RADIUS_CELLS = 4             # focus fire on the weakest enemy fighter this close (about weapon range); the target sticks
STUCK_SECONDS = 10.0               # an ordered unit that hasn't moved this long gets a stop
ENGAGED_RADIUS_CELLS = 7           # ...unless an enemy is this close
# The concave: ranks on arcs round a kill point FORMATION_FOCUS_AHEAD cells in
# front of the massing point, each rank within rifle range (5) of it, so every
# rifleman fires at whatever walks through (it was 16 cells wide, its ends 10
# cells forward: most of the line out of range of the middle).
FORMATION_FOCUS_AHEAD = 2.0
FORMATION_RANKS = (3.5, 4.5)       # arc radii, cells from the kill point
FORMATION_ARC_DEGREES = 70.0       # each arc spans this either side of straight back
FORMATION_SPACING_CELLS = 1.0      # cells between slots along an arc
FORMATION_MAX_UNITS = 20
STANCE_HOLD = "defend"             # waiting and guarding units
STANCE_PUSH = "attack_anything"    # attacking and defending the base
ARMY_STANCES = ("defend", "attack_anything", "return_fire", "hold_fire")   # the advisor may pick one for the waiting army
RALLY_TOLERANCE_CELLS = 2

# -- enemy memory ----------------------------------------------------------------------------------
ENEMY_UNIT_MEMORY_SECONDS = 90.0   # forget an unseen enemy unit after this
MEMORY_CLEAR_RADIUS_CELLS = 4      # or when we stand on its last cell and it isn't there
ATTACK_STALL_SECONDS = 40.0        # attacking with no kills and nothing in sight this long: a stale target
TRANSFORM_RADIUS_CELLS = 4         # an enemy MCV this close to a remembered yard: the yard packed up
HOME_RAID_RADIUS_CELLS = 10        # a group this close to the main base (buildings, trucks) also draws the idle units at home...
SMALL_RAID_VALUE = 1500            # ...and a group worth at most this is the idle units' job, more is the army's
RUSH_MIN_VALUE = 600               # in the opening, enemies at our base worth less are scouts, not a rush
SECOND_FACTORY_SECONDS = 360.0     # a second war factory from this (service depot up, its price in hand or money floating)
STALL_FORGET_CELLS = 10            # ...forget remembered buildings this close to the army...
STALL_IGNORE_ADVICE_SECONDS = 60.0 # ...and ignore the advisor's army target this long (the hunter leads)

# -- models (off the hot path; never waited on) -------------------------------------------------------
ADVISOR_INTERVAL_SECONDS = 6.0
ADVISOR_TIMEOUT_SECONDS = 10.0
ADVISOR_MAX_CALLS = 250
ADVICE_TTL_SECONDS = 20.0          # game seconds
ADVISOR_MEMORY_READS = 4           # its last this many changes of read go back to it with each call (advisor.memory)
ADVISOR_ACT_CONFIDENCE = 0.6       # its "aggressive" read puts turrets up only this sure
ADVISOR_FAST_CONFIDENCE = 0.8      # this sure, with evidence: its change takes effect without a second answer
ADVISOR_NO_EVIDENCE_CONFIDENCE = 0.5   # an answer with no evidence from this game is at most this sure
SQUAD_LEAD_INTERVAL_SECONDS = 4.0
SQUAD_LEAD_TIMEOUT_SECONDS = 6.0
SQUAD_LEAD_MAX_CALLS = 300         # per squad
SQUAD_LEAD_TTL_SECONDS = 8.0       # game seconds
SQUAD_LEAD_SIGHT_CELLS = 12

# -- order hygiene ---------------------------------------------------------------------------------------
ARMY_REISSUE_SECONDS = 3.0         # an unchanged army order isn't resent sooner
UNIT_REISSUE_SECONDS = 20.0        # a unit sent somewhere this recently...
ORDER_JITTER_CELLS = 5             # ...isn't re-sent to a spot this close to it (it would turn and lose its path)
REFUSAL_BLOCK_SECONDS = 4.0        # a refused order waits this long before a retry
PENDING_PRODUCTION_SECONDS = 1.5   # a build/train isn't resent before the state shows it


def resolve(candidates, available: set[str]) -> str | None:
    """First candidate id that the match actually offers (faction-agnostic)."""
    for item in candidates:
        if item in available:
            return item
    return None
