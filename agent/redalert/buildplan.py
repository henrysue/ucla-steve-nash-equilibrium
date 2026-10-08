"""The opening build order, progressed off the OBSERVED state.

Robustness comes from being *stateless*: we never track "which step are we on"
with a counter (a refused or retried order would desync it). Instead, every
view, we look at what is built and queued and decide the single next thing to
do. That self-corrects through refusals, deaths, and retries.

Two interrupts take priority over the sequence:

1. **Deploy the MCV** before anything (nothing builds without a construction
   yard).
2. **Power** — if the power margin is thin, queue a power plant before any
   other building, because low power slows all production (strategy hole #1).

A building is only queued when it is not already queued (strategy hole #5:
every extra build order queues another copy), and only when we can see it in
``available_production``.
"""

from __future__ import annotations

from . import config as C
from . import orders
from .observe import View


def _available(view: View, candidates) -> str | None:
    cands = (candidates,) if isinstance(candidates, str) else candidates
    return C.resolve(cands, view.available)


def _count_built_and_queued(view: View, type_ids) -> int:
    ids = (type_ids,) if isinstance(type_ids, str) else tuple(type_ids)
    built = sum(1 for b in view.buildings if b.type in ids and not b.wreck)
    queued = sum(view.queued_count(i) for i in ids)
    return built + queued


def next_build_orders(view: View) -> list[dict]:
    """Zero or one production order that advances the build order this view.

    Returns a list (0 or 1 orders) so the executor can concatenate it with
    other work. Placement of finished buildings is handled separately by
    :func:`placement_orders`.
    """
    # Interrupt 1: deploy the MCV into a construction yard.
    if not view.has_building(C.CONSTRUCTION_YARD):
        mcvs = view.units_of(C.MCV)
        if mcvs and mcvs[0].id is not None:
            return [orders.deploy(mcvs[0].id)]
        return []  # no yard and no MCV visible yet: nothing to do

    # Interrupt 2: power margin thin -> queue power before anything else.
    if view.power_margin < C.POWER_MARGIN:
        powr = _available(view, C.POWER_PLANT) or _available(view, C.ADV_POWER_PLANT)
        if powr and view.queued_count(powr) == 0:
            return [orders.build(powr)]
        # If power is queued/placing, fall through — don't stall other logic.

    # Walk the build order; queue the first step we don't yet have enough of.
    for step in C.BUILD_ORDER:
        resolved = _available(view, step)
        if resolved is None:
            continue  # not offered yet (e.g. weap needs prereqs) — skip for now
        want = _desired_count(step)
        if _count_built_and_queued(view, _ids(step)) < want:
            if view.queued_count(resolved) == 0:  # don't double-queue
                return [orders.build(resolved)]
            return []  # it's queued; wait for it
    return []


def placement_orders(view: View) -> list[dict]:
    """Place every finished-but-unplaced building (ready in the queue).

    ``place`` without ``at`` drops it next to the base, which is exactly what a
    one-base aggro agent wants — no manual positioning micro.
    """
    out = []
    seen: set[str] = set()
    for entry in view.production:
        item = str(entry.get("item") or "").lower()
        if entry.get("ready") and item and item not in seen and _is_building(view, item):
            out.append(orders.place(item))
            seen.add(item)
    return out


# -- helpers -----------------------------------------------------------------


def _ids(step):
    return (step,) if isinstance(step, str) else tuple(step)


def _desired_count(step) -> int:
    # The build order lists REFINERY twice (starter + one income expansion);
    # everything else we want exactly one of.
    if step == C.REFINERY:
        return 2
    return 1


def _is_building(view: View, item: str) -> bool:
    """Whether a ready queue item is a building (placed) vs a unit (auto-exits).

    We treat the known building ids as buildings; everything else (units) does
    not need a place order. Robust to unknown ids: default to not-a-building so
    we never emit a spurious place for a unit.
    """
    building_ids = {C.CONSTRUCTION_YARD, C.POWER_PLANT, C.ADV_POWER_PLANT,
                    C.REFINERY, C.WAR_FACTORY, *C.BARRACKS}
    return item in building_ids
