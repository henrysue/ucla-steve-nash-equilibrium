"""The strategist: one match, three kinds of decision.

- Draft pick: every card on the board is scored by draft.py (team plan,
  roles, coverage, matchup with the opponent's picks, strength); the model
  picks from the best few. No model: the top score.
- Team Preview: ONE model call writes the match plan and chooses the four
  and the leads, checked against the lineup rules (re-asked once if it
  breaks them). No model: the best-scoring legal four.
- Battle turn, in four steps:
    1. rules.py decides what is withheld and what a blunder is (a second
       Protect in a row from the same Pokemon is never sent);
    2. predict.py guesses each foe's action (Protect only when protect.py's
       one table says it is likely, i.e. its first turn out; otherwise it
       attacks the one of ours it hurts most), and the model refines that read;
    3. lookahead.ranked_lines ranks our legal answers - principles first,
       reads next: the read is blended with the code's guess
       (predict.blend), and principles.Judge prices every line that breaks
       a decision principle (protect the plan, do your job, stay robust,
       mind self-drops);
    4. the model makes the final call: play the code's pick, or veto it for
       another ranked line within VETO_MARGIN. A veto only chooses among
       rule-checked lines, and is sent only once the model locks it in on a
       second, thinking look (the code's pick beside it, the value gap, and
       the code's checks of its reason against its own read).
  The code's guess and its answer are prepared first: no model, a slow
  model or a bad answer all still get a sensible move.

One promise runs through all of it: a model problem costs decision quality,
never a legal move. Every path prepares a fallback first, and any
unexpected failure ends in a guaranteed-legal action.
"""

from __future__ import annotations

import collections
import itertools
import os
import random
import time
from pathlib import Path
from typing import Any, Callable

from altruagent import DecisionContext, GameState, LegalAction, WithReasoning
from altruagent.console import print_line

from . import actions as act
from . import archetypes, cards as card_sets, decisions, draft, lookahead, meta, movelist, observed, pick_priority, predict, principles, protect, rules
from .style import OpponentStyle
from .moverules import MoveTracker
from .providers import NO_CREDENTIALS, StrategistProvider
from .report import Analysis, Decision, MatchLog
from .typechart import effectiveness

# Platform clocks (GAMES.md): 15 s a draft pick, 90 s Team Preview, 55 s a
# battle decision from a shared bank. We answer well inside each.
TEAM_PREVIEW_SECONDS = 78.0
DECISION_SECONDS = 40.0
MIN_ATTEMPT_SECONDS = 4.0  # less time than this left: play the fallback now
MAX_ATTEMPTS = 2  # the first answer plus one retry carrying the reason
RETRY_BUDGET_SHARE = 0.5  # a retry only starts with half the budget left
FASTER_EFFORT = {"max": "high", "xhigh": "high", "high": "medium", "medium": "low", "low": "low", "none": "none"}

SUMMARY_LIMIT = 280
# The platform shows reasoning_summary to spectators (and maybe the
# opponent). By default it carries the decision's one-line note (battle:
# "Read: ... Play: ... <note>"); VGC_PUBLIC_REASONING=0 sends the neutral
# PUBLIC_SUMMARY instead. The model's private thinking is never sent.
PUBLIC_SUMMARY = "Move submitted."
PUBLIC_REASONING_ENV = "VGC_PUBLIC_REASONING"
PUBLIC_REASONING_OFF = ("0", "false", "no", "off")

MIN_BRING_ATTACKERS = 3
RECENT_TURNS = 3  # turns of history in each battle prompt
# A veto may only pick a line within this much of the code's pick (damage
# scale). Match 4e60a5a2, turn 1: a veto 121 points down sent Thunder Wave
# into a Protect.
VETO_MARGIN = 40.0
# The best line that breaks no principle is offered beyond the margin, but not
# this far down: match 4b317ad0 vetoed a 154-point-worse line and lost Iron Valiant.
CLEAN_LINE_MARGIN = 2 * VETO_MARGIN
# A veto is sent only once the strategist locks it in on a second, thinking
# look (match a938ac5c, turn 1: a "sure KO" veto read Follow Me, which would
# have pulled that very attack). Less time than this left: the code's pick.
MIN_LOCK_IN_SECONDS = 8.0


class VGCDraftStrategist:
    """One match's agent; ``create_agent()`` builds a fresh one per match."""

    def __init__(
        self,
        provider: StrategistProvider | None,
        *,
        log: Callable[[str], None] = print_line,
        rng: random.Random | None = None,
        clock: Callable[[], float] = time.monotonic,
        log_dir: str | None = None,
    ) -> None:
        self._provider = provider
        self._log = log
        self._public_reasoning = str(os.environ.get(PUBLIC_REASONING_ENV, "")).strip().lower() not in PUBLIC_REASONING_OFF
        self._rng = rng or random.Random()  # unseeded: used for the opening team-plan choice
        self._clock = clock
        self._matchlog = MatchLog(log, log_dir)
        # Every draft card is an exact set; the opponent drafts from the same
        # board, so their moves, items and abilities are known, not guessed.
        self._cards = card_sets.CardSets()
        self._meta = self._cards.overlay(meta.table())
        self._table_checked = False
        # Late-bound so the reference table can be swapped mid-match.
        self._moves = MoveTracker(
            known_moves=lambda s: self._meta.common_moves(s), offense=lambda s: self._meta.offense(s),
            exact_moves=lambda s: (self._meta.card_set(s) or {}).get("moves"),
            ko_threats=self._ko_threats,
        )
        self._draft_plan = ""  # the model's one-line team plan, carried pick to pick
        self._draft_target: dict = {}
        self._game_plan: dict | None = None
        self._recent: collections.deque = collections.deque(maxlen=RECENT_TURNS)
        self._turn_rules: rules.TurnRules | None = None
        self._turn_odds: dict[str, dict] = {}
        self._line_flags: list[list[str]] = []  # principle flags of the last ranked lines, in order
        self._offered: list | None = None  # what the last final call showed the strategist
        self._board_now: dict | None = None  # this decision's exact board, for the log
        self._brought: list[str] = []  # species ids of the four we brought
        self._in_range: set[str] = set()  # foes one of our attacks likely KOs this turn
        # What happened between decisions (no battle log is sent): the last
        # snapshot, the last full turn's prediction, and this decision's report.
        self._before: dict | None = None
        self._last_prediction: dict | None = None
        self._happened: dict | None = None
        self._our_switches: set[str] = set()  # who we switched out with the last payload
        self._our_moves: dict[str, str] = {}  # species id -> the move it used with the last payload
        self._board: list | None = None  # this draft pick's cards, for the log
        # Book player or one who plays around our reads? Learned from its Protects.
        self._style = OpponentStyle()
        self._opponent_id: str | None = None
        self._preview_opponent: list[str] = []  # their six at Team Preview
        self._draft_reasons: list[dict] = []  # why each of our picks was made (pick_priority's rules)
        self._turn = 0
        self.model_calls = 0
        self._last_attempts, self._last_seconds, self._last_reasoning = 0, 0.0, None
        self._opponent_seen: list[str] = []
        self._opponent_names: list[str] = []  # their display names, from the seat's assignment
        self._agent_id: str | None = None
        if provider is None:
            self._log(f"[vgc] {NO_CREDENTIALS}")
        else:
            self._log(f"[vgc] model: {provider!r}")
        if not len(self._meta):
            self._log(f"[vgc] No reference table loaded. Expected {meta.DEFAULT_CSV}, or set VGC_META_CSV.")

    # -- entry point ----------------------------------------------------------

    def choose_action(self, state: GameState, context: DecisionContext):
        if _is_realtime(state):
            raise act.TemplateError(
                f"Your agent can't play {_game_name(context)}: the VGC Draft strategist is a "
                "Pokémon specialist, and a move in a real-time game is a batch of orders it has "
                "no way to write. See GAMES.md, or run with --agent examples.llm_agent, which "
                "plays all three games."
            )
        self._agent_id = str(getattr(context, "agent_id", "") or "") or self._agent_id
        if getattr(state, "is_terminal", False):
            self.summarize_match(state)
            raise act.TemplateError("choose_action was called on a terminal state")
        if not state.legal_actions:
            raise act.TemplateError("choose_action was called with no legal actions")
        self._matchlog.name_match(context.session_id, context.seat_position, context.game_type)
        self._opponent_names = [str(n) for n in getattr(context, "opponents", ()) or () if n] or self._opponent_names
        self._last_attempts, self._last_seconds, self._last_reasoning = 0, 0.0, None
        first = state.legal_actions[0]
        if first.action_id == "submit_team":
            raise act.TemplateError(
                "submit_team (build-your-own-team modes) needs a full 6-Pokémon team; this "
                "strategist plays pokemon_vgc_doubles_draft."
            )
        # The floor beneath every fallback: an exception escaping here
        # forfeits the match, so any unexpected failure plays a legal move.
        try:
            if first.action_id == "select_lineup":
                return self._team_preview(first, state, context)
            if first.action_id == "doubles_turn":
                return self._doubles_turn(first, state, context)
            return self._draft_pick(state, context)
        except act.TemplateError:
            raise
        except Exception as exc:  # noqa: BLE001 - deliberately broad; see above
            self._log(f"[vgc] decision failed before a move was chosen [{type(exc).__name__}]: {exc}")
            return self._last_resort(state, exc)

    # -- draft ----------------------------------------------------------------

    def _draft_pick(self, state: GameState, context: DecisionContext):
        observation = _observation(state)
        cards = _cards_by_id(observation)
        self._cards.remember(list(cards.values()))
        self._board = [{k: card[k] for k in ("card_id", "species", "item", "ability", "moves") if k in card}
                       for card in cards.values()]
        self._check_table([_species(card) for card in cards.values()])
        self._meet_opponent(observation, context)
        mine = _roster_species(observation, context, own=True)
        theirs = _roster_species(observation, context, own=False)
        turn = f"Draft pick {observation.get('pick_number', self._matchlog.decisions + 1)}"

        # Species clause: a second form of a species we hold is never offered.
        held = {self._meta.base_species(s) for s in mine}
        offered = [
            (action, card_id) for action in state.legal_actions
            if (card_id := _card_id(action)) is None or self._meta.base_species(_species(cards.get(card_id))) not in held
        ] or [(action, _card_id(action)) for action in state.legal_actions]

        board = [s for _, card_id in offered if (s := _species(cards.get(card_id)))]
        styles = archetypes.infer_playstyle(theirs, self._meta)
        plan, runner_up, self._draft_target = archetypes.choose_plan(mine, theirs, board, styles, self._meta, rng=self._rng)
        mine_roles = [self._roles(s) for s in mine]
        fit = {
            card_id: draft.score(
                species, self._roles(species, (cards.get(card_id) or {}).get("moves")),
                mine, mine_roles, theirs, plan, runner_up, styles, self._meta,
            )
            for _, card_id in offered
            if (species := _species(cards.get(card_id)))
        }
        have_restricted = bool(self._meta.restricted_in(mine))

        def value(pair) -> tuple[int, float]:
            species = _species(cards.get(pair[1]))
            # Only one restricted/Mega is worth bringing: a second ranks last.
            spare = 0 if have_restricted and self._meta.is_restricted(species) else 1
            return (spare, fit.get(pair[1], {}).get("total", 0.0))

        offered.sort(key=value, reverse=True)
        # The pick: (1) a must pick, (2) a proven partner of our first pick,
        # (3) the strongest card, nudged by team fit. No model call: fast on the
        # 15-second clock, and the same board always gets the same pick.
        legal = [pair for pair in offered if value(pair)[0]]  # never a second restricted / Mega
        by_species = {}
        for action, card_id in legal or offered:
            species = _species(cards.get(card_id))
            if species and species not in by_species:
                by_species[species] = (action, card_id)
        pick = pick_priority.choose(
            list(by_species), mine, self._meta,
            fit={s: fit.get(cid, {}).get("total", 0.0) for s, (_, cid) in by_species.items()},
            cards={s: cards.get(cid) or {} for s, (_, cid) in by_species.items()},
            theirs=theirs,
        )
        if pick is None:
            return self._fallback(offered[0][0], "No card could be ranked; took the best-fitting card.", turn=turn,
                                  phase=state.phase, kind="draft_pick")
        chosen = by_species[pick.species][0]
        self._draft_reasons.append({"pick": observation.get("pick_number"), "species": pick.species, "rule": pick.rule,
                                    "reason": pick.reason})
        self._draft_plan = self._draft_summary()
        return self._record(chosen, f"{pick.species}: {pick.reason}", turn=turn, phase=state.phase, kind="draft_pick")

    def _draft_summary(self) -> str:
        """Our draft in a few lines: must picks, partner cores, what we build toward."""
        species = [r["species"] for r in self._draft_reasons]
        musts = [r["species"] for r in self._draft_reasons if r["rule"] == "must pick"]
        cores = []
        for i, a in enumerate(species):
            for b in species[i + 1:]:
                record = pick_priority.complements(a, b) or pick_priority.complements(b, a)
                if record:
                    cores.append(f"{a} + {b} ({record[0]:.1f}% together over {record[1]:,} games)")
        target = (self._draft_target or {}).get("name") if isinstance(self._draft_target, dict) else None
        parts = []
        if musts:
            parts.append("must picks: " + ", ".join(musts))
        if cores:
            parts.append("proven pairs: " + "; ".join(cores))
        if target:
            parts.append(f"building toward: {target}")
        return " | ".join(parts)[:SUMMARY_LIMIT * 2]

    def draft_strategy(self) -> dict:
        """What the draft was trying to do, for Team Preview and every battle turn."""
        return {"summary": self._draft_summary() or "(no draft reasoning recorded)",
                "picks": [f"{r['species']} - {r['rule']}: {r['reason']}" for r in self._draft_reasons]}

    def _check_table(self, names: list[str]) -> None:
        """Once per match, use the reference table matching the format the
        server is actually dealing (Reg I or Champions Reg M-C)."""
        names = [n for n in names if n]
        if self._table_checked or len(names) < 6:
            return
        self._table_checked = True
        better = meta.best_table(names, self._meta)
        if better is not self._meta:
            self._log(
                f"[vgc] reference table switched to {Path(better.source or '').name}: it covers "
                f"{better.coverage(names):.0%} of the species on offer vs {self._meta.coverage(names):.0%}"
            )
            self._meta = self._cards.overlay(better)

    def _roles(self, species: str | None, moves: list[str] | None = None) -> dict:
        roles = self._meta.roles(species, moves)
        for role in archetypes.declared_roles(species or "", self._meta):  # forms the table lacks
            roles[role] = True
        return roles

    # -- team preview: one call, plan + lineup -------------------------------------------

    def _team_preview(self, action: LegalAction, state: GameState, context: DecisionContext):
        roster = act.parse_lineup(action)
        observation = _observation(state)
        self._meet_opponent(observation, context)
        self._preview_opponent = [
            e.get("species") or e.get("name") for e in observation.get("opponent_roster") or []
            if isinstance(e, dict) and (e.get("species") or e.get("name"))
        ]
        # Both full rosters are shown here; any sets they carry are exact.
        self._cards.remember(observation.get("your_roster"))
        self._cards.remember(observation.get("opponent_roster"))
        opponent = [
            e.get("species") or e.get("name") for e in observation.get("opponent_roster") or []
            if isinstance(e, dict) and (e.get("species") or e.get("name"))
        ]
        self._check_table(roster + opponent)
        try:
            bring = self._best_four(roster)
            fallback = act.build_lineup(roster, bring, self._leads_for(bring))
        except Exception as exc:  # noqa: BLE001
            self._log(f"[vgc] lineup fallback unavailable [{type(exc).__name__}]: {exc}")
            fallback = act.guaranteed_action(state)
        if self._provider is None:
            return self._fallback(fallback, "No model configured: brought the best legal four.", turn="Team Preview", phase=state.phase, kind="select_lineup")

        decision = decisions.team_preview_decision(
            roster,
            [self._team_entry(s, opponent, own=True) for s in roster],
            [self._team_entry(s, opponent, own=False) for s in opponent],
            self._lineup_rules_text(),
        ).with_prompt({
            "our_draft_strategy": self.draft_strategy(),
            "our_draft_plan": self._draft_target or "(none recorded)",
            "their_playstyle_read": archetypes.infer_playstyle(opponent, self._meta),
        })
        first_answer = iter((True,))  # only the first answer must follow the lineup rules

        def checked(answer: dict) -> dict:
            lineup = act.build_lineup(roster, answer.get("bring"), answer.get("leads"))
            problems = self._lineup_problems(roster, lineup["bring"], lineup["leads"])
            if problems and next(first_answer, False):
                raise act.InvalidAnswer(f"the lineup breaks lineup_rules: {'; '.join(problems)}")
            return lineup

        asked = self._ask(decision, state, context, TEAM_PREVIEW_SECONDS, parse=checked)
        if asked is None:
            return self._fallback(fallback, "Out of time at Team Preview; brought the best legal four.", turn="Team Preview", phase=state.phase, kind="select_lineup")
        plan, lineup = asked
        self._game_plan = {k: v for k, v in plan.items() if k not in ("reasoning_summary", "bring", "leads")}
        self._matchlog.write(Decision(
            turn="Game plan", phase=str(state.phase or ""), kind="game_plan", payload=self._game_plan,
            public_summary="(private; not sent to the platform)", model=getattr(self._provider, "model", None),
            attempts=self._last_attempts, seconds=self._last_seconds,
        ))
        return self._record(lineup, _summary(plan), turn="Team Preview", phase=state.phase, kind="select_lineup")

    def _team_entry(self, species: str, opponent: list[str], *, own: bool) -> dict:
        entry = {
            "species": species,
            "roles": [k for k, v in self._roles(species).items() if v],
            "abilities": self._meta.abilities(species),
        }
        if own:
            entry["moves"] = self._meta.common_moves(species)
            entry["item"] = self._meta.common_item(species)
            score, detail = draft.versus(species, opponent, self._meta)
            entry["vs_their_team"] = {"score": round(score, 2), **detail}
        elif self._meta.card_set(species):
            entry["moves"] = self._meta.common_moves(species)  # its draft card: exact
            entry["item"] = self._meta.common_item(species)
        else:
            entry["likely_moves"] = self._meta.common_moves(species)
            entry["likely_item"] = self._meta.common_item(species)
        return entry

    def _lineup_rules_text(self) -> list[str]:
        return [
            f"bring at least {MIN_BRING_ATTACKERS} attackers when the roster has them",
            "if you bring a Fake Out user, lead with one",
            "bring a Tailwind setter if the roster has a strong one; we never play Trick Room",
            "do not lead two Pokémon where one's spread move (Earthquake, Surf...) hits the other",
            "at most one restricted legendary or Mega in the four",
            "do not bring two Pokémon 4x weak, or three weak, to an attack type their team carries",
            "do not bring two Pokémon whose shared STAB type half their team resists",
        ]

    def _lineup_problems(self, roster: list[str], bring: list[str], leads: list[str]) -> list[str]:
        """Every lineup rule a bring/lead breaks (empty when it fits)."""
        roles = {species: self._roles(species) for species in roster}
        problems = []
        need = min(MIN_BRING_ATTACKERS, sum(r["attacker"] for r in roles.values()))
        if sum(roles[s]["attacker"] for s in bring) < need:
            problems.append(f"fewer than {need} attackers")
        fake_out = [s for s in bring if roles[s]["fake_out"]]
        if fake_out and not set(leads) & set(fake_out):
            problems.append(f"no Fake Out lead ({', '.join(fake_out)} brought)")
        setters = [s for s in roster if self._sets_speed(s)]
        if setters and not any(s in setters for s in bring):
            problems.append(f"no Tailwind setter ({', '.join(setters)} left home)")
        clash = self._lead_clash(leads)
        if clash:
            problems.append(clash)
        if len(self._meta.restricted_in(bring)) > 1:
            problems.append("more than one restricted legendary or Mega")
        stacked = self._stacked_weakness(bring)
        # Only when some four from the roster avoids it: never rule out every lineup.
        if stacked and any(not self._stacked_weakness(list(four)) for four in itertools.combinations(roster, 4)):
            problems.append(stacked)
        walled = self._stacked_attack(bring)
        if walled and any(not self._stacked_attack(list(four)) for four in itertools.combinations(roster, 4)):
            problems.append(walled)
        return problems

    def _stacked_attack(self, bring: list[str]) -> str | None:
        """Two or more of ours share a STAB type that at least half their team
        resists or is immune to. Match ead979d8: Dondozo and Basculegion (Water)
        into Archaludon, Rillaboom and Amoonguss - their Water moves were
        withheld as resisted, leaving Earthquake and Aqua Jet."""
        theirs = [[t.upper() for t in self._meta.species_types(s)] for s in self._preview_opponent]
        theirs = [t for t in theirs if t]
        if not theirs:
            return None
        sharing: dict[str, list[str]] = {}
        for species in bring:
            for attack in {t.upper() for t in self._meta.species_types(species)}:
                sharing.setdefault(attack, []).append(species)
        for attack, users in sorted(sharing.items()):
            if len(users) < 2:
                continue
            resist = sum(1 for types in theirs if effectiveness(attack, types) < 1)
            if resist * 2 >= len(theirs):
                return (f"{', '.join(users)} share {attack.title()} attacks, which {resist} of their "
                        f"{len(theirs)} resist")
        return None

    def _stacked_weakness(self, bring: list[str]) -> str | None:
        """An attack type their team carries that hits two of ours 4x, or three 2x+.
        Match 45d8b07d brought Salamence and Garchomp (both 4x) into Iron Bundle's Freeze-Dry."""
        their_attacks = draft.attack_types(self._preview_opponent, self._meta)
        for attack in sorted(their_attacks):
            hits = {s: effectiveness(attack, [t.upper() for t in self._meta.species_types(s)]) for s in bring}
            quad = [s for s, x in hits.items() if x >= 4]
            weak = [s for s, x in hits.items() if x > 1]
            if len(quad) >= 2 or len(weak) >= 3:
                names = ", ".join(f"{s} ({hits[s]:g}x)" for s in weak)
                return f"stacks a weakness to {attack.title()}, which their team can hit: {names}"
        return None

    def _lead_clash(self, leads: list[str]) -> str | None:
        """A lead whose spread move (Earthquake, Surf...) would hit the other lead."""
        if len(leads) != 2:
            return None
        for user, partner in ((leads[0], leads[1]), (leads[1], leads[0])):
            moves = self._meta.common_moves(user)
            if not moves:  # no usage data: the same-type spread moves it can learn
                own_types = {t.upper() for t in self._meta.species_types(user)}
                learnable = [m.strip() for m in self._meta.field(user, "spread_moves").split(",") if m.strip()]
                moves = [m for m in learnable if (movelist.lookup(m) or {}).get("type") in own_types]
            partner_types = [t.upper() for t in self._meta.species_types(partner)]
            for name in moves:
                move = movelist.lookup(name)
                if move and move["target"] == "allAdjacent" and move["power"] and partner_types:
                    if effectiveness(move["type"], partner_types) > 0:
                        return f"{user}'s {move['name']} hits lead partner {partner}"
        return None

    def _sets_speed(self, species: str) -> bool:
        """A Tailwind setter worth bringing: the drafted card's moves, else usage
        data, else (no data at all) what it can learn - and easy enough to play
        (a weak setter is never forced over a stronger Pokemon; match f14f1652
        forced in a 44.6% Hatterene). We never play Trick Room."""
        if pick_priority.easy_strength(species, self._meta, self._meta.card_set(species))[0] < 0:
            return False
        card = self._meta.card_set(species)
        moves = card["moves"] if card else None
        if moves:
            return "tailwind" in {meta.species_id(m) for m in moves}
        known = self._meta.common_moves(species) or self._meta.field(species, "role")
        return "tailwind" in archetypes.tags(species, self._meta, learnable=not known)

    def _leads_for(self, bring: list[str]) -> list[str]:
        """No-model leads: a Fake Out user beside the strongest attacker."""
        by_offense = sorted(bring, key=lambda s: -(self._meta.offense(s) or 0))
        fake_out = [s for s in by_offense if self._roles(s)["fake_out"]]
        if not fake_out:
            return by_offense[:2]
        first = fake_out[-1]  # the weakest Fake Out user; stronger bodies attack
        return [first, next(s for s in by_offense if s != first)]

    def _best_four(self, roster: list[str]) -> list[str]:
        """No-model bring: the highest-value four that breaks no lineup rule."""
        fitting = [
            list(four) for four in itertools.combinations(roster, 4)
            if not self._lineup_problems(roster, list(four), self._leads_for(list(four)))
        ]
        if fitting:
            # Keep the draft's proven pairs together, then the strongest four.
            return max(fitting, key=lambda four: (_proven_pairs(four), sum(self._meta.draft_points(s) or 0 for s in four)))
        ranked = sorted(roster, key=lambda s: -(self._meta.draft_points(s) or 0))
        bring = [s for s in ranked if not self._meta.is_restricted(s)][:4]
        return bring + [s for s in ranked if s not in bring][: 4 - len(bring)]

    # -- battle turn: rules -> predict the opponent -> best legal answer ----------------

    def _doubles_turn(self, action: LegalAction, state: GameState, context: DecisionContext):
        self._turn += 1
        turn = f"Turn {self._turn}"
        started = self._clock()  # the read and the final call share one budget
        slots = act.parse_doubles(action)
        observation = _observation(state)
        self._board_now = _safely(lambda: _board_detail(observation, self._brought), None)
        self._remember_opponent(observation)

        # 1. Rules: what cannot work or would be a blunder is withheld/rejected.
        turn_rules = rules.build(slots, observation, self._moves, self._meta, self._log)
        self._turn_rules = turn_rules
        self._observe(observation)
        if self._recent and self._moves.protected_last_turn:
            self._recent[-1]["they_protected"] = self._moves.protected_last_turn
        self._style.resolve(self._happened, self._turn - 1)  # what last turn's Protect/switch choices say about them
        self._turn_odds = self._protect_odds(observation)
        self._watch_style(observation, self._turn_odds)
        calc = lookahead.Calculator(observation, self._meta)
        self._calc = calc
        self._in_range = _safely(lambda: self._ko_range(turn_rules, observation, calc), set())
        if any(o.get("type") == "move" for slot in slots for o in slot.options):
            self._style.watch_ko(self._in_range)  # what they do with these is scored next turn
        ours = _active_species(observation.get("active_pokemon"))
        foes = _active_species(observation.get("opponent_active_pokemon"))

        # 2. The code's guess at the opponent; the move it implies is the fallback.
        guess = _safely(lambda: predict.code_guess(observation, calc, self._moves, self._meta, self._turn_odds,
                                                   in_range=self._in_range), {})
        fallback = self._answer(turn_rules, observation, calc, guess, ours, state)

        if self._provider is None or not any(o["type"] == "move" for slot in slots for o in slot.options):
            reason = "No model configured" if self._provider is None else "A replacement after a faint"
            return self._play(fallback, turn, state, guess, "code", guess, "", fallback_reason=f"{reason}: played the best answer to the code's prediction." if self._provider is None else None)

        # 2b. The model reads the opponent; its prediction wins when it is usable.
        facts = self._prediction_facts(observation, calc, guess)
        decision = decisions.predict_decision(foes, ours, facts)
        asked = self._ask(decision, state, context, DECISION_SECONDS,
                          parse=lambda answer: predict.from_model(answer, foes, ours))
        if asked is None:
            return self._play(fallback, turn, state, guess, "code", guess, "",
                              fallback_reason="No usable model prediction in time; played the best answer to the code's prediction.")
        answer, from_model = asked
        prediction = {**guess, **from_model}  # any foe the model skipped keeps the code's guess
        # 3. Our best legal answers to that prediction, ranked by code.
        line = self._answer(turn_rules, observation, calc, prediction, ours, state, base=guess)
        state_summary = str(answer.get("state_summary") or "")
        # 4. The strategist makes the final call: play the code's pick or veto it.
        played, review = self._review(line, prediction, state_summary, facts, state, context, started)
        return self._play(line, turn, state, prediction, "model", guess, state_summary,
                          summary=_summary(answer), played=played, review=review)

    def _review(self, ranked: list, prediction: dict, state_summary: str, facts: dict,
                state: GameState, context: DecisionContext, started: float) -> tuple[int, dict | None]:
        """The model plays ranked[0] or vetoes it for another ranked line.

        Every option already passed the hard rules, so a veto can never send
        a blocked move (a second Protect in a row, a used-up Fake Out). Lines
        within VETO_MARGIN of the code's pick are offered, plus - when the
        code's pick breaks a decision principle - the best line that breaks
        none. No answer in the time left: the code's pick stands.
        """
        self._offered = None
        offered = [i for i, line in enumerate(ranked) if line[0] >= ranked[0][0] - VETO_MARGIN]
        flags = self._line_flags if len(self._line_flags) == len(ranked) else [[] for _ in ranked]
        if flags[0]:
            # The code's pick breaks a principle: the best line that breaks none
            # is offered too, if it is within CLEAN_LINE_MARGIN.
            clean = next((i for i in range(len(ranked)) if not flags[i]), None)
            if clean is not None and clean not in offered and ranked[clean][0] >= ranked[0][0] - CLEAN_LINE_MARGIN:
                offered.append(clean)
        if len(offered) < 2:
            return 0, None  # nothing close enough to veto into
        read = (self._last_attempts, self._last_seconds, self._last_reasoning)
        slots = self._turn_rules.slots if self._turn_rules else []
        options = [{"option": number, "value": ranked[i][0], "line": _name_payload(ranked[i][1], slots),
                    "breaks_principles": flags[i] or "none"}
                   for number, i in enumerate(offered, start=1)]
        self._offered = options
        decision = decisions.review_decision(options, {
            "your_read": {"state_summary": state_summary, "prediction": prediction},
            "our_move_history": self._moves.snapshot(self._turn_rules.per_slot)["ours"] if self._turn_rules else {},
            **{k: facts[k] for k in ("our_actives", "their_actives", "their_threats", "our_threats", "move_order",
                                     "their_possible_switch_ins", "game_plan", "recent_turns")},
        })

        def parse(answer: dict) -> int:
            number = int(answer.get("play_option") or 0)
            if answer.get("verdict") == "play":
                chosen = 0
            elif not 2 <= number <= len(offered):
                raise act.InvalidAnswer(f"a veto must name another option (2-{len(offered)}), got {number}")
            else:
                chosen = offered[number - 1]
            # Yuriko games, turn 7: "Draco Meteor secures the Porygon2 kill" for an
            # option that sent Draco Meteor into Archaludon. The restated line must match.
            said = str(answer.get("chosen_line") or "")
            missing = _line_mismatch(said, ranked[chosen][1], slots)
            if said and missing:
                raise act.InvalidAnswer(
                    f"option {offered.index(chosen) + 1} is '{_name_payload(ranked[chosen][1], slots)}', but chosen_line "
                    f"'{said}' does not name {missing}. Re-read the options.")
            return chosen

        asked = self._ask(decision, state, context, DECISION_SECONDS - (self._clock() - started), parse=parse)
        attempts, _, reasoning = self._last_attempts, self._last_seconds, self._last_reasoning
        self._last_attempts = read[0] + attempts
        self._last_seconds = self._clock() - started
        if reasoning and read[2] and reasoning != read[2]:
            self._last_reasoning = f"Read: {read[2]}\n\nFinal call: {reasoning}"
        else:
            self._last_reasoning = read[2] or reasoning
        if asked is None:
            self._log("[vgc] no final call from the strategist in time; played the code's pick")
            return 0, None
        answer, played = asked
        review = {"verdict": "veto" if played else "play", "play_option": offered.index(played) + 1,
                  "why": str(answer.get("why") or "")}
        if played:
            locked, lock_why = self._lock_in(ranked, played, flags, review["why"], prediction, decision,
                                             state, context, started)
            review["lock_in"] = {"locked": locked, "why": lock_why}
            if not locked:
                self._log(f"[vgc] strategist withdrew its veto for {_name_payload(ranked[played][1], slots)}: {lock_why}")
                return 0, {**review, "verdict": "play", "play_option": 1, "withdrawn": offered.index(played) + 1}
        if played:
            self._log(f"[vgc] strategist vetoed {_name_payload(ranked[0][1], slots)} for "
                      f"{_name_payload(ranked[played][1], slots)}: {review['why']}")
        return played, review

    def _lock_in(self, ranked: list, played: int, flags: list, why: str, prediction: dict,
                 review: decisions.Decision, state: GameState, context: DecisionContext,
                 started: float) -> tuple[bool, str]:
        """The second, thinking look a veto must pass before it is sent: the
        code's pick beside the veto, the value gap, and the code's checks of
        the strategist's reason against its own read. No confirmation in
        time, or a withdrawal: the code's pick stands."""
        left = DECISION_SECONDS - (self._clock() - started)
        if left < MIN_LOCK_IN_SECONDS:
            return False, f"no time to lock in the veto ({left:.0f}s left)"
        slots = self._turn_rules.slots if self._turn_rules else []
        veto = {
            "code_pick": _name_payload(ranked[0][1], slots),
            "your_line": _name_payload(ranked[played][1], slots),
            "value_gap": round(ranked[0][0] - ranked[played][0], 1),
            "code_pick_breaks": flags[0] or "none",
            "your_line_breaks": flags[played] or "none",
            "your_reason": why,
            "checks": _veto_checks(ranked[played][1], prediction, slots) or ["none found"],
        }
        facts = {k: v for k, v in review.prompt.items() if k not in ("instructions", "options", "decision")}
        reasoning, attempts = self._last_reasoning, self._last_attempts
        asked = self._ask(decisions.veto_lock_in_decision(veto, facts), state, context, left,
                          parse=lambda answer: bool(answer.get("lock_in")))
        self._last_attempts += attempts
        self._last_seconds = self._clock() - started
        if self._last_reasoning and reasoning and self._last_reasoning != reasoning:
            self._last_reasoning = f"{reasoning}\n\nLock-in: {self._last_reasoning}"
        else:
            self._last_reasoning = reasoning
        if asked is None:
            return False, "no lock-in answer in time"
        answer, locked = asked
        return locked, str(answer.get("why") or "")

    def _observe(self, observation: dict) -> None:
        """Compare with the previous decision; check the last full turn's read."""
        try:
            now = observed.snapshot(observation)
            self._happened = None if self._before is None else observed.last_turn(
                self._before, now, self._last_prediction, self._moves.protected_last_turn, self._our_switches, self._our_moves)
            self._before = now
            self._last_prediction = None  # each read is checked once
        except Exception as exc:  # noqa: BLE001 - bookkeeping must never cost the move
            self._log(f"[vgc] last turn could not be reconstructed [{type(exc).__name__}]: {exc}")
            self._happened = None

    def _answer(self, turn_rules: rules.TurnRules, observation: dict, calc, prediction: dict, ours: list[str],
                state: GameState, *, base: dict | None = None):
        """Step 3: ranked legal answers; the first is the code's pick.

        Principles first, reads next: with ``base`` (the code's rule-based
        guess) the model's read is blended with it (predict.blend), and every
        line pays principles.Judge's price for breaking a decision principle.
        """
        self._line_flags = []
        try:
            expected = predict.expectations(prediction, ours)
            if base is not None and base is not prediction:
                expected = predict.blend(expected, predict.expectations(base, ours))
            predict.no_repeat_protect(expected, self._moves.protected_last_turn)
            predict.escape(expected, self._in_range,
                           {f: (self._turn_odds.get(f) or {}).get("likelihood", 0) > 0 for f in self._in_range},
                           self._style.ko_escape_rate, bool(self._their_bench(observation)))
            target, _ = _safely(lambda: predict.focus_target(calc.threats() or {}, {
                meta.species_id(e.get("species") or e.get("name")): predict._hp_of(e)
                for e in observation.get("active_pokemon") or [] if isinstance(e, dict)}), (None, ""))
            predict.focus(expected, target, self._style.focus_rate)  # who they double-target
            self._style.add_switching(expected)  # this opponent switches this often
            judge = principles.Judge(turn_rules.slots, observation, calc, expected,
                                     preserve=(self._game_plan or {}).get("preserve") or (),
                                     effects=self._effects_up(observation), just_in=self._moves.just_in())
            ranked = lookahead.ranked_lines(turn_rules, observation, calc, expected, bench=self._their_bench(observation),
                                            cost=lambda payload: judge.judge(payload)[0])
            if ranked:
                lines = [(value, turn_rules.check(answer)) for value, answer in ranked]
                self._line_flags = [judge.judge(payload)[1] for _, payload in lines]
                return lines
        except Exception as exc:  # noqa: BLE001 - never cost the move
            self._log(f"[vgc] move selection failed [{type(exc).__name__}]: {exc}")
        return [(0.0, act.guaranteed_action(state))]

    def _ko_range(self, turn_rules, observation: dict, calc) -> set[str]:
        """Their actives one of our attacks likely KOs this turn."""
        judge = principles.Judge(turn_rules.slots, observation, calc, {})
        found: set[str] = set()
        for slot in turn_rules.slots:
            found |= judge._kos(slot)
        return found

    def _ko_threats(self, observation: dict) -> set[str]:
        """Their actives whose best hit KOs one of ours this turn (for the opening Fake Out)."""
        calc = lookahead.Calculator(observation, self._meta)
        return {meta.species_id(foe) for foe, row in (calc.threats() or {}).items()
                if any(hit.get("ko_this_turn_likely") for hit in row.values())}

    def _effects_up(self, observation: dict) -> set[str]:
        """Our side's and the field's effects now up: the server's, plus the move tracker's."""
        shown = observed.field_state(observation) or {}
        found = set(shown.get("ours") or []) | set(shown.get("field") or [])
        return found | {m for m in ("tailwind", "reflect", "lightscreen", "auroraveil", "trickroom")
                        if self._moves.effect_turns_left(m) > 0}

    def _play(self, ranked: list, turn: str, state: GameState, prediction: dict, source: str, guess: dict,
              state_summary: str, *, summary: str = "", fallback_reason: str | None = None,
              played: int = 0, review: dict | None = None):
        analysis = Analysis(
            turn=turn, state_summary=state_summary, prediction=prediction, source=source, code_guess=guess,
            lines=[(value, _describe_payload(payload) + _flag_text(self._line_flags, index))
                   for index, (value, payload) in enumerate(ranked)], played=played, review=review,
            offered=self._offered if source == "model" else None, board=self._board_now,
        )
        payload = ranked[played][1]
        note = summary or (fallback_reason or "")
        if review and review["verdict"] == "veto":
            note = f"{note} Veto: {review['why']}".strip()
        summary = _turn_summary(prediction, payload, note)
        slots = self._turn_rules.slots if self._turn_rules else []
        if any(o.get("type") == "move" for slot in slots for o in slot.options):
            self._last_prediction = prediction  # a full turn: check it against what happens
        return self._record(payload, summary, turn=turn, phase=state.phase,
                            kind="doubles_turn", analysis=analysis, fallback_reason=fallback_reason)

    def _prediction_facts(self, observation: dict, calc, guess: dict) -> dict:
        """What the model reads the opponent from."""
        def side(entries) -> list[dict]:
            out = []
            for e in entries or []:
                if isinstance(e, dict) and (e.get("species") or e.get("name")):
                    keep = ("species", "types", "hp_fraction", "current_hp", "max_hp", "boosts", "status", "item", "ability")
                    out.append({k: e[k] for k in keep if k in e})
            return out

        theirs = _active_species(observation.get("opponent_active_pokemon"))
        return {
            "our_actives": side(observation.get("active_pokemon")),
            "their_actives": side(observation.get("opponent_active_pokemon")),
            "their_sets": {
                n: {"moves": self._meta.common_moves(n), "item": self._meta.common_item(n), "ability": self._meta.abilities(n),
                    "source": "draft card (exact)" if self._meta.card_set(n) else "usage guess"}
                for n in theirs
            },
            "protected_last_turn": self._moves.protected_last_turn or "none known",
            "protect_odds": self._turn_odds,
            "their_threats": _safely(calc.threats, {}),
            "our_threats": _safely(lambda: _our_threats(observation, calc), {}),
            "first_turn_on_field": self._moves.opponent_first_turn(),
            "opponent_switch_rate": self._style.summary()["switch_rate"],
            "in_our_ko_range": sorted(self._in_range) or "none",
            "their_focus_target": _safely(lambda: predict.focus_target(calc.threats() or {}, {
                meta.species_id(e.get("species") or e.get("name")): predict._hp_of(e)
                for e in observation.get("active_pokemon") or [] if isinstance(e, dict)})[1], "") or "none",
            "focus_fire": self._style.summary()["focus_fire"],
            "ko_range_escape": self._style.summary()["ko_range_escape"],
            "our_draft_strategy": self.draft_strategy()["summary"],
            "their_possible_switch_ins": self._their_bench(observation),
            "move_order": _safely(lambda: self._move_order(observation, calc), {}),
            "code_guess": guess,
            "game_plan": self._game_plan or "(none)",
            "recent_turns": list(self._recent),
        }

    def _their_bench(self, observation: dict) -> list[str]:
        """Their Pokemon that could switch in now: seen in battle (or, until all
        four are seen, shown at Team Preview), not on the field, not fainted."""
        active = {meta.species_id(s) for s in _active_species(observation.get("opponent_active_pokemon"))}
        fainted = {k for k, gone in ((self._before or {}).get("their_fainted") or {}).items() if gone}
        seen = [meta.species_id(s) for s in self._opponent_seen]
        pool = seen + ([meta.species_id(s) for s in self._preview_opponent] if len(set(seen)) < 4 else [])
        return [s for s in dict.fromkeys(pool) if s not in active and s not in fainted]

    def _meet_opponent(self, observation: dict, context: DecisionContext) -> None:
        """Once: who we play (the other agent in ``rosters``), and what we learned about them before."""
        if self._opponent_id is not None:
            return
        rosters = observation.get("rosters")
        seat = str(getattr(context, "agent_id", "") or "")
        others = [k for k in rosters if k != seat] if isinstance(rosters, dict) and seat else []
        if len(others) == 1:
            self._opponent_id = str(others[0])
            self._style = OpponentStyle.for_opponent(self._opponent_id)

    def _watch_style(self, observation: dict, odds: dict[str, dict]) -> None:
        """Log how this opponent uses Protect. A record only: the odds stay
        protect.py's one table (match 4e60a5a2: the style blend saw its
        Protects but not its attacks, and over-predicted Protect all game)."""
        try:
            carries = {}
            for entry in observation.get("opponent_active_pokemon") or []:
                if isinstance(entry, dict) and (entry.get("species") or entry.get("name")):
                    name = entry.get("species") or entry.get("name")
                    known = {meta.species_id(m) for m in self._meta.common_moves(name)} | set(_revealed(entry))
                    carries[meta.species_id(name)] = bool(known & predict.PROTECT_CHAIN)
            chained = {foe: True for foe in self._moves.protected_last_turn}
            self._style.open(odds, carries, chained, self._turn)
        except Exception as exc:  # noqa: BLE001 - a record only; never cost the move
            self._log(f"[vgc] opponent style unavailable this turn [{type(exc).__name__}]: {exc}")

    def _move_order(self, observation: dict, calc) -> dict:
        """Who acts first this turn: priority brackets, then effective Speed
        (Choice Scarf, boosts, Tailwind; reversed under Trick Room)."""
        from .observed import field_state

        shown = field_state(observation) or {"ours": [], "theirs": [], "field": []}
        trick_room = "trickroom" in shown["field"]
        first_turn = self._moves.opponent_first_turn()
        speeds, priority = {}, {}
        for side, key in (("active_pokemon", "ours"), ("opponent_active_pokemon", "theirs")):
            for entry in observation.get(side) or []:
                if not isinstance(entry, dict) or not (entry.get("species") or entry.get("name")):
                    continue
                name = entry.get("species") or entry.get("name")
                speed = calc.speed(entry)
                if speed:
                    speeds[name] = round(speed * (2 if "tailwind" in shown[key] else 1))
                moves = []
                for move_name in self._meta.common_moves(name) or []:
                    move = movelist.lookup(move_name)
                    if not move:
                        continue
                    bracket = _priority_bracket(move, name, entry, self._meta, calc)
                    if bracket > 0:
                        note = ""
                        if meta.species_id(move_name) in ("fakeout", "firstimpression"):
                            live = first_turn.get(meta.species_id(name)) if key == "theirs" else None
                            note = " (first turn out only" + ("; live now)" if live else ")")
                        elif meta.species_id(move_name) in ("suckerpunch", "thunderclap"):
                            note = " (fails unless the target attacks)"
                        moves.append(f"{move['name']} +{bracket}{note}")
                if moves:
                    priority[name] = moves
        order = sorted(speeds, key=lambda n: speeds[n], reverse=not trick_room)
        return {
            "rule": "higher priority bracket first (+4 Protect, +3 Fake Out / Quick Guard, +2 Extreme Speed, "
                    "+1 Sucker Punch / Aqua Jet / Prankster status), then faster first"
                    + (" - Trick Room is up: SLOWER first within a bracket" if trick_room else ""),
            "speed_order_within_a_bracket": order,
            "effective_speeds": speeds,
            "priority_moves": priority,
        }

    def _protect_odds(self, observation: dict) -> dict[str, dict]:
        try:
            return protect.odds(
                observation, first_turn=self._moves.opponent_first_turn(),
                common_moves=self._meta.common_moves, card_set=self._meta.card_set,
                protected_last_turn=set(self._moves.protected_last_turn),
                first_turn_rate=self._style.first_turn_rate,
            )
        except Exception as exc:  # noqa: BLE001 - advice only; never cost the move
            self._log(f"[vgc] protect odds unavailable this turn [{type(exc).__name__}]: {exc}")
            return {}

    def _remember_opponent(self, observation: dict) -> None:
        seen = _active_species(observation.get("opponent_active_pokemon")) + [
            e.get("species") or e.get("name") for e in observation.get("opponent_roster") or [] if isinstance(e, dict)
        ]
        for name in seen:
            if name and name not in self._opponent_seen:
                self._opponent_seen.append(name)

    # -- match end -------------------------------------------------------------

    def on_match_end(self, final_state: GameState, context: DecisionContext) -> None:
        """The runner's end-of-match hook: record the result in the match log."""
        self._agent_id = str(getattr(context, "agent_id", "") or "") or self._agent_id
        self._matchlog.name_match(context.session_id, context.seat_position, context.game_type)
        summary = self.summarize_match(final_state)
        self._log(f"[vgc] match over: {summary.get('outcome') or 'outcome unknown'}")

    def summarize_match(self, state: GameState | None = None, *, note: str | None = None) -> dict:
        """End-of-match summary; the outcome comes only from the server."""
        self._style.save(self._opponent_id)
        return self._matchlog.summarize(outcome=self._outcome_from(state), opponent=self._opponent_seen, note=note,
                                        opponent_names=self._opponent_names,
                                        ending=_ending(state, self._before), style={**self._style.summary(), "notes": self._style.notes,
                                                                      "opponent_id": self._opponent_id})

    def _outcome_from(self, state: GameState | None) -> str | None:
        if state is None:
            return None
        from altruagent.worker import game_outcome

        said = game_outcome(state, str(self._agent_id or ""))
        if said in ("won", "lost", "drew"):
            return {"won": "win", "lost": "loss", "drew": "draw"}[said]
        final = getattr(state, "final_result", None)
        winner = final.get("winner_agent_id") if isinstance(final, dict) else None
        if winner is None:
            return None
        return "win" if str(winner) == str(self._agent_id or "") else "loss"

    # -- model plumbing -------------------------------------------------------

    def _ask(self, decision: decisions.Decision, state: GameState, context: DecisionContext, budget: float, *,
             parse: Callable[[dict], Any] | None = None):
        """One decision's model call, retried once, inside ``budget`` seconds.

        ``parse`` runs inside the retry loop, so every way an answer can be
        unusable earns the one retry rather than raising. Returns ``None``
        when no valid answer arrived in time: the caller plays its fallback.
        """
        assert self._provider is not None
        messages = decision.messages(state, context.seat_position, context.game_type)
        started = self._clock()
        deadline = started + budget
        effort = decision.effort
        for attempt in range(1, MAX_ATTEMPTS + 1):
            left = deadline - self._clock()
            if left < MIN_ATTEMPT_SECONDS or (attempt > 1 and left < budget * RETRY_BUDGET_SHARE):
                self._log(f"[vgc] {decision.kind}: out of time for another model call ({left:.0f}s left)")
                break
            self.model_calls += 1
            self._last_attempts = attempt
            try:
                answer = self._provider.complete_structured(messages, decision.kind, decision.schema, timeout=left, effort=effort)
                self._last_seconds = self._clock() - started
                self._last_reasoning = getattr(self._provider, "last_reasoning", None)
                return answer if parse is None else (answer, parse(answer))
            except Exception as exc:  # noqa: BLE001 - a model problem must never cost the move
                self._log(f"[vgc] {decision.kind}: unusable answer (attempt {attempt}/{MAX_ATTEMPTS}) [{type(exc).__name__}]: {exc}")
                if "Timeout" in str(exc):
                    effort = FASTER_EFFORT.get(effort, effort)
                    continue
                messages = messages + [{
                    "role": "user",
                    "content": f"That answer was invalid: {exc}. Answer again using only the options given.",
                }]
        self._last_seconds = self._clock() - started
        return None

    def _record(self, payload: Any, summary: str, *, turn: str, phase: object, kind: str,
                analysis: Analysis | None = None, fallback_reason: str | None = None):
        """Audit one decision, then hand it to the runtime with a neutral public line."""
        move_state = None
        if kind == "select_lineup" and isinstance(payload, dict):
            self._brought = [meta.species_id(s) for s in payload.get("bring") or []]
        if kind == "doubles_turn":
            self._our_switches = _switched_out(payload, self._turn_rules)
            self._our_moves = _moves_used(payload, self._turn_rules)
            self._recent.append({
                "turn": turn,
                "we_did": _describe_payload(payload),
                "board": (analysis.state_summary if analysis else "")[:SUMMARY_LIMIT],
            })
            if self._turn_rules is not None:
                try:
                    move_state = {
                        **self._moves.snapshot(self._turn_rules.per_slot),
                        "protect_odds": self._turn_odds,
                        "opponent_style": {"label": self._style.label, "p_strong": round(self._style.p_strong, 2)},
                        "protected_last_turn": self._moves.protected_last_turn,
                    }
                    self._moves.record(payload)
                    self._moves.expect(_least_damage(payload, self._turn_rules, getattr(self, "_calc", None)))
                except Exception as exc:  # noqa: BLE001 - bookkeeping must never cost the move
                    self._log(f"[vgc] move tracker could not record this turn [{type(exc).__name__}]: {exc}")
        self._matchlog.write(Decision(
            turn=turn, phase=str(phase or ""), kind=kind, payload=payload, public_summary=summary,
            model=getattr(self._provider, "model", None), model_reasoning=self._last_reasoning,
            attempts=self._last_attempts, seconds=self._last_seconds, fallback_reason=fallback_reason,
            analysis=analysis, move_state=move_state,
            last_turn=self._happened if kind == "doubles_turn" else None,
            board=self._board if kind == "draft_pick" else None,
        ))
        public = summary.strip()[:SUMMARY_LIMIT] if self._public_reasoning and summary.strip() else PUBLIC_SUMMARY
        return WithReasoning(payload, public)

    def _fallback(self, payload: Any, reason: str, *, turn: str, phase: object, kind: str):
        return self._record(payload, reason, turn=turn, phase=phase, kind=kind, fallback_reason=reason)

    def _last_resort(self, state: GameState, exc: Exception):
        """Play a guaranteed-legal move after an unexpected failure."""
        reason = f"Recovered from an unexpected {type(exc).__name__} with a guaranteed-legal move: {exc}"
        try:
            payload = act.guaranteed_action(state)
        except Exception as inner:  # noqa: BLE001  # pragma: no cover
            payload = {"type": "doubles_turn", "slot_0": dict(act.PASS), "slot_1": dict(act.PASS)}
            reason = f"{reason} (even the guaranteed action raised {type(inner).__name__}: {inner})"
        kind = "doubles_turn" if isinstance(payload, dict) and payload.get("type") == "doubles_turn" else "select_lineup"
        return self._record(payload, reason, turn=f"Turn {self._turn}" if kind == "doubles_turn" else "Team Preview",
                            phase=state.phase, kind=kind, fallback_reason=reason)


# -- small helpers ------------------------------------------------------------


def _safely(function: Callable[[], Any], default: Any) -> Any:
    try:
        return function()
    except Exception:  # noqa: BLE001 - a missing fact never costs the move
        return default


def _describe_payload(payload: object) -> str:
    """A compact 'slot: action' line for the battle history."""
    if not isinstance(payload, dict):
        return str(payload)
    parts = []
    for number in (0, 1):
        choice = payload.get(f"slot_{number}") or {}
        if choice.get("type") == "move":
            target = choice.get("target")
            parts.append(f"slot {number}: {choice.get('move_id')}" + (f" -> {target}" if target is not None else ""))
        elif choice.get("type") == "switch":
            parts.append(f"slot {number}: switch to {choice.get('species')}")
    return "; ".join(parts) or "pass"


def _board_detail(observation: dict, brought: list[str] | None = None) -> dict:
    """Every active's exact HP, boosts, status and item, and our bench's HP, so a
    turn can be replayed (match 41d776ed, turn 12 could not be)."""
    def row(entry: dict) -> dict:
        found = {"hp": round(lookahead._fraction(entry), 3)}
        if entry.get("current_hp") is not None and entry.get("max_hp"):
            found["hp_exact"] = f"{entry['current_hp']}/{entry['max_hp']}"
        for key in ("boosts", "status", "item", "ability"):
            if entry.get(key):
                found[key] = entry[key]
        return found

    def side(entries) -> dict:
        return {e.get("species") or e.get("name"): row(e) for e in entries or [] if isinstance(e, dict)
                and (e.get("species") or e.get("name"))}

    team = observation.get("team")
    bench = team.values() if isinstance(team, dict) else team if isinstance(team, list) else []
    board = {"ours": side(observation.get("active_pokemon")), "theirs": side(observation.get("opponent_active_pokemon"))}
    names = {meta.species_id(n) for n in board["ours"]}
    wanted = {meta.species_id(s) for s in brought or []}
    benched = {e.get("species") or e.get("name"): round(lookahead._fraction(e), 3) for e in bench
               if isinstance(e, dict) and (e.get("species") or e.get("name"))
               and meta.species_id(e.get("species") or e.get("name")) not in names
               and (not wanted or meta.species_id(e.get("species") or e.get("name")) in wanted)}
    if benched:
        board["our_bench_hp"] = benched
    fields = observed.field_state(observation)
    if fields:
        board["field"] = fields
    return board


def _veto_checks(payload: object, prediction: dict, slots: list) -> list[str]:
    """What the strategist's own read says about the line it vetoed for: a
    redirection read pulls its single-target attacks, a Protect read wastes them."""
    found = []
    if not isinstance(payload, dict):
        return found
    for slot in slots:
        choice = payload.get(f"slot_{slot.number}") or {}
        move = movelist.lookup(choice.get("move_id")) if choice.get("type") == "move" else None
        target = choice.get("target")
        if not move or move["target"] != "normal" or not isinstance(target, int) or target <= 0:
            continue
        names = {e.get("target"): e.get("species") for o in slot.options for e in o.get("target_options") or []
                 if isinstance(e, dict) and e.get("species")}
        aimed = meta.species_id(names.get(target) or "")
        active = slot.active.get("species") if isinstance(slot.active, dict) else slot.active
        who = f"{active}'s {move['name']}"
        for foe, guess in (prediction or {}).items():
            if not isinstance(guess, dict):
                continue
            foe_id, confidence = meta.species_id(foe), float(guess.get("confidence") or 0)
            if (guess.get("action") == "support" and meta.species_id(guess.get("move") or "") in predict.REDIRECTION
                    and foe_id != aimed and move["power"]):
                found.append(f"Your read: {foe} uses {guess.get('move')} ({confidence:.0%}). {who} at {names.get(target)} "
                             f"is pulled into {foe} instead, so it does not do what your reason says.")
            if guess.get("action") == "protect" and foe_id == aimed:
                found.append(f"Your read: {foe} Protects ({confidence:.0%}). {who} into it is wasted.")
    return found


def _line_mismatch(said: str, payload: object, slots: list) -> str:
    """What the restated line leaves out of the option's moves, targets and
    switch-ins, slot by slot ('' when it names them all). Each slot is read
    from the part of the line that names its Pokemon (parts split on ';'),
    else the part in its position."""
    parts = [part for part in str(said).split(";") if part.strip()] or [str(said)]
    missing = []
    for position, slot in enumerate(slots):
        choice = (payload or {}).get(f"slot_{slot.number}") if isinstance(payload, dict) else None
        if not isinstance(choice, dict):
            continue
        active = slot.active.get("species") if isinstance(slot.active, dict) else slot.active
        mine = [part for part in parts if active and meta.species_id(active) in meta.species_id(part)]
        text = meta.species_id(mine[0] if mine else parts[min(position, len(parts) - 1)])
        if choice.get("type") == "switch":
            want = [str(choice.get("species") or "")]
        elif choice.get("type") == "move":
            move = movelist.lookup(choice.get("move_id"))
            want = [move["name"] if move else str(choice.get("move_id"))]
            target = choice.get("target")
            if isinstance(target, int) and target > 0:
                names = {e.get("target"): e.get("species") for o in slot.options for e in o.get("target_options") or []
                         if isinstance(e, dict) and e.get("species")}
                if names.get(target):
                    want.append(str(names[target]))
        else:
            continue
        missing += [w for w in want if w and meta.species_id(w) not in text]
    return ", ".join(missing)


def _flag_text(flags: list[list[str]], index: int) -> str:
    """' [breaks: ...]' for a ranked line's principle flags, or ''."""
    found = flags[index] if 0 <= index < len(flags) else []
    return f" [breaks: {'; '.join(found)}]" if found else ""


PIVOT_MOVES = frozenset({"uturn", "voltswitch", "flipturn", "partingshot", "teleport", "chillyreception", "shedtail"})


def _id_of(name: object) -> str:
    return meta.species_id(name)


def _name_payload(payload: object, slots: list) -> str:
    """'Dondozo: Wave Crash -> Gholdengo; Tornadus: Protect' - species by name, for the model."""
    if not isinstance(payload, dict):
        return str(payload)
    parts = []
    for slot in slots:
        choice = payload.get(f"slot_{slot.number}") or {}
        active = slot.active.get("species") or slot.active.get("name") if isinstance(slot.active, dict) else slot.active
        who = str(active or f"slot {slot.number}")
        if choice.get("type") == "move":
            move = movelist.lookup(choice.get("move_id"))
            text = f"{who}: {move['name'] if move else choice.get('move_id')}"
            if move and _id_of(move["name"]) in PIVOT_MOVES and not any(o.get("type") == "switch" for o in slot.options):
                text += " (no one to switch in: it stays)"  # match 4b317ad0: a Parting Shot read as an escape
            if move and move["power"] and move["target"] == "allAdjacentFoes":
                text += " (hits both foes)"
            elif move and move["power"] and move["target"] == "allAdjacent":
                text += " (hits both foes and our partner)"
            target = choice.get("target")
            names = {e.get("target"): e.get("species") for o in slot.options for e in o.get("target_options") or []
                     if isinstance(e, dict) and e.get("species")}
            if isinstance(target, int) and target != 0:
                text += f" -> {names.get(target) or ('foe' if target > 0 else 'ally')} {abs(target)}"
            parts.append(text)
        elif choice.get("type") == "switch":
            parts.append(f"{who}: switch to {choice.get('species')} (it takes the hits aimed here)")
    return "; ".join(parts) or "pass"


def _our_threats(observation: dict, calc) -> dict:
    """our species -> foe species -> our best estimated hit (from our real moves)."""
    from .actions import move_details

    found = {}
    for number, ours in enumerate(observation.get("active_pokemon") or []):
        if not isinstance(ours, dict):
            continue
        name = ours.get("species") or ours.get("name")
        moves = (observation.get("available_moves") or [[], []])[number] if number < len(observation.get("available_moves") or []) else []
        row = {}
        for foe in calc._theirs:
            foe_name = foe.get("species") or foe.get("name")
            best = None
            for move in moves or []:
                if not isinstance(move, dict):
                    continue
                details = move_details({"move_id": move.get("id")}, number, observation)
                hit = calc.damage(calc.ours(number), foe, {**details, "id": move.get("id")})
                if hit and (best is None or hit["expected_pct"] > best["expected_pct"]):
                    best = {"move": move.get("id"), **hit}
            if best:
                row[foe_name] = best
        if row:
            found[name] = row
    return found


def _turn_summary(prediction: dict, payload: object, note: str) -> str:
    """'Read: <their predicted actions>. Play: <our move>. <note>' for a battle turn."""
    reads = "; ".join(
        f"{foe} {guess.get('action')}" + (f"->{guess['target']}" if guess.get("target") else "")
        for foe, guess in prediction.items()
    )
    parts = [f"Read: {reads}." if reads else "", f"Play: {_describe_payload(payload)}.", note]
    return " ".join(p for p in parts if p)[:SUMMARY_LIMIT]


def _summary(answer: dict) -> str:
    return str(answer.get("reasoning_summary") or "").strip()[:SUMMARY_LIMIT]


def _observation(state: GameState) -> dict:
    observation = state.raw.get("observation")
    return observation if isinstance(observation, dict) else {}


def _is_realtime(state: GameState) -> bool:
    pacing = state.raw.get("pacing")
    return isinstance(pacing, dict) and pacing.get("mode") == "realtime"


def _game_name(context: DecisionContext) -> str:
    game_type = context.game_type or ""
    if game_type == "red_alert":
        return "Red Alert"
    return f"this real-time game ({game_type})" if game_type else "this real-time game"


def _switched_out(payload: Any, turn_rules) -> set[str]:
    """Species ids our payload switches out by choice (not a replacement after a faint)."""
    found = set()
    for slot in (turn_rules.slots if turn_rules else []):
        choice = payload.get(f"slot_{slot.number}") if isinstance(payload, dict) else None
        active = slot.active.get("species") if isinstance(slot.active, dict) else slot.active
        if isinstance(choice, dict) and choice.get("type") == "switch" and active and not slot.force_switch:
            found.add(meta.species_id(str(active)))
    return found


def _ending(state: GameState | None, last_seen: dict | None = None) -> dict | None:
    """How the match ended: the server's final_result (a KO, a forfeit, a
    timeout...) and the HP left on the final board, when it shows one; else
    the board at our last decision (match 666b209d's log ended at a 72%
    Porygon2 with no final board)."""
    if state is None:
        return None
    found = {}
    final = getattr(state, "final_result", None)
    if final:
        found["final_result"] = final
    try:
        board = observed.snapshot(_observation(state))
        if board["ours"] or board["theirs"]:
            found["last_board"] = {side: {m: row["hp"] for m, row in board[side].items()} for side in ("ours", "theirs")}
    except Exception:  # noqa: BLE001 - a summary detail only
        pass
    if "last_board" not in found and isinstance(last_seen, dict) and (last_seen.get("ours") or last_seen.get("theirs")):
        found["board_at_our_last_decision"] = {side: {m: row.get("hp") for m, row in (last_seen.get(side) or {}).items()}
                                               for side in ("ours", "theirs")}
    return found or None


def _priority_bracket(move: dict, species: str, entry: dict, table, calc) -> int:
    """A move's priority for this Pokemon: Prankster status +1, Gale Wings
    Flying at full HP +1, Grassy Glide +1 in Grassy Terrain."""
    from .lookahead import _fraction

    value = move.get("priority") or 0
    abilities = {meta.species_id(a) for a in calc._abilities(entry)} if calc else set()
    if abilities == {"prankster"} and move.get("category") == "STATUS":
        value += 1
    if abilities == {"galewings"} and move.get("type") == "FLYING" and _fraction(entry) >= 0.995:
        value += 1
    if meta.species_id(move.get("name")) == "grassyglide" and getattr(getattr(calc, "_field", None), "terrain", "") == "grassy":
        value += 1
    return value


def _proven_pairs(four: list[str]) -> int:
    """How many proven partner pairs (pick_priority.complements) are in this four."""
    return sum(1 for i, a in enumerate(four) for b in four[i + 1:]
               if pick_priority.complements(a, b) or pick_priority.complements(b, a))


def _revealed(entry: dict) -> list[str]:
    raw = entry.get("moves") or entry.get("known_moves") or entry.get("revealed_moves") or []
    items = raw.keys() if isinstance(raw, dict) else raw if isinstance(raw, list) else []
    return [meta.species_id(m.get("id") or m.get("name") if isinstance(m, dict) else m) for m in items]


def _least_damage(payload: Any, turn_rules, calc) -> dict[str, float]:
    """Foe id -> the least HP fraction our attacks in ``payload`` should take off it."""
    found: dict[str, float] = {}
    if calc is None or turn_rules is None or not isinstance(payload, dict):
        return found
    for slot in turn_rules.slots:
        choice = payload.get(f"slot_{slot.number}")
        move = movelist.lookup(choice.get("move_id")) if isinstance(choice, dict) and choice.get("type") == "move" else None
        if not move or move["category"] not in ("PHYSICAL", "SPECIAL"):
            continue
        spread = move["target"] in ("allAdjacentFoes", "allAdjacent")
        target = choice.get("target")
        foes = calc._theirs if spread else ([calc._theirs[target - 1]] if isinstance(target, int) and 0 < target <= len(calc._theirs) else [])
        data = {"id": choice.get("move_id"), "base_power": move["power"], "type": move["type"], "category": move["category"]}
        for foe in foes:
            hit = calc.damage(calc.ours(slot.number), foe, data, spread=spread) if move["power"] else None
            if hit and not hit.get("immune"):
                key = meta.species_id(foe.get("species") or foe.get("name"))
                found[key] = max(found.get(key, 0.0), hit["range_pct"][0] / 100)
    return found


def _moves_used(payload: Any, turn_rules) -> dict[str, str]:
    """Species id -> the move our payload has it use."""
    found = {}
    for slot in (turn_rules.slots if turn_rules else []):
        choice = payload.get(f"slot_{slot.number}") if isinstance(payload, dict) else None
        active = slot.active.get("species") if isinstance(slot.active, dict) else slot.active
        if isinstance(choice, dict) and choice.get("type") == "move" and active:
            found[meta.species_id(str(active))] = str(choice.get("move_id") or "")
    return found


def _cards_by_id(observation: dict) -> dict[str, dict]:
    return {
        card["card_id"]: card
        for card in observation.get("available_cards") or []
        if isinstance(card, dict) and isinstance(card.get("card_id"), str)
    }


def _card_id(action: LegalAction) -> str | None:
    template = action.input.get("action")
    if isinstance(template, dict) and isinstance(template.get("card_id"), str):
        return template["card_id"]
    prefix, _, suffix = action.action_id.partition(":")
    return suffix or None if prefix == "draft_pick" else None


def _species(card: dict | None) -> str | None:
    if not isinstance(card, dict):
        return None
    species = card.get("species") or card.get("name")
    return species if isinstance(species, str) else None


def _roster_species(observation: dict, context: DecisionContext, *, own: bool) -> list[str]:
    """Species drafted so far, read from the observation's ``rosters`` map."""
    rosters = observation.get("rosters")
    if not isinstance(rosters, dict):
        return []
    seat = str(getattr(context, "agent_id", "") or "")
    mine = rosters.get(seat) if seat in rosters else None
    if mine is None and len(rosters) == 1 and own:
        mine = next(iter(rosters.values()))
    picked = mine if own else [entry for key, value in rosters.items() if key != seat for entry in (value or [])]
    return [e["species"] for e in picked or [] if isinstance(e, dict) and isinstance(e.get("species"), str)]


def _active_species(actives: object) -> list[str]:
    if not isinstance(actives, list):
        return []
    return [
        entry.get("species") or entry.get("name")
        for entry in actives
        if isinstance(entry, dict) and (entry.get("species") or entry.get("name"))
    ]
