"""The strategist: one match, three kinds of decision.

- Draft pick: every card on the board is scored by draft.py (team plan,
  roles, coverage, matchup with the opponent's picks, strength); the model
  picks from the best few. No model: the top score.
- Team Preview: ONE model call writes the match plan and chooses the four
  and the leads, checked against the lineup rules (re-asked once if it
  breaks them). No model: the best-scoring legal four.
- Battle turn, in three steps:
    1. rules.py decides what is withheld and what a blunder is;
    2. predict.py guesses each foe's action (it carries Protect and did not
       Protect last turn -> it will probably Protect; otherwise it attacks the
       one of ours it hurts most), and the model refines that guess;
    3. lookahead.ranked_lines plays our best legal answer to the prediction.
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
from . import archetypes, cards as card_sets, decisions, draft, lookahead, meta, movelist, observed, predict, protect, rules
from .style import OpponentStyle
from .moverules import MoveTracker
from .providers import NO_CREDENTIALS, StrategistProvider
from .report import Analysis, Decision, MatchLog
from .typechart import effectiveness

# Platform clocks (GAMES.md): 15 s a draft pick, 90 s Team Preview, 55 s a
# battle decision from a shared bank. We answer well inside each.
DRAFT_SECONDS = 9.0
TEAM_PREVIEW_SECONDS = 78.0
DECISION_SECONDS = 40.0
MIN_ATTEMPT_SECONDS = 4.0  # less time than this left: play the fallback now
MAX_ATTEMPTS = 2  # the first answer plus one retry carrying the reason
RETRY_BUDGET_SHARE = 0.5  # a retry only starts with half the budget left
FASTER_EFFORT = {"max": "high", "xhigh": "high", "high": "medium", "medium": "low", "low": "low", "none": "none"}

SUMMARY_LIMIT = 280
# The platform shows reasoning_summary to spectators (and maybe the
# opponent), so by default it carries no strategy: the model's note stays in
# the local log. `--show-reasoning` / VGC_PUBLIC_REASONING=1 sends the real
# one-line reasoning instead. The model's private thinking is never sent.
PUBLIC_SUMMARY = "Move submitted."
PUBLIC_REASONING_ENV = "VGC_PUBLIC_REASONING"

MIN_BRING_ATTACKERS = 3
RECENT_TURNS = 3  # turns of history in each battle prompt
DRAFT_CARDS_SHOWN = 12
_DRAFT_HIDDEN = ("draft_points", "usage_pct", "tier")  # usage anchors kept out of the draft prompt


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
        self._public_reasoning = str(os.environ.get(PUBLIC_REASONING_ENV, "")).strip().lower() in ("1", "true", "yes", "on")
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
            known_moves=lambda s: self._meta.common_moves(s), offense=lambda s: self._meta.offense(s)
        )
        self._draft_plan = ""  # the model's one-line team plan, carried pick to pick
        self._draft_target: dict = {}
        self._game_plan: dict | None = None
        self._recent: collections.deque = collections.deque(maxlen=RECENT_TURNS)
        self._turn_rules: rules.TurnRules | None = None
        self._turn_odds: dict[str, dict] = {}
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
        self._turn = 0
        self.model_calls = 0
        self._last_attempts, self._last_seconds, self._last_reasoning = 0, 0.0, None
        self._opponent_seen: list[str] = []
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
        fallback = offered[0][0]
        if self._provider is None:
            return self._fallback(fallback, "No model configured: took the best-fitting card.", turn=turn, phase=state.phase, kind="draft_pick")

        offered = offered[:DRAFT_CARDS_SHOWN]
        decision = decisions.draft_decision(
            [action.action_id for action, _ in offered],
            [{**self._describe_card(action, card_id, cards), "fit": fit.get(card_id)} for action, card_id in offered],
        ).with_prompt({
            "your_roster_so_far": mine,
            "opponent_roster_so_far": theirs,
            "draft_target": self._draft_target or "(no team plan fits this board)",
            "opponent_playstyle": styles or "(not readable yet)",
            "your_team_plan": self._draft_plan or "(first pick: choose the win condition to build around)",
        })
        answer = self._ask(decision, state, context, DRAFT_SECONDS)
        chosen = next((a for a, _ in offered if answer and a.action_id == answer.get("action_id")), None)
        if chosen is None:
            reason = "Out of time on the draft clock" if answer is None else "The model named a card not on the board"
            return self._fallback(fallback, f"{reason}; took the best-fitting card.", turn=turn, phase=state.phase, kind="draft_pick")
        self._draft_plan = str(answer.get("team_plan") or self._draft_plan).strip()[:SUMMARY_LIMIT]
        return self._record(chosen, _summary(answer), turn=turn, phase=state.phase, kind="draft_pick")

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

    def _describe_card(self, action: LegalAction, card_id: str | None, cards: dict) -> dict:
        card = dict(cards.get(card_id) or {})
        card["action_id"] = action.action_id
        card["roles"] = [k for k, v in self._roles(_species(card), card.get("moves")).items() if v]
        reference = self._meta.brief(_species(card))
        if reference:
            card["reference"] = {k: v for k, v in reference.items() if k not in _DRAFT_HIDDEN}
        return card

    # -- team preview: one call, plan + lineup -------------------------------------------

    def _team_preview(self, action: LegalAction, state: GameState, context: DecisionContext):
        roster = act.parse_lineup(action)
        observation = _observation(state)
        self._meet_opponent(observation, context)
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
            "bring a Tailwind or Trick Room setter if the roster has one",
            "do not lead two Pokémon where one's spread move (Earthquake, Surf...) hits the other",
            "at most one restricted legendary or Mega in the four",
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
            problems.append(f"no Tailwind/Trick Room setter ({', '.join(setters)} left home)")
        clash = self._lead_clash(leads)
        if clash:
            problems.append(clash)
        if len(self._meta.restricted_in(bring)) > 1:
            problems.append("more than one restricted legendary or Mega")
        return problems

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
        """Can set Tailwind or Trick Room: the drafted card's moves, else usage
        data, else (no data at all) what it can learn."""
        card = self._meta.card_set(species)
        moves = card["moves"] if card else None
        if moves:
            return bool({meta.species_id(m) for m in moves} & {"tailwind", "trickroom"})
        known = self._meta.common_moves(species) or self._meta.field(species, "role")
        return bool({"tailwind", "trick_room"} & archetypes.tags(species, self._meta, learnable=not known))

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
            return max(fitting, key=lambda four: sum(self._meta.draft_points(s) or 0 for s in four))
        ranked = sorted(roster, key=lambda s: -(self._meta.draft_points(s) or 0))
        bring = [s for s in ranked if not self._meta.is_restricted(s)][:4]
        return bring + [s for s in ranked if s not in bring][: 4 - len(bring)]

    # -- battle turn: rules -> predict the opponent -> best legal answer ----------------

    def _doubles_turn(self, action: LegalAction, state: GameState, context: DecisionContext):
        self._turn += 1
        turn = f"Turn {self._turn}"
        slots = act.parse_doubles(action)
        observation = _observation(state)
        self._remember_opponent(observation)

        # 1. Rules: what cannot work or would be a blunder is withheld/rejected.
        turn_rules = rules.build(slots, observation, self._moves, self._meta, self._log)
        self._turn_rules = turn_rules
        self._observe(observation)
        if self._recent and self._moves.protected_last_turn:
            self._recent[-1]["they_protected"] = self._moves.protected_last_turn
        self._style.resolve(self._happened)  # what last turn's Protect choices say about them
        self._turn_odds = self._styled_odds(observation, self._protect_odds(observation))
        calc = lookahead.Calculator(observation, self._meta)
        self._calc = calc
        ours = _active_species(observation.get("active_pokemon"))
        foes = _active_species(observation.get("opponent_active_pokemon"))

        # 2. The code's guess at the opponent; the move it implies is the fallback.
        guess = _safely(lambda: predict.code_guess(observation, calc, self._moves, self._meta, self._turn_odds), {})
        fallback = self._answer(turn_rules, observation, calc, guess, ours, state)

        if self._provider is None or not any(o["type"] == "move" for slot in slots for o in slot.options):
            reason = "No model configured" if self._provider is None else "A replacement after a faint"
            return self._play(fallback, turn, state, guess, "code", guess, "", fallback_reason=f"{reason}: played the best answer to the code's prediction." if self._provider is None else None)

        # 2b. The model reads the opponent; its prediction wins when it is usable.
        decision = decisions.predict_decision(foes, ours, self._prediction_facts(observation, calc, guess))
        asked = self._ask(decision, state, context, DECISION_SECONDS,
                          parse=lambda answer: predict.from_model(answer, foes, ours))
        if asked is None:
            return self._play(fallback, turn, state, guess, "code", guess, "",
                              fallback_reason="No usable model prediction in time; played the best answer to the code's prediction.")
        answer, from_model = asked
        prediction = {**guess, **from_model}  # any foe the model skipped keeps the code's guess
        # 3. Our best legal answer to that prediction.
        line = self._answer(turn_rules, observation, calc, prediction, ours, state)
        return self._play(line, turn, state, prediction, "model", guess, str(answer.get("state_summary") or ""),
                          summary=_summary(answer))

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

    def _answer(self, turn_rules: rules.TurnRules, observation: dict, calc, prediction: dict, ours: list[str], state: GameState):
        """Step 3: ranked legal answers to ``prediction``; the first is played."""
        try:
            expected = predict.expectations(prediction, ours)
            ranked = lookahead.ranked_lines(turn_rules, observation, calc, expected)
            if ranked:
                return [(value, turn_rules.check(answer)) for value, answer in ranked]
        except Exception as exc:  # noqa: BLE001 - never cost the move
            self._log(f"[vgc] move selection failed [{type(exc).__name__}]: {exc}")
        return [(0.0, act.guaranteed_action(state))]

    def _play(self, ranked: list, turn: str, state: GameState, prediction: dict, source: str, guess: dict,
              state_summary: str, *, summary: str = "", fallback_reason: str | None = None):
        analysis = Analysis(
            turn=turn, state_summary=state_summary, prediction=prediction, source=source, code_guess=guess,
            lines=[(value, _describe_payload(payload)) for value, payload in ranked],
        )
        summary = _turn_summary(prediction, ranked[0][1], summary or (fallback_reason or ""))
        slots = self._turn_rules.slots if self._turn_rules else []
        if any(o.get("type") == "move" for slot in slots for o in slot.options):
            self._last_prediction = prediction  # a full turn: check it against what happens
        return self._record(ranked[0][1], summary, turn=turn, phase=state.phase,
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
            "opponent_style": self._style.summary(),
            "speeds": {n: self._meta.base_speed(n) for n in _active_species(observation.get("active_pokemon")) + theirs},
            "code_guess": guess,
            "game_plan": self._game_plan or "(none)",
            "recent_turns": list(self._recent),
        }

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

    def _styled_odds(self, observation: dict, odds: dict[str, dict]) -> dict[str, dict]:
        """The by-the-book Protect odds re-read through this opponent's style."""
        try:
            carries = {}
            for entry in observation.get("opponent_active_pokemon") or []:
                if isinstance(entry, dict) and (entry.get("species") or entry.get("name")):
                    name = entry.get("species") or entry.get("name")
                    known = {meta.species_id(m) for m in self._meta.common_moves(name)} | set(_revealed(entry))
                    carries[meta.species_id(name)] = bool(known & predict.PROTECT_CHAIN)
            chained = {foe: True for foe in self._moves.protected_last_turn}
            return self._style.open(odds, carries, chained, self._turn)
        except Exception as exc:  # noqa: BLE001 - advice only; never cost the move
            self._log(f"[vgc] opponent style unavailable this turn [{type(exc).__name__}]: {exc}")
            return odds

    def _protect_odds(self, observation: dict) -> dict[str, dict]:
        try:
            return protect.odds(
                observation, first_turn=self._moves.opponent_first_turn(),
                common_moves=self._meta.common_moves, card_set=self._meta.card_set,
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
                                        ending=_ending(state), style={**self._style.summary(), "notes": self._style.notes,
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


def _ending(state: GameState | None) -> dict | None:
    """How the match ended: the server's final_result (a KO, a forfeit, a
    timeout...) and the HP left on the final board, when it shows one."""
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
    return found or None


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
