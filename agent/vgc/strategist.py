"""The strategist: one match's per-turn workflow.

Section 8 of ``vgc-draft-strategist.md``, in order, every turn:

1. update the information state (the server's observation plus what the
   opponent has done so far this match);
2. name the threats and win conditions;
3. enumerate and prune both sides' actions;
4. estimate payoffs as win probabilities;
5. draw the payoff matrix, and a tree when the call is close;
6. compute the Nash equilibrium — our mix, theirs, and the game value;
7. check the opponent model against the cheese criteria;
8. recommend an action labelled EQUILIBRIUM or EXPLOIT;
9. log what the opponent actually did so the model updates.

Steps 3 and 4 need judgment, so they go to the model; everything else is
deterministic code here and in ``nash.py``/``opponent.py``. That split is
deliberate: the model is good at "which four lines matter and roughly what
are they worth", and bad at solving a linear program or keeping an honest
Dirichlet posterior. It also means a model that fails, stalls, or answers
nonsense costs us estimate quality, never a legal move — the clock is hard
(55 s a decision out of a 7-minute bank, 15 s a draft pick) and a lost turn
is worse than a merely decent one.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any, Callable

from altruagent import DecisionContext, GameState, LegalAction, WithReasoning
from altruagent.console import print_line

from . import actions as act
from . import decisions, meta, nash, speed
from .opponent import (
    CATEGORIES,
    RISK_BUDGET,
    CheeseCheck,
    OpponentModel,
    check_cheese,
    fold_to_categories,
    unfold_to_columns,
)
from .providers import NO_CREDENTIALS, StrategistProvider
from .report import Analysis, Decision, MatchLog

# Clocks (GAMES.md): 90 s at Team Preview, 55 s a battle decision out of a
# 420 s bank, 15 s a draft pick. We answer well inside them — the runtime
# still has to read the state and send the move, and a turn that eats the
# bank loses the battle later even when it wins the turn.
# Budgets against the platform's limits in GAMES.md: 90s for Team Preview,
# 55s per battle decision from a shared bank, 15s per draft pick. Battle
# decisions stay under their ceiling on purpose — the bank is shared
# across the game. Measured: a doubles turn takes 10-14s at low effort and
# ~40s at medium, so 40s fits low comfortably and medium only barely.
TEAM_PREVIEW_SECONDS = 78.0
DECISION_SECONDS = 40.0
DRAFT_SECONDS = 9.0
MIN_ATTEMPT_SECONDS = 4.0  # less time than this left: play the fallback now
MAX_ATTEMPTS = 2  # the first answer plus one retry carrying the reason
# A timeout means "too slow", so the retry trades depth for speed.
FASTER_EFFORT = {"max": "high", "xhigh": "high", "high": "medium", "medium": "low", "low": "low"}

SUMMARY_LIMIT = 280


class VGCDraftStrategist:
    """One match's agent. ``create_agent()`` builds a fresh one per match, so
    the opponent model never leaks between games of a set."""

    def __init__(
        self,
        provider: StrategistProvider | None,
        *,
        # print() writes the text and the newline separately, which lets two
        # concurrent match workers glue their lines together. The analysis is
        # dozens of lines a turn, so this matters here.
        log: Callable[[str], None] = print_line,
        rng: random.Random | None = None,
        clock: Callable[[], float] = time.monotonic,
        log_dir: str | None = None,
    ) -> None:
        self._provider = provider
        self._log = log
        # Unseeded by default: an equilibrium mix is only unexploitable if
        # the draws are genuinely unpredictable.
        self._rng = rng or random.Random()
        self._clock = clock
        self._matchlog = MatchLog(log, log_dir)
        self._opponent = OpponentModel()
        self._meta = meta.table()
        self._last_situation: str | None = None
        self._turn = 0
        self.model_calls = 0
        # Per-decision audit facts, set by _ask and read by _record.
        self._last_attempts = 0
        self._last_seconds = 0.0
        self._last_reasoning: str | None = None
        # Tracked for the end-of-match summary: the opponent species we have
        # seen and the archetype they matched, so the offline review knows
        # what we faced without re-reading every turn.
        self._opponent_seen: list[str] = []
        self._opponent_archetype: str | None = None
        self._agent_id: str | None = None

        # Said once, at startup, where it can be acted on — the per-turn
        # note is a public reasoning_summary and has to stay short.
        if provider is None:
            self._log(f"[vgc] {NO_CREDENTIALS}")
        else:
            self._log(f"[vgc] model: {provider!r}")
        if not len(self._meta):
            self._log(
                "[vgc] No draft reference table loaded: drafting and Team Preview lose their "
                "usage, draft-point, counter and partner data. Expected "
                f"{meta.DEFAULT_CSV}, or set VGC_META_CSV."
            )

    # -- entry point ----------------------------------------------------------

    def choose_action(self, state: GameState, context: DecisionContext):
        if _is_realtime(state):
            # Said plainly here rather than left to the runner, whose
            # equivalent message is inferred from a placeholder agent
            # indexing an empty legal_actions — which this one never does.
            raise act.TemplateError(
                f"Your agent can't play {_game_name(context)}: the VGC Draft strategist is a "
                "Pokémon specialist, and a move in a real-time game is a batch of orders it has "
                "no way to write. See GAMES.md, or run with --agent examples.llm_agent, which "
                "plays all three games."
            )
        self._agent_id = str(getattr(context, "agent_id", "") or "") or self._agent_id
        if getattr(state, "is_terminal", False):
            # The runner normally stops asking once a match ends, but if a
            # terminal state ever reaches us, emit the summary now (with the
            # real outcome from final_result) rather than relying on it.
            self.summarize_match(state)
            raise act.TemplateError("choose_action was called on a terminal state")
        if not state.legal_actions:
            raise act.TemplateError("choose_action was called with no legal actions")
        self._matchlog.name_match(context.session_id, context.seat_position, context.game_type)
        first = state.legal_actions[0]
        if first.action_id == "submit_team":
            raise act.TemplateError(
                "submit_team (build-your-own-team modes) needs a full 6-Pokémon team; this "
                "strategist plays pokemon_vgc_doubles_draft."
            )

        # Hard rule #1: a failure anywhere below must cost estimate quality,
        # never a legal move. The deliberation path prepares its own
        # heuristic fallback and plays it on a timeout or an unusable answer,
        # but the preparation itself — and the Nash/cheese arithmetic — can
        # still raise on a malformed template or a degenerate matrix, and an
        # exception escaping choose_action forfeits the whole match. This
        # guard is the floor beneath every other fallback: on any unexpected
        # failure it commits to a guaranteed-legal move instead of raising.
        try:
            if first.action_id == "select_lineup":
                return self._team_preview(first, state, context)
            if first.action_id == "doubles_turn":
                return self._doubles_turn(first, state, context)
            return self._draft_pick(state, context)
        except act.TemplateError:
            # The server offered something this agent cannot read at all;
            # that is a genuine misconfiguration, not a per-turn hiccup, and
            # the runner's own message is clearer than any move we could
            # invent. Let it surface.
            raise
        except Exception as exc:  # noqa: BLE001 - deliberately broad; see above
            label = type(exc).__name__
            self._log(f"[vgc] decision failed before a move was chosen [{label}]: {exc}")
            return self._last_resort(state, exc)

    # -- draft ----------------------------------------------------------------

    def _draft_pick(self, state: GameState, context: DecisionContext):
        observation = _observation(state)
        cards = _cards_by_id(observation)
        offered = [(action, _card_id(action)) for action in state.legal_actions]
        have_restricted = bool(
            self._meta.restricted_in(_roster_species(observation, context, own=True))
        )

        def value(pair) -> tuple[int, int]:
            species = _species(cards.get(pair[1]))
            # A second restricted can only be brought if the league allows
            # two, so rank it behind everything else rather than above it.
            spare = 0 if have_restricted and self._meta.is_restricted(species) else 1
            return (spare, self._meta.draft_points(species) or 0)

        ranked = sorted(offered, key=value, reverse=True)
        fallback = ranked[0][0] if ranked else state.legal_actions[0]

        if self._provider is None:
            return self._fallback(
                fallback, "No model configured, so the highest-value card on the draft board was taken.",
                turn=f"Draft pick {observation.get('pick_number', self._matchlog.decisions + 1)}",
                phase=state.phase, kind="draft_pick",
            )

        described = [
            self._describe_card(action, card_id, cards)
            for action, card_id in offered
        ]
        decision = decisions.draft_decision([action.action_id for action, _ in offered], described)
        mine = _roster_species(observation, context, own=True)
        decision = _with_prompt(decision, {
            "your_roster_so_far": mine,
            "your_restricted_legendaries_so_far": self._meta.restricted_in(mine),
            "opponent_roster_so_far": _roster_species(observation, context, own=False),
            "reference_table_note": (
                "draft_points / usage / tier / restricted come from a reference table for this "
                "format and are prior knowledge only — each card's real moves, item and ability "
                "are in available_cards and are authoritative."
            ),
        })

        turn = f"Draft pick {observation.get('pick_number', self._matchlog.decisions + 1)}"
        answer = self._ask(decision, state, context, DRAFT_SECONDS)
        if answer is None:
            return self._fallback(
                fallback, "Out of time on the draft clock; took the highest-value card available.",
                turn=turn, phase=state.phase, kind="draft_pick",
            )
        chosen = next((action for action, _ in offered if action.action_id == answer.get("action_id")), None)
        if chosen is None:  # the enum should prevent this
            return self._fallback(
                fallback, "The model named a card that was not on the board.",
                turn=turn, phase=state.phase, kind="draft_pick",
            )
        return self._record(chosen, _summary(answer), turn=turn, phase=state.phase, kind="draft_pick")

    def _describe_card(self, action: LegalAction, card_id: str | None, cards: dict) -> dict:
        card = dict(cards.get(card_id) or {})
        card["action_id"] = action.action_id
        reference = self._meta.brief(_species(card))
        if reference:
            card["reference"] = reference
        return card

    # -- team preview ---------------------------------------------------------

    def _team_preview(self, action: LegalAction, state: GameState, context: DecisionContext):
        roster = act.parse_lineup(action)
        observation = _observation(state)
        opponent = [
            entry.get("species") or entry.get("name")
            for entry in observation.get("opponent_roster") or []
            if isinstance(entry, dict) and (entry.get("species") or entry.get("name"))
        ]
        # Prepare the fallback bring first (hard rule #1). A torn roster that
        # defeats the value heuristic drops to the first four distinct
        # species, which is still a legal lineup.
        try:
            bring = self._best_four(roster)
            fallback = act.build_lineup(roster, bring, bring[:2])
        except Exception as exc:  # noqa: BLE001
            self._log(f"[vgc] lineup fallback unavailable, using a guaranteed-legal lineup [{type(exc).__name__}]: {exc}")
            fallback = act.guaranteed_action(state)

        if self._provider is None:
            return self._fallback(
                fallback, "No model configured, so the four highest-value Pokémon were brought.",
                turn="Team Preview", phase=state.phase, kind="select_lineup",
            )

        decision = decisions.lineup_decision(roster, observation.get("opponent_roster"))
        preview_reference = {
            "your_restricted_legendaries": self._meta.restricted_in(roster),
            "opponent_restricted_legendaries": self._meta.restricted_in(opponent),
            "reference_your_roster": self._meta.briefs(roster, battle=True),
            "reference_opponent_roster": self._meta.briefs(opponent, battle=True),
            "reference_matchups": self._meta.matchup_notes(roster, opponent),
            "reference_partners": self._meta.partner_notes(roster),
            "reference_table_note": (
                "The reference_* fields are prior knowledge from a format table: roles, named "
                "counters, partners, the Worlds Top 8 teams each appeared on, and — when the "
                "opponent's roster matches one — a known team archetype with its structural, "
                "crit-independent counterplay. Useful for judging matchups, but the rosters and "
                "the server's own data are authoritative about this match."
            ),
        }
        archetypes = self._meta.archetype_notes(opponent)
        if archetypes:
            preview_reference["reference_opponent_archetypes"] = archetypes
        decision = _with_prompt(decision, preview_reference)

        return self._matrix_decision(
            decision, state, context, TEAM_PREVIEW_SECONDS,
            turn="Team Preview",
            build_row=lambda option: act.build_lineup(roster, option.get("bring"), option.get("leads")),
            fallback=fallback, use_opponent_model=False,
            fallback_summary="Out of time at Team Preview; brought the four highest-value Pokémon.",
        )

    # -- doubles turn ---------------------------------------------------------

    def _doubles_turn(self, action: LegalAction, state: GameState, context: DecisionContext):
        self._turn += 1
        slots = act.parse_doubles(action)
        # Prepare the most logical move FIRST, with no deliberation, so a
        # slow or broken model costs analysis quality and never a turn. If
        # even the type-aware heuristic cannot read this template, drop to
        # the guaranteed-legal floor rather than losing the whole turn.
        try:
            fallback = act.heuristic_doubles(
                slots, state, base_speed=self._meta.base_speed, species_types=self._meta.species_types
            )
        except Exception as exc:  # noqa: BLE001
            self._log(f"[vgc] heuristic fallback unavailable, using a guaranteed-legal move [{type(exc).__name__}]: {exc}")
            fallback = act.guaranteed_action(state)

        if self._provider is None:
            return self._fallback(
                fallback, "No model configured; played the best type-effective line available.",
                turn=f"Turn {self._turn}", phase=state.phase, kind="doubles_turn",
            )

        observation = _observation(state)
        template = action.input["action"]
        decision = decisions.doubles_decision(
            act.describe_slots(slots, state),
            template.get("target_legend"),
            template.get("instructions"),
            self._memory(),
        )
        decision = _with_prompt(decision, self._battle_reference(observation))

        return self._matrix_decision(
            decision, state, context, DECISION_SECONDS,
            turn=f"Turn {self._turn}",
            build_row=lambda option: act.build_doubles(slots, option),
            fallback=fallback, use_opponent_model=True,
            fallback_summary="Out of time on the turn clock; played the best type-effective line available.",
        )

    def _best_four(self, roster: list[str]) -> list[str]:
        """The no-model bring: highest draft value, at most one restricted.

        The asymmetry decides it. If the league allows a second restricted
        we give up a little value; if it caps them at one — as VGC 2025
        Reg I does — an illegal lineup costs the match. Without a model to
        reason about the ruleset, take the safe side.
        """
        ranked = sorted(roster, key=lambda species: -(self._meta.draft_points(species) or 0))
        bring, restricted_taken = [], False
        for species in ranked:
            if len(bring) == 4:
                break
            if self._meta.is_restricted(species):
                if restricted_taken:
                    continue
                restricted_taken = True
            bring.append(species)
        # A roster of nothing but restricteds still has to field four.
        for species in ranked:
            if len(bring) == 4:
                break
            if species not in bring:
                bring.append(species)
        return bring

    def _battle_reference(self, observation: dict) -> dict:
        ours = _active_species(observation.get("active_pokemon"))
        theirs = _active_species(observation.get("opponent_active_pokemon"))
        reference = {
            "reference_our_actives": self._meta.briefs(ours, battle=True),
            "reference_their_actives": self._meta.briefs(theirs, battle=True),
            "reference_table_note": (
                "reference_* is prior knowledge from a format table — common items, common and "
                "learnable moves, named counters. Use it to put distributions over what you have "
                "NOT been shown (their unrevealed moves, item, Tera); the observation is "
                "authoritative about everything you have been shown."
            ),
        }
        matchups = self._meta.matchup_notes(ours, theirs)
        if any(matchups.values()):
            reference["reference_matchups"] = matchups
        # Deterministic base-speed turn-order prior (speed.py). Computed once
        # here so the model does not re-derive it every turn; labelled as a
        # base-speed prior so unseen items/boosts are the model's job, not
        # ours.
        order = speed.turn_order(observation, self._meta.base_speed)
        if order is not None:
            reference["reference_speed_order"] = order
        # Remember what we faced, for the end-of-match summary.
        self._remember_opponent(self._opponent_species_seen(observation))
        # Team-level archetype read from everything the opponent has shown so
        # far (actives plus any revealed back line), not just this turn's
        # actives, so a core is still recognized after it rotates out.
        archetypes = self._meta.archetype_notes(self._opponent_species_seen(observation))
        if archetypes:
            reference["reference_opponent_archetypes"] = archetypes
            self._opponent_archetype = archetypes[0].get("name") or self._opponent_archetype
            reference["reference_archetype_note"] = (
                "Prior knowledge: the opponent's revealed Pokémon match this known Reg I "
                "archetype. crit_independent_counterplay lists STRUCTURAL answers (type, "
                "immunity, redirection, protection, speed control, weather) that work on a "
                "normal damage roll with no critical hit. It is a hint for building the matrix, "
                "not an instruction and not a substitute for the live observation."
            )
        return reference

    def _opponent_species_seen(self, observation: dict) -> list[str]:
        """Every opponent species the observation reveals: actives first, then
        any named back-line entries, de-duplicated in that order."""
        seen = list(_active_species(observation.get("opponent_active_pokemon")))
        for entry in observation.get("opponent_roster") or []:
            if isinstance(entry, dict):
                name = entry.get("species") or entry.get("name")
                if isinstance(name, str) and name:
                    seen.append(name)
        return list(dict.fromkeys(seen))

    def _remember_opponent(self, species: list[str]) -> None:
        """Accumulate the opponent species we have seen, in first-seen order."""
        for name in species:
            if name and name not in self._opponent_seen:
                self._opponent_seen.append(name)

    def summarize_match(self, state: GameState | None = None, *, note: str | None = None) -> dict:
        """Write the end-of-match summary for offline, no-time-pressure review.

        Public so a runner hook, a test, or a script can call it when a match
        ends. The outcome is read ONLY from ``state.final_result`` when that
        is present (``winner_agent_id`` compared to our seat); it is never
        guessed, so an unknown outcome is recorded as ``None`` rather than a
        made-up win or loss. Returns the summary dict and also writes it to
        the match log's ``.summary.json`` / markdown footer when a log dir is
        configured.
        """
        outcome = self._outcome_from(state)
        return self._matchlog.summarize(
            outcome=outcome,
            opponent=self._opponent_seen,
            archetype=self._opponent_archetype,
            note=note,
        )

    def _outcome_from(self, state: GameState | None) -> str | None:
        """win/loss/None from the server's final_result, never guessed."""
        if state is None:
            return None
        final = getattr(state, "final_result", None)
        if not isinstance(final, dict):
            return None
        winner = final.get("winner_agent_id")
        if winner is None:
            # Some games report a per-player numeric return instead.
            returns = getattr(state, "returns", None)
            if isinstance(returns, dict):
                mine = returns.get(str(getattr(self, "_agent_id", "")) or "")
                if isinstance(mine, (int, float)) and not isinstance(mine, bool):
                    return "win" if mine > 0 else "loss" if mine < 0 else "draw"
            return None
        return "win" if str(winner) == str(self._agent_id or "") else "loss"

    def _memory(self) -> dict:
        """What this match has taught us, for the model's own reasoning."""
        return {
            "turn": self._turn,
            "observed_opponent_actions_by_situation": self._opponent.counts or "nothing observed yet",
            "categories": list(CATEGORIES),
        }

    # -- the shared matrix workflow -------------------------------------------

    def _matrix_decision(
        self,
        decision: decisions.Decision,
        state: GameState,
        context: DecisionContext,
        budget: float,
        *,
        turn: str,
        build_row: Callable[[dict], Any],
        fallback: Any,
        use_opponent_model: bool,
        fallback_summary: str,
    ):
        asked = self._ask(
            decision, state, context, budget,
            parse=lambda answer: _parse_matrix(answer, build_row, self._log),
        )
        if asked is None:
            return self._fallback(
                fallback, fallback_summary, turn=turn, phase=state.phase, kind=decision.kind
            )
        answer, parsed = asked
        matrix = parsed.matrix
        equilibrium = nash.solve_zero_sum(matrix)

        cheese: CheeseCheck | None = None
        exploitation = None
        predicted = None
        mix = equilibrium.row_mix

        if use_opponent_model:
            self._record_observation(answer)
            situation = str(answer.get("situation") or "even")
            self._last_situation = situation
            prior = fold_to_categories(equilibrium.col_mix, parsed.categories)
            posterior = self._opponent.posterior(situation, prior)
            predicted = unfold_to_columns(posterior, parsed.categories)
            budget_used = _risk_budget(equilibrium.value)
            exploitation = nash.safe_exploit(matrix, predicted, equilibrium, risk_budget=budget_used)
            cheese = check_cheese(
                situation=situation,
                observations=self._opponent.observations(situation),
                posterior=posterior,
                equilibrium_prior=prior,
                game_value=equilibrium.value,
                exploit_value=exploitation.value_vs_prediction,
                exploit_worst_case=exploitation.worst_case,
                risk_budget=budget_used,
            )
            if cheese.exploit:
                mix = exploitation.mix

        chosen = nash.sample(mix, self._rng)
        analysis = Analysis(
            turn=turn,
            state_summary=str(answer.get("state_summary") or ""),
            threats=str(answer.get("threats") or ""),
            row_labels=parsed.labels,
            col_labels=parsed.col_labels,
            matrix=matrix,
            equilibrium=equilibrium,
            chosen_row=chosen,
            recommendation=cheese.verdict if cheese else "EQUILIBRIUM",
            cheese=cheese,
            exploitation=exploitation,
            predicted_col_mix=predicted,
            key_assumption=str(answer.get("key_assumption") or ""),
            pruned=str(answer.get("pruned") or ""),
        )
        return self._record(
            parsed.payloads[chosen], _summary(answer),
            turn=turn, phase=state.phase, kind=decision.kind, analysis=analysis,
        )

    def _record_observation(self, answer: dict) -> None:
        """Step 9: attribute the opponent's last action to the situation it
        was taken in, which is the *previous* turn's situation."""
        category = str(answer.get("opponent_last_action") or "none")
        if category == "none" or self._last_situation is None:
            return
        try:
            self._opponent.observe(self._last_situation, category)
        except ValueError as exc:  # an out-of-enum category; drop it rather than skew the model
            self._log(f"[vgc] ignored an unusable opponent observation: {exc}")

    # -- model plumbing -------------------------------------------------------

    def _ask(
        self,
        decision: decisions.Decision,
        state: GameState,
        context: DecisionContext,
        budget: float,
        *,
        parse: Callable[[dict], Any] | None = None,
    ):
        """One decision's model call, retried once, inside ``budget`` seconds.

        ``parse`` runs inside the retry loop, so *every* way an answer can be
        unusable — a bad option index, an illegal target, a payoff matrix of
        the wrong shape — earns the one retry rather than raising. Returns
        ``None`` when no valid answer arrived in time, which is the caller's
        signal to play its deterministic fallback.
        """
        assert self._provider is not None
        messages = decision.messages(state, context.seat_position, context.game_type)
        started = self._clock()
        deadline = started + budget
        self._last_attempts, self._last_seconds, self._last_reasoning = 0, 0.0, None

        effort = decision.effort
        for attempt in range(1, MAX_ATTEMPTS + 1):
            left = deadline - self._clock()
            if left < MIN_ATTEMPT_SECONDS:
                self._last_seconds = self._clock() - started
                self._log(f"[vgc] {decision.kind}: out of time for another model call")
                return None
            self.model_calls += 1
            self._last_attempts = attempt
            try:
                answer = self._provider.complete_structured(
                    messages, decision.kind, decision.schema, timeout=left, effort=effort
                )
                self._last_seconds = self._clock() - started
                self._last_reasoning = getattr(self._provider, "last_reasoning", None)
                return answer if parse is None else (answer, parse(answer))
            except Exception as exc:
                # Deliberately broad. A provider is third-party code on a
                # clock, and this agent's one hard promise is that a model
                # problem costs estimate quality, never a legal move: an
                # escaping exception forfeits the whole match. The type is
                # logged so a real bug here is still visible, not silent.
                label = type(exc).__name__
                self._log(f"[vgc] {decision.kind}: unusable answer (attempt {attempt}/{MAX_ATTEMPTS}) [{label}]: {exc}")
                if "Timeout" in str(exc):
                    # Too slow, not wrong: retry with less thinking and
                    # without a "your answer was invalid" nudge.
                    effort = FASTER_EFFORT.get(effort, effort)
                    continue
                messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            f"That answer was invalid: {exc}. Answer again using only the options "
                            "given, and keep every payoff a win probability between 0 and 1."
                        ),
                    }
                ]
        self._last_seconds = self._clock() - started
        return None

    def _record(
        self,
        payload: Any,
        summary: str,
        *,
        turn: str,
        phase: object,
        kind: str,
        analysis: Analysis | None = None,
        fallback_reason: str | None = None,
    ):
        """Audit one decision, then hand it to the runtime.

        ``model_reasoning`` is the model's private deliberation and goes to
        the local log only — never into ``summary``, which the platform
        shows to spectators.
        """
        self._matchlog.write(Decision(
            turn=turn,
            phase=str(phase or ""),
            kind=kind,
            payload=payload,
            public_summary=summary,
            model=getattr(self._provider, "model", None),
            model_reasoning=self._last_reasoning,
            attempts=self._last_attempts,
            seconds=self._last_seconds,
            fallback_reason=fallback_reason,
            analysis=analysis,
        ))
        return WithReasoning(payload, summary[:SUMMARY_LIMIT])

    def _fallback(self, payload: Any, reason: str, *, turn: str, phase: object, kind: str):
        return self._record(
            payload, reason, turn=turn, phase=phase, kind=kind, fallback_reason=reason
        )

    def _last_resort(self, state: GameState, exc: Exception):
        """Play a guaranteed-legal move after an unexpected failure.

        Reached only when a phase handler raised past its own heuristic
        fallback — a torn template the heuristic could not read, or a bug in
        the deliberation code. ``act.guaranteed_action`` is written not to
        raise, but it is wrapped anyway: the one promise this method makes is
        that it returns a payload rather than letting the match forfeit.
        """
        reason = f"Recovered from an unexpected {type(exc).__name__} with a guaranteed-legal move: {exc}"
        try:
            payload = act.guaranteed_action(state)
        except Exception as inner:  # noqa: BLE001  # pragma: no cover - guaranteed_action is written not to raise
            payload = {"type": "doubles_turn", "slot_0": dict(act.PASS), "slot_1": dict(act.PASS)}
            reason = f"{reason} (even the guaranteed action raised {type(inner).__name__}: {inner})"
        kind = "doubles_turn" if isinstance(payload, dict) and payload.get("type") == "doubles_turn" else "select_lineup"
        return self._record(
            payload, reason, turn=f"Turn {self._turn}" if kind == "doubles_turn" else "Team Preview",
            phase=state.phase, kind=kind, fallback_reason=reason,
        )


# -- helpers ------------------------------------------------------------------


def _risk_budget(game_value: float) -> float:
    """Section 6.3: widen the risk budget when behind, narrow it when ahead.

    The spec frames this as series context, which a per-match process cannot
    see, so the game value stands in for it: a losing position is where a
    read is worth the variance.
    """
    if game_value < 0.40:
        return RISK_BUDGET + 0.05
    if game_value > 0.60:
        return RISK_BUDGET - 0.03
    return RISK_BUDGET


@dataclass(frozen=True)
class _Parsed:
    """A model answer validated down to playable payloads and a clean matrix."""

    labels: list[str]
    payloads: list[Any]
    col_labels: list[str]
    categories: list[str]
    matrix: list[list[float]]


def _parse_matrix(answer: dict, build_row: Callable[[dict], Any], log: Callable[[str], None]) -> _Parsed:
    """Validate a whole matrix answer, raising ``InvalidAnswer`` on anything unusable."""
    labels, payloads = [], []
    for option in _options(answer, "our_options"):
        labels.append(str(option.get("label") or "").strip())
        payloads.append(build_row(option))  # raises if the line is not legal

    columns = _options(answer, "opponent_options")
    return _Parsed(
        labels=labels,
        payloads=payloads,
        col_labels=[str(column.get("label") or "").strip() for column in columns],
        categories=[str(column.get("category") or "attack") for column in columns],
        matrix=_matrix(answer, len(payloads), len(columns), log),
    )


def _matrix(answer: dict, rows: int, cols: int, log: Callable[[str], None]) -> list[list[float]]:
    """Validate the payoff matrix against the row and column counts."""
    raw = answer.get("payoffs")
    if not isinstance(raw, list) or len(raw) != rows:
        raise act.InvalidAnswer(f"payoffs must have exactly {rows} rows, one per entry in our_options")
    matrix, clamped = [], False
    for index, row in enumerate(raw):
        if not isinstance(row, list) or len(row) != cols:
            raise act.InvalidAnswer(
                f"payoffs row {index} must have exactly {cols} cells, one per entry in opponent_options"
            )
        built = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise act.InvalidAnswer(f"payoffs row {index} has a non-numeric cell: {value!r}")
            if not 0.0 <= value <= 1.0:
                clamped = True
            built.append(min(1.0, max(0.0, float(value))))
        matrix.append(built)
    if clamped:
        log("[vgc] a payoff fell outside 0..1 and was clamped; win probabilities only")
    return matrix


def _options(answer: dict, key: str) -> list[dict]:
    options = answer.get(key)
    if not isinstance(options, list) or not options:
        raise act.InvalidAnswer(f"{key} must be a non-empty list")
    if len(options) > decisions.MAX_COLS and key == "opponent_options":
        raise act.InvalidAnswer(f"{key} must have at most {decisions.MAX_COLS} entries")
    if len(options) > decisions.MAX_ROWS and key == "our_options":
        raise act.InvalidAnswer(f"{key} must have at most {decisions.MAX_ROWS} entries")
    if not all(isinstance(option, dict) for option in options):
        raise act.InvalidAnswer(f"every entry in {key} must be an object")
    return options


def _with_prompt(decision: decisions.Decision, extra: dict) -> decisions.Decision:
    return decisions.Decision(
        kind=decision.kind,
        prompt={**decision.prompt, **extra},
        schema=decision.schema,
        effort=decision.effort,
    )


def _summary(answer: dict) -> str:
    return str(answer.get("reasoning_summary") or "").strip()[:SUMMARY_LIMIT]


def _observation(state: GameState) -> dict:
    observation = state.raw.get("observation")
    return observation if isinstance(observation, dict) else {}


def _is_realtime(state: GameState) -> bool:
    """A game with no turns, where a move is a batch of orders (Red Alert)."""
    pacing = state.raw.get("pacing")
    return isinstance(pacing, dict) and pacing.get("mode") == "realtime"


def _game_name(context: DecisionContext) -> str:
    game_type = context.game_type or ""
    if game_type == "red_alert":
        return "Red Alert"
    return f"this real-time game ({game_type})" if game_type else "this real-time game"


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
    picked = mine if own else [
        entry for key, value in rosters.items() if key != seat for entry in (value or [])
    ]
    return [
        entry["species"]
        for entry in picked or []
        if isinstance(entry, dict) and isinstance(entry.get("species"), str)
    ]


def _active_species(actives: object) -> list[str]:
    if not isinstance(actives, list):
        return []
    return [
        entry["species"] if entry.get("species") else entry["name"]
        for entry in actives
        if isinstance(entry, dict) and (entry.get("species") or entry.get("name"))
    ]
