"""The Werewolf agent (agent/werewolf/): table, plan, model moves, guard rails."""

from __future__ import annotations

from altruagent import TERMINATE_MESSAGING, WAIT, SendMessage
from altruagent.models import DecisionContext, GameState

from agent import dispatch
from agent.werewolf.agent import WerewolfAgent, _trim
from agent.werewolf.ledger import Ledger
from agent.werewolf.plan import Memory, choose_kill, day_plan, kill_briefing, villager_vote, wolf_vote


def _obs(role, *, me, day=1, phase="day_vote", living=range(7), dead=(), votes=(), extra=""):
    lines = ["=== Werewolf ===", f"Your role: {role}", extra or "You do not know anyone else's role."]
    lines.append(f"Day {day} | phase {phase} | alive {len(list(living))}/7")
    lines.append("Living: " + ", ".join(f"Player{s}" for s in living))
    lines.append("Dead:")
    lines += [f"  day {d}: Player{s} was {r} ({c})" for d, s, r, c in dead]
    lines += list(votes)
    return "\n".join(lines)


def _state(observation, *, me, messages=(), phase="messaging", legal=(), terminated=(), seconds_left=100,
           first_seq=0, names=None):
    return GameState.from_mcp_state(
        {
            "session_id": "s1",
            "game_type": "werewolf",
            "observation": observation,
            "phase": phase,
            "your_position": me,
            "players": [{"position": i, "name": (names or {}).get(i, f"Bot{i}")} for i in range(7)],
            "new_messages": [
                {"seq": first_seq + i, "sender": s, "recipients": list(r), "content": t, "type": "chat"}
                for i, (s, t, r) in enumerate(messages)
            ],
            "terminated_messaging": list(terminated),
            "messaging_seconds_left": seconds_left if phase == "messaging" else None,
            "is_current_actor": bool(legal),
            "legal_actions": {"state_version": 3, "actions": [
                {"action_id": a, "label": lab, "input": {"action_id": a}} for a, lab in legal
            ]} if legal else None,
            "state_version": 3,
        }
    )


def _ctx(me):
    return DecisionContext(session_id="s1", tournament_id=None, game_type="werewolf", agent_id="a", seat_position=me)


def _msg(sender, text, to=()):
    return (sender, text, to)


class FakeProvider:
    def __init__(self, answer):
        self.answer = answer
        self.prompts = []

    def complete_structured(self, messages, schema_name, schema, *, timeout, effort="medium"):
        self.prompts.append(messages[-1]["content"])
        return dict(self.answer)


# -- ledger -------------------------------------------------------------------


def test_ledger_reads_roles_deaths_and_votes():
    ledger = Ledger(me=3)
    ledger.update(_state(_obs(
        "WOLF", me=3, day=2, living=[0, 2, 3, 4, 6],
        dead=[(1, 1, "villager", "eliminated by vote"), (2, 5, "seer", "killed at night")],
        votes=["Day 1 vote: Player0 -> Player1, Player1 -> abstain, Player2 -> Player1, Player3 -> Player1 — Player1 eliminated by vote"],
        extra="Your fellow wolves: Player4",
    ), me=3))
    assert ledger.role == "wolf" and ledger.allies == {4}
    assert ledger.living() == [0, 2, 3, 4, 6]
    assert ledger.players[5].revealed_role == "seer" and ledger.real_seer_dead()
    assert ledger.players[0].votes == {1: 1} and ledger.players[1].votes == {1: None}
    assert ledger.vote_results == {1: 1}
    assert ledger.players[0].wrong_calls == {1}


def test_ledger_reads_claims_checks_and_accusations():
    ledger = Ledger(me=0)
    ledger.update(_state(_obs("VILLAGER", me=0), me=0, messages=[
        _msg(2, "I'm the seer. Night 1 I checked Player5: wolf. Vote Player5."),
        _msg(5, "Player2 is lying, that's a wolf move."),
        _msg(4, "I trust Player2, they seem clean."),
    ]))
    p2, p5, p4 = ledger.players[2], ledger.players[5], ledger.players[4]
    assert p2.claimed_role == "seer" and p2.claimed_checks == {5: ("wolf", 1)}
    assert 5 in p2.accused and 2 in p5.accused and 2 in p4.defended
    assert ledger.heat_today(5) == 1
    assert "Player2" not in ledger.render_table() or "| P2 |" in ledger.render_table()


def test_talking_about_a_seer_claim_is_not_a_claim():
    ledger = Ledger(me=0)
    ledger.update(_state(_obs("VILLAGER", me=0), me=0, messages=[
        _msg(3, "Player2's seer claim is proven false. Fake seer = wolf."),
        _msg(4, "I'm not the seer, just a villager."),
        _msg(6, "As the seer, I checked Player1: villager."),
    ]))
    assert ledger.players[3].claimed_role is None
    assert ledger.players[4].claimed_role is None or ledger.players[4].claimed_role == "villager"
    assert ledger.players[6].claimed_role == "seer" and ledger.players[6].claimed_checks == {1: ("villager", 1)}


def test_correct_wolf_callers_are_tracked_after_a_reveal():
    ledger = Ledger(me=0)
    ledger.update(_state(_obs("VILLAGER", me=0), me=0, messages=[_msg(2, "Player5 is suspicious, vote Player5."), _msg(3, "Player1 is the wolf.")]))
    ledger.update(_state(_obs(
        "VILLAGER", me=0, day=2, living=[0, 1, 2, 3, 4, 6],
        dead=[(1, 5, "wolf", "eliminated by vote")],
        votes=["Day 1 vote: Player2 -> Player5, Player4 -> Player5, Player3 -> Player1 — Player5 eliminated by vote"],
    ), me=0))
    assert ledger.players[2].correct_calls == {5} and ledger.players[4].correct_calls == {5}
    callers = ledger.render_callers()
    assert "Revealed wolf P5: called correctly by P2, P4" in callers
    assert ledger.wolf_probability()[2] < ledger.wolf_probability()[3]


# -- wolf plan ----------------------------------------------------------------


def _wolf_ledger(day, messages=(), living=range(7), dead=()):
    ledger = Ledger(me=3)
    ledger.update(_state(_obs("WOLF", me=3, day=day, living=list(living), dead=dead, extra="Your fellow wolves: Player4"), me=3, messages=messages))
    return ledger


def test_wolf_blends_in_on_day_one_and_whispers_the_plan():
    ledger, memory = _wolf_ledger(1), Memory()
    plan = day_plan(ledger, memory)
    assert plan.posture == "blend"
    assert plan.lines[0].recipients == [4] and "Kill Player" in plan.lines[0].fallback
    assert not memory.fake_checks


def test_wolf_fakes_the_seer_on_day_two_with_a_true_clean_check_first():
    ledger = _wolf_ledger(2, living=[0, 1, 2, 3, 4, 6], dead=[(2, 5, "villager", "killed at night")])
    memory = Memory()
    plan = day_plan(ledger, memory)
    assert plan.posture == "fake_seer"
    night1, night2 = memory.fake_checks[1], memory.fake_checks[2]
    assert night1[1] == "villager" and night1[0] not in (3, 4)  # a real villager, reported truthfully
    assert night2[1] == "wolf" and night2[0] not in (3, 4, night1[0])
    public = [t for t in plan.lines if not t.recipients]
    assert "seer" in public[0].fallback and f"Player{night1[0]}" in public[0].fallback
    assert f"Player{night2[0]}: WOLF" in public[1].fallback
    assert "claiming seer" in plan.lines[0].fallback  # the ally is told to back it


def test_only_one_wolf_claims():
    ledger = _wolf_ledger(2, messages=[_msg(4, "I'm claiming seer, back me.", (3,))])
    plan = day_plan(ledger, Memory())
    assert plan.posture != "fake_seer"


def test_wolf_counter_claims_when_the_real_seer_calls_it_out():
    ledger = _wolf_ledger(1, messages=[_msg(1, "I'm the seer, I checked Player3: wolf.")])
    memory = Memory()
    plan = day_plan(ledger, memory)
    assert plan.posture == "fake_seer"
    assert memory.fake_checks == {1: (1, "wolf")}
    assert wolf_vote(ledger, memory) in (0, 1, 2, 5, 6)


def test_wolf_claim_syncs_from_what_was_actually_said():
    ledger = _wolf_ledger(2, messages=[_msg(3, "I'm the seer. Night 1 I checked Player0: villager. Night 2 I checked Player6: wolf.")])
    memory = Memory()
    day_plan(ledger, memory)
    assert memory.fake_claim_day == 2 and memory.frame == 6
    assert wolf_vote(ledger, memory) == 6


def test_wolf_kills_the_real_seer_first():
    ledger = _wolf_ledger(1, messages=[_msg(6, "I am the seer. Player2 is villager."), _msg(0, "Player3 is suspicious.")])
    assert choose_kill(ledger, Memory(), ["0", "1", "2", "5", "6"]) == 6


# -- agent: model moves and guard rails -----------------------------------------


def test_agent_whispers_then_lets_the_model_talk():
    model = FakeProvider({"guesses": [{"seat": 0, "p_wolf": 0.1, "why": "quiet"}], "strategy": "blend", "action": "chat", "to": "everyone", "message": "Who started the Player1 wagon?"})
    agent = WerewolfAgent(model, log_dir="")
    obs = _obs("WOLF", me=3, extra="Your fellow wolves: Player4")
    first = agent.choose_message(_state(obs, me=3), _ctx(3))
    assert isinstance(first, SendMessage) and first.recipients == [4]
    second = agent.choose_message(_state(obs, me=3), _ctx(3))
    assert isinstance(second, SendMessage) and second.content == "Who started the Player1 wagon?"
    assert "PLAYER TABLE" in model.prompts[0] and agent.guesses[0][0] == 0.1


def test_a_public_line_that_outs_the_wolf_falls_back_to_code():
    model = FakeProvider({"guesses": [], "strategy": "", "action": "chat", "to": "everyone", "message": "My fellow wolf and I agree."})
    agent = WerewolfAgent(model, log_dir="")
    agent.memory.sent[1] = ["dm"]
    reply = agent.choose_message(_state(_obs("WOLF", me=3, extra="Your fellow wolves: Player4"), me=3), _ctx(3))
    assert isinstance(reply, SendMessage) and "fellow wolf" not in reply.content
    assert len(reply.content.split()) <= 50


def test_agent_ends_discussion_once_terminated_or_out_of_time():
    agent = WerewolfAgent(None, use_model=False, log_dir="")
    obs = _obs("VILLAGER", me=0)
    assert agent.choose_message(_state(obs, me=0, terminated=[0]), _ctx(0)) is TERMINATE_MESSAGING
    first = agent.choose_message(_state(obs, me=0), _ctx(0))
    assert isinstance(first, SendMessage)
    # Nothing new: keep listening while there's time, end it near the close.
    assert agent.choose_message(_state(obs, me=0), _ctx(0)) is WAIT
    assert agent.choose_message(_state(obs, me=0, seconds_left=10), _ctx(0)) is TERMINATE_MESSAGING
    assert agent.choose_message(_state(obs, me=0, seconds_left=None), _ctx(0)) is TERMINATE_MESSAGING


def test_model_makes_the_vote_but_illegal_or_abstain_falls_back():
    legal = [(str(s), f"Vote to eliminate Player{s}") for s in (1, 2, 3, 4, 5, 6)] + [("7", "Abstain")]
    obs = _obs("VILLAGER", me=0)
    picked = WerewolfAgent(FakeProvider({"guesses": [], "strategy": "", "action_id": "5", "reason": "r"}), log_dir="")
    assert picked.choose_action(_state(obs, me=0, phase="moving", legal=legal), _ctx(0)).action_id == "5"
    abstain = WerewolfAgent(FakeProvider({"guesses": [], "strategy": "", "action_id": "7", "reason": "r"}), log_dir="")
    assert abstain.choose_action(_state(obs, me=0, phase="moving", legal=legal), _ctx(0)).action_id != "7"


def test_code_alone_plays_every_phase():
    agent = WerewolfAgent(None, use_model=False, log_dir="")
    kill = [(str(s), f"Kill Player{s}") for s in (0, 1, 2, 5, 6)]
    obs = _obs("WOLF", me=3, phase="night_wolf", extra="Your fellow wolves: Player4")
    assert agent.choose_action(_state(obs, me=3, phase="moving", legal=kill), _ctx(3)).action_id in {"0", "1", "2", "5", "6"}
    seer = WerewolfAgent(None, use_model=False, log_dir="")
    look = [(str(s), f"Investigate Player{s}") for s in (0, 1, 2, 3, 4, 6)]
    obs = _obs("SEER", me=5, phase="night_seer", extra="You have not investigated anyone yet.")
    assert seer.choose_action(_state(obs, me=5, phase="moving", legal=look), _ctx(5)).action_id in {"0", "1", "2", "3", "4", "6"}


def test_match_log_holds_the_table(tmp_path):
    agent = WerewolfAgent(None, use_model=False, log_dir=str(tmp_path))
    legal = [(str(s), f"Vote to eliminate Player{s}") for s in (1, 2)] + [("7", "Abstain")]
    agent.choose_action(_state(_obs("VILLAGER", me=0), me=0, phase="moving", legal=legal), _ctx(0))
    text = next(tmp_path.glob("*werewolf*seat0.md")).read_text(encoding="utf-8")
    assert "| Seat | Name | Status | Claim | Votes |" in text


def test_dispatch_routes_werewolf():
    assert dispatch._build("werewolf").__class__.__name__ == "WerewolfAgent"


# -- regressions from real games --------------------------------------------------
#
# 438990c2: P2 seer, P5+P6 wolves. P6 fake-claimed seer on day 1 after being
# accused and survived the vote; P6 went out day 2 (wolf); the wolves killed
# P2 night 3 (revealed seer). Day 3 left P3, P4, P5: P2's checks prove P4 a
# villager, so P5 is the wolf. Both our villagers voted each other instead.

_G1_D1 = "Day 1 vote: Player0 -> Player1, Player1 -> Player6, Player2 -> Player6, Player3 -> Player1, Player4 -> Player1, Player5 -> Player6, Player6 -> Player1 — Player1 eliminated by vote"
_G1_D2 = "Day 2 vote: Player2 -> Player6, Player3 -> Player6, Player4 -> Player6, Player5 -> Player3, Player6 -> Player3 — Player6 eliminated by vote"


def _game1_day3(me):
    ledger = Ledger(me=me)
    ledger.update(_state(_obs("VILLAGER", me=me), me=me, messages=[
        _msg(6, "I'll claim now, since a lynch on me would waste the seer. I'm the seer. Night 1 I checked Player2 and they are a villager. Vote Player1 today."),
    ]))
    ledger.update(_state(_obs("VILLAGER", me=me, day=2, living=[2, 3, 4, 5, 6],
                              dead=[(1, 1, "villager", "eliminated by vote"), (2, 0, "villager", "killed at night")],
                              votes=[_G1_D1]), me=me, first_seq=10, messages=[
        _msg(2, "I'm the real seer. Checks: Player0 villager, Player4 villager. Player6 counter-claimed only after being pressured. Lynch Player6 today."),
        _msg(6, "Night 2 I checked Player3: WOLF. Night 1 was Player2, a villager. Everyone vote Player3 today."),
    ]))
    ledger.update(_state(_obs("VILLAGER", me=me, day=3, living=[3, 4, 5],
                              dead=[(1, 1, "villager", "eliminated by vote"), (2, 0, "villager", "killed at night"),
                                    (2, 6, "wolf", "eliminated by vote"), (3, 2, "seer", "killed at night")],
                              votes=[_G1_D1, _G1_D2]), me=me, first_seq=20))
    return ledger


def test_game1_a_revealed_seers_checks_are_facts_so_day3_is_a_deduction():
    ledger = _game1_day3(me=3)
    assert ledger.players[2].claimed_checks[4] == ("villager", 2)  # "Checks: Player0 villager, ..." parsed
    assert ledger.proven_roles()[4] == "villager"
    probs = ledger.wolf_probability()
    assert probs[5] == 1.0 and probs[4] == 0.0
    assert villager_vote(ledger) == 5
    assert "P4 villager" in ledger.render_table()


def test_game1_the_partner_is_who_voted_with_the_wolf_the_day_it_went_out():
    ledger = _game1_day3(me=4)  # P4 is cleared, but from its own seat P3 vs P5 is open
    probs = ledger.wolf_probability()
    assert probs[5] > probs[3]
    assert any("voted with wolf P6 on P3 the day P6 went out" in note for note in ledger.explain(5))


def test_the_seer_knows_any_other_seer_claim_is_a_wolf():
    ledger = Ledger(me=2)
    ledger.update(_state(_obs("SEER", me=2, extra="Your investigations: Player0 is villager"), me=2, messages=[
        _msg(6, "I'm the seer. Night 1 I checked Player2 and they are a villager."),
    ]))
    assert ledger.wolf_probability()[6] == 1.0


# b4bdb262: P0 seer, P4+P6 wolves (ours), P0-P2 one other agent (alime588).
# Night 3, P4 alone with P2, P3, P5: the village's top suspect was P2, yet
# P4 killed P2 and lost. Day 2, our villagers voted P1 off a "copy-paste"
# theory about two seats of the same bot, overriding the code's P4.

_G2_NAMES = {i: f"alime588 (Player{i}, Self-hosted)" for i in range(3)} | {i: f"Henry (Player{i}, Self-hosted)" for i in range(3, 7)}
_G2_D1 = "Day 1 vote: Player0 -> Player6, Player1 -> Player6, Player2 -> Player6, Player3 -> Player6, Player4 -> Player6, Player5 -> Player6, Player6 -> Player0 — Player6 eliminated by vote"
_G2_DAY2_TALK = [
    _msg(1, "As a villager, I'm confidently voting for Player4. They've openly attacked the seer claim, a red flag."),
    _msg(2, "I'm a villager and confident in my vote for Player4. They directly attacked the seer claim, a red flag."),
]


def _game2_day2(me, role="VILLAGER", extra=""):
    ledger = Ledger(me=me)
    ledger.update(_state(_obs(role, me=me, extra=extra), me=me, names=_G2_NAMES, messages=[
        _msg(0, "I'm the seer. Night 1 I checked Player6: wolf. Vote Player6."),
        _msg(6, "Player0 is lying. I'm the real seer. Night 1 I checked Player0 and got WOLF."),
    ]))
    ledger.update(_state(_obs(role, me=me, day=2, living=[1, 2, 3, 4, 5], extra=extra,
                              dead=[(1, 6, "wolf", "eliminated by vote"), (2, 0, "seer", "killed at night")],
                              votes=[_G2_D1]), me=me, names=_G2_NAMES, first_seq=10, messages=_G2_DAY2_TALK))
    return ledger


def test_game2_same_agent_seats_are_flagged_and_the_code_still_reads_p4():
    ledger = _game2_day2(me=3)
    assert ledger.same_agent_groups()[0] == [0, 1, 2]
    assert "Same agent: P0, P1, P2 are all 'alime588'" in ledger.render_table()
    assert villager_vote(ledger) == 4


def test_game2_endgame_kill_keeps_the_village_suspect_alive():
    extra = "Your fellow wolves: Player6"
    ledger = _game2_day2(me=4, role="WOLF", extra=extra)
    d2 = "Day 2 vote: Player1 -> Player4, Player2 -> Player1, Player3 -> Player1, Player4 -> Player1, Player5 -> Player1 — Player1 eliminated by vote"
    ledger.update(_state(_obs("WOLF", me=4, day=3, phase="night_wolf", living=[2, 3, 4, 5], extra=extra,
                              dead=[(1, 6, "wolf", "eliminated by vote"), (2, 0, "seer", "killed at night"),
                                    (2, 1, "villager", "eliminated by vote")],
                              votes=[_G2_D1, d2]), me=4, names=_G2_NAMES, phase="moving", first_seq=20))
    village_view = {2: 0.40, 3: 0.20, 4: 0.35, 5: 0.15}  # the model's read before the kill
    kill = choose_kill(ledger, Memory(), ["2", "3", "5"], village_view)
    assert kill in (3, 5)
    assert any("Keep Player2 ALIVE" in note for note in kill_briefing(ledger, Memory(), ["2", "3", "5"], village_view))


def test_a_wolf_line_hinting_at_private_orders_is_blocked():
    model = FakeProvider({"guesses": [], "strategy": "", "action": "chat", "to": "everyone",
                          "message": "I held back because I was told not to claim early."})
    agent = WerewolfAgent(model, log_dir="")
    agent.memory.sent[1] = ["dm"]
    reply = agent.choose_message(_state(_obs("WOLF", me=3, extra="Your fellow wolves: Player4"), me=3), _ctx(3))
    assert isinstance(reply, SendMessage) and "told" not in reply.content


def test_long_chats_are_cut_at_a_sentence_end():
    text = "Player1 and Player2 posted copy-paste accusations, and that is the strongest pattern today by far. " * 3 + "I'm voting Player4."
    out = _trim(text)
    assert len(out.split()) <= 50 and out.endswith(".")


def test_vote_prompt_carries_our_public_vote_and_the_codes_reasons():
    model = FakeProvider({"guesses": [], "strategy": "", "action_id": "4", "reason": "r", "why_not_code": ""})
    agent = WerewolfAgent(model, log_dir="")
    legal = [(str(s), f"Vote to eliminate Player{s}") for s in (1, 2, 4, 5)] + [("7", "Abstain")]
    obs = _obs("VILLAGER", me=3, day=2, living=[1, 2, 3, 4, 5],
               dead=[(1, 6, "wolf", "eliminated by vote"), (2, 0, "seer", "killed at night")], votes=[_G2_D1])
    agent.ledger.update(_state(obs, me=3, names=_G2_NAMES, messages=[*_G2_DAY2_TALK, _msg(3, "I'm voting Player1. Don't split.")]))
    agent.choose_action(_state(obs, me=3, phase="moving", legal=legal, names=_G2_NAMES), _ctx(3))
    prompt = model.prompts[-1]
    assert "You said in public today that you would vote Player1" in prompt
    assert "Code's reasons for Player" in prompt and "why_not_code" in prompt
