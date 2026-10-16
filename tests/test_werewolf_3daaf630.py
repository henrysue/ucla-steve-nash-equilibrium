"""Regression tests from Werewolf match 3daaf630 (2026-10-15), a win as the seer (Player4).

Day 1 the code's fallback claimed seer on a single villager check: "I agree with
Player4, vote Player2" read as an accusation of us. Day 2 "I checked Player3 myself,
so a Player3 vote is a wolf move" was parsed as our check "Player3 = wolf", and
"Player2 was a wolf" (already revealed) as a proved wolf call, so the table showed
"Proven: P3 wolf, P5 villager, P6 villager" (Player6 was the wolf) and later called
us a liar. The strategist voted out Player1, who had voted for the wolf Player2 on
Day 1 and been voted by it, and spent a check on Player3 for sharing a bot with Player2.
"""

from __future__ import annotations

from agent.werewolf.agent import WerewolfAgent
from agent.werewolf.ledger import Ledger
from agent.werewolf.plan import Memory, day_plan, in_danger

from test_werewolf import FakeProvider, _ctx, _msg, _obs, _state

NAMES = {0: "alime588 (Player0, Self-hosted)", 1: "alime588 (Player1, Self-hosted)", 2: "Yuriko Red (Player2, Self-hosted)",
         3: "Yuriko Red (Player3, Self-hosted)", 4: "Henry (Self-hosted)", 5: "Alina Liu (Player5, Self-hosted)",
         6: "Alina Liu (Player6, Self-hosted)"}
D1_VOTES = "Day 1 vote: Player0 -> Player2, Player1 -> Player2, Player2 -> Player1, Player3 -> Player1, " \
           "Player4 -> Player2, Player5 -> abstain, Player6 -> abstain — Player2 eliminated by vote"
D1_TALK = [
    _msg(2, "No hard info yet. To avoid a split vote I'll vote Player0 unless the Seer speaks."),
    _msg(4, "Player2 and Player3, why Player0? Player1 hasn't said anything yet."),
    _msg(1, "I agree with Player4, vote Player2."),
]


def _day2(messages=(), checks="Player0 is villager, Player3 is villager"):
    obs = _obs("SEER", me=4, day=2, living=[1, 3, 4, 5, 6], votes=[D1_VOTES],
               dead=[(1, 2, "wolf", "eliminated by vote"), (2, 0, "villager", "killed at night")],
               extra=f"Your investigations: {checks}")
    return obs


def _ledger_day2(messages):
    ledger = Ledger(me=4)
    ledger.update(_state(_obs("SEER", me=4, extra="Your investigations: Player0 is villager"), me=4, names=NAMES, messages=D1_TALK))
    ledger.update(_state(_day2(), me=4, names=NAMES, messages=messages, first_seq=10))
    return ledger


# -- 1. our seer results are read correctly ---------------------------------------------------


def test_a_player_named_between_check_and_result_is_not_a_wolf_check():
    ledger = _ledger_day2([
        _msg(4, "I'm the seer. Checks: Player0 villager, Player3 villager. Player2 was a wolf. The last wolf is among Player1, 5, 6."),
        _msg(4, "Player1, Player3 is a cleared villager. I checked Player3 myself, so a Player3 vote is a wolf move."),
    ])
    assert ledger.players[4].claimed_checks[3][0] == "villager"
    assert 2 not in ledger.players[4].claimed_checks  # the revealed wolf is the record, not our check
    assert not ledger.caught_lying(4) and not ledger.confirmed_seer(4)
    proven = ledger.proven_roles()
    assert 6 not in proven and 5 not in proven and proven.get(3) != "wolf"
    assert "P3 wolf" not in ledger.render_table() and "LIAR" not in ledger.render_table()


def test_another_seers_checked_with_no_name_between_still_counts():
    ledger = Ledger(me=0)
    ledger.update(_state(_obs("VILLAGER", me=0), me=0, messages=[_msg(2, "I'm the seer. I checked Player5 last night and they are a wolf.")]))
    assert ledger.players[2].claimed_checks == {5: ("wolf", 1)}


def test_a_restated_reveal_is_not_a_check():
    ledger = _ledger_day2([_msg(5, "I'm the seer. Player2 was a wolf and Player1 is a villager.")])
    assert ledger.players[5].claimed_checks == {1: ("villager", 2)}


# -- 2. claim only when really in danger ---------------------------------------------------------


def test_agreeing_with_us_on_a_vote_does_not_accuse_us():
    ledger = Ledger(me=4)
    ledger.update(_state(_obs("SEER", me=4, extra="Your investigations: Player0 is villager"), me=4, names=NAMES, messages=D1_TALK))
    assert 4 not in ledger.players[1].accused and 2 in ledger.players[1].accused
    assert not in_danger(ledger)
    assert day_plan(ledger, Memory()).posture == "seer_hidden"


def test_a_stated_vote_on_us_or_two_accusers_is_danger():
    ledger = Ledger(me=4)
    ledger.update(_state(_obs("SEER", me=4, extra="Your investigations: Player0 is villager"), me=4, names=NAMES,
                         messages=[_msg(3, "I'm voting Player4, the deflection is suspicious.")]))
    assert in_danger(ledger) and day_plan(ledger, Memory()).posture == "seer_claim"
    ledger = Ledger(me=4)
    ledger.update(_state(_obs("SEER", me=4, extra="Your investigations: Player0 is villager"), me=4, names=NAMES,
                         messages=[_msg(3, "Player4 is suspicious."), _msg(6, "Player4 feels off to me, shady.")]))
    assert in_danger(ledger)


# -- 3 and 4. votes beat chat; same bot is not same team -------------------------------------------


def test_the_vote_record_clears_player1():
    ledger = _ledger_day2([])
    clears = ledger.vote_clears(1)
    assert "voted for wolf Player2 on D1" in clears and "wolf Player2 voted for them on D1" in clears
    assert ledger.vote_clears(5) == []


def test_the_vote_and_check_prompts_carry_the_vote_record():
    model = FakeProvider({"guesses": [], "strategy": "", "action_id": "1", "reason": "r", "why_not_code": ""})
    agent = WerewolfAgent(model, log_dir="")
    agent.ledger.update(_state(_obs("SEER", me=4, extra="Your investigations: Player0 is villager"), me=4, names=NAMES, messages=D1_TALK))
    obs = _day2()
    agent.ledger.update(_state(obs, me=4, names=NAMES, first_seq=10))
    legal = [(str(s), f"Vote to eliminate Player{s}") for s in (1, 3, 5, 6)] + [("7", "Abstain")]
    agent.choose_action(_state(obs, me=4, phase="moving", legal=legal, names=NAMES), _ctx(4))
    assert "Vote record: Player1 voted for wolf Player2 on D1" in model.prompts[-1]
    night = _obs("SEER", me=4, day=2, phase="night_seer", living=[1, 3, 4, 5, 6], votes=[D1_VOTES],
                 dead=[(1, 2, "wolf", "eliminated by vote")], extra="Your investigations: Player0 is villager")
    legal = [(str(s), f"Investigate Player{s}") for s in (1, 3, 5, 6)]
    agent.choose_action(_state(night, me=4, phase="moving", legal=legal, names=NAMES), _ctx(4))
    assert "Player2, Player3 are the same bot" in model.prompts[-1]
