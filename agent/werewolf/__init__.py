"""A Werewolf agent (simplified 7-player ruleset: 2 wolves, 1 seer, 4 villagers).

    from agent.werewolf import create_agent

Modules:

- ``ledger``  - the player table: statements, votes, claims, accusations, suspicion, wolf-callers
- ``plan``    - deterministic posture and moves per role (fake-seer script, wolf coordination, votes)
- ``prompts`` - the model's rules, role strategies and answer schemas
- ``agent``   - Claude makes each move from that context; code checks it and falls back
"""

from .agent import WerewolfAgent, create_agent

__all__ = ["WerewolfAgent", "create_agent"]
