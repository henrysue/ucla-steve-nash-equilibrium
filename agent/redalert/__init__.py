"""A hyper-aggressive Red Alert agent.

Hybrid design (see PLAN.md): a deterministic executor plays routine orders on
every view with no model call, and a throttled, off-the-hot-path "brain" sets a
high-level Mode the executor obeys. Tuned to win the 20-minute tiebreak
(``kills_cost - deaths_cost``, then ``assets_value``) rather than a long game:
cheap units that trade up, relentless pressure, and never stalling.

``create_agent()`` in ``agent.py`` builds one ``RedAlertAgent`` per match.
"""
