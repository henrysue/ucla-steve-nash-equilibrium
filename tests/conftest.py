"""Test-wide settings."""

import pytest


@pytest.fixture(autouse=True)
def _no_match_logs_from_tests(monkeypatch):
    """Tests must never write into ./matchlogs.

    The project's .env sets VGC_LOG_DIR, and code under test calls
    load_dotenv(), which would otherwise turn match logging on for every
    test that follows. An empty value wins: load_dotenv never overrides a
    variable that is already set, and the match log treats "" as "off".
    Tests that want a log pass log_dir explicitly.
    """
    monkeypatch.setenv("VGC_LOG_DIR", "")
    monkeypatch.setenv("REDALERT_LOG_DIR", "")
    # Same for public reasoning: off unless a test turns it on, and restored
    # afterwards even when the code under test (the CLI flag) sets it itself.
    monkeypatch.setenv("VGC_PUBLIC_REASONING", "")
    # And the draft-card catalog: tests never read or write data/card_sets.json.
    monkeypatch.setenv("VGC_CARD_SETS", "")
    # And per-opponent style memory (data/opponent_styles.json).
    monkeypatch.setenv("VGC_OPPONENT_STYLES", "")
    # And the Red Alert self-play sparring archetypes (agent/redalert/sparring.py):
    # off, so tests never write marker files or turn a seat into an archetype.
    monkeypatch.setenv("RA_TEST_RUSH", "0")
    monkeypatch.setenv("RA_SPARRING", "")
    # And the post-game analyst never starts a background process from a test.
    monkeypatch.setenv("REDALERT_ANALYST_DETACH", "0")
    monkeypatch.setenv("REDALERT_OPPONENT", "")
    # And the per-seat models-off switch (A/B tests in self-play).
    monkeypatch.setenv("REDALERT_MODELS_OFF_SEAT", "")
