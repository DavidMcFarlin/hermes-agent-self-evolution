"""Tests for the triage session-DB query (single aggregate + schema config)."""

import sqlite3
from pathlib import Path

import pytest

from evolution.core.config import EvolutionConfig
from evolution.monitor.triage import PerformanceTriage, _safe_identifier


def _make_db(path: Path):
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE messages (session_id TEXT, content TEXT)")
    rows = []
    # session A: short (3 msgs), mentions the skill once
    rows += [("A", "let's use github-code-review here")] + [("A", "ok")] * 2
    # session B: long (20 msgs), mentions the skill once -> weak failure signal
    rows += [("B", "trying github-code-review")] + [("B", "retry")] * 19
    # session C: never mentions the skill
    rows += [("C", "unrelated chatter")] * 5
    conn.executemany("INSERT INTO messages VALUES (?, ?)", rows)
    conn.commit()
    conn.close()


def _triage_for(tmp_path: Path, config=None) -> PerformanceTriage:
    db = tmp_path / "state.db"
    _make_db(db)
    triage = PerformanceTriage(config or EvolutionConfig())
    triage.session_db = db
    # Isolate from any real ~/.hermes data on the host.
    triage.session_dir = tmp_path / "no-sessions"
    triage.error_log = tmp_path / "no-errors.log"
    return triage


def test_usage_counts_distinct_sessions(tmp_path):
    triage = _triage_for(tmp_path)
    stats = triage.mine_usage_stats("github-code-review", "skill")
    # Sessions A and B reference the skill; C does not.
    assert stats["usage"] == 2


def test_long_session_flagged_as_weak_failure(tmp_path):
    triage = _triage_for(tmp_path)
    stats = triage.mine_usage_stats("github-code-review", "skill")
    # Only session B exceeds the 15-message threshold.
    assert stats["potential_failures"] == 1


def test_configurable_threshold_changes_failure_count(tmp_path):
    cfg = EvolutionConfig()
    cfg.triage_long_session_threshold = 2  # now both A (3) and B (20) qualify
    triage = _triage_for(tmp_path, cfg)
    stats = triage.mine_usage_stats("github-code-review", "skill")
    assert stats["potential_failures"] == 2


def test_unsafe_identifier_is_rejected(tmp_path):
    cfg = EvolutionConfig()
    cfg.session_db_content_column = "content; DROP TABLE messages"
    triage = _triage_for(tmp_path, cfg)
    # Bad identifier -> ValueError caught internally -> usage stays 0, no crash.
    stats = triage.mine_usage_stats("github-code-review", "skill")
    assert stats["usage"] == 0
    # Table must survive the injection attempt.
    conn = sqlite3.connect(str(triage.session_db))
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] > 0
    conn.close()


def test_safe_identifier_helper():
    assert _safe_identifier("session_id", "column") == "session_id"
    for bad in ("a b", "1col", "x;y", ""):
        with pytest.raises(ValueError):
            _safe_identifier(bad, "column")
