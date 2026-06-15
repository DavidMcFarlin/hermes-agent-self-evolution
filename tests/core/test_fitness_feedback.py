"""Tests for GEPA-aware fitness metrics (score + feedback)."""

from types import SimpleNamespace

import dspy

from evolution.core.fitness import (
    SkillJudge,
    make_gepa_metric,
    skill_fitness_metric,
    skill_overlap_feedback,
)


def _example(expected="flag security risk", task="review this code"):
    return SimpleNamespace(task_input=task, expected_behavior=expected, skill_text="skill body")


def test_gepa_metric_returns_score_and_feedback():
    ex = _example()
    pred = SimpleNamespace(output="this looks fine, nothing to mention")
    metric = make_gepa_metric(skill_fitness_metric, skill_overlap_feedback)
    result = metric(ex, pred)
    # GEPA reads .score and .feedback off the prediction
    assert hasattr(result, "score") and hasattr(result, "feedback")
    assert 0.0 <= result.score <= 1.0
    assert "missing" in result.feedback.lower()


def test_gepa_metric_accepts_full_five_arg_signature():
    ex = _example()
    pred = SimpleNamespace(output="risk flagged")
    metric = make_gepa_metric(skill_fitness_metric, skill_overlap_feedback)
    # GEPA calls metric(gold, pred, trace, pred_name, pred_trace)
    result = metric(ex, pred, None, "predictor", None)
    assert 0.0 <= result.score <= 1.0


def test_overlap_feedback_flags_empty_output():
    ex = _example()
    fb = skill_overlap_feedback(ex, SimpleNamespace(output=""))
    assert "empty" in fb.lower()


def test_overlap_feedback_reports_full_coverage():
    ex = _example(expected="security risk")
    fb = skill_overlap_feedback(ex, SimpleNamespace(output="security risk found here"))
    assert "covered all" in fb.lower()


def _raising_judge():
    """Build a SkillJudge whose underlying judge call raises (no network)."""
    judge = SkillJudge(dspy.LM("openai/gpt-4.1-mini"), fallback_weight=0.0)

    def _boom(**_):
        raise RuntimeError("judge offline")

    judge.judge = _boom
    return judge


def test_skill_judge_degrades_gracefully_when_judge_errors():
    """When the judge LM call fails, the judge must fall back, not crash."""
    ex = _example()
    pred = SimpleNamespace(output="some response text")
    judge = _raising_judge()
    score, feedback = judge.evaluate(ex, pred)
    assert 0.0 <= score <= 1.0
    assert "judge error" in feedback.lower()


def test_skill_judge_score_and_gepa_shapes():
    ex = _example()
    pred = SimpleNamespace(output="risk flagged")
    judge = _raising_judge()
    # float shape
    assert isinstance(judge.score(ex, pred), float)
    # gepa shape
    gepa_result = judge.gepa(ex, pred, None, "p", None)
    assert hasattr(gepa_result, "score") and hasattr(gepa_result, "feedback")


def test_skill_judge_empty_output_short_circuits():
    judge = _raising_judge()
    score, feedback = judge.evaluate(_example(), SimpleNamespace(output=""))
    assert score == 0.0
    assert "empty" in feedback.lower()
