"""Fitness functions for evaluating evolved artifacts.

Uses LLM-as-judge with rubrics to score agent outputs.
Supports length penalties and multi-dimensional scoring.
"""

from dataclasses import dataclass
from typing import Optional

import dspy

from evolution.core.config import EvolutionConfig


@dataclass
class FitnessScore:
    """Multi-dimensional fitness score."""
    correctness: float = 0.0  # Did the agent produce correct output? (0-1)
    procedure_following: float = 0.0  # Did it follow the skill's procedure? (0-1)
    conciseness: float = 0.0  # Was it appropriately concise? (0-1)
    length_penalty: float = 0.0  # Penalty for being too verbose (0-1, 0 = no penalty)
    feedback: str = ""  # Textual feedback for GEPA's reflective analysis

    @property
    def composite(self) -> float:
        """Weighted composite score."""
        raw = (
            0.5 * self.correctness
            + 0.3 * self.procedure_following
            + 0.2 * self.conciseness
        )
        return max(0.0, raw - self.length_penalty)


class LLMJudge:
    """LLM-as-judge scorer with rubric-based evaluation.

    Scores agent outputs on multiple dimensions and provides
    textual feedback that GEPA can use for reflective mutation.
    """

    class JudgeSignature(dspy.Signature):
        """Evaluate an agent's response against an expected behavior rubric.

        Score the response on three dimensions (0.0 to 1.0 each):
        1. correctness: Did the response correctly address the task?
        2. procedure_following: Did it follow the expected approach/procedure?
        3. conciseness: Was it appropriately concise without omitting important info?

        Also provide specific, actionable feedback on what could be improved.
        """
        task_input: str = dspy.InputField(desc="The task the agent was given")
        expected_behavior: str = dspy.InputField(desc="Rubric describing what a good response looks like")
        agent_output: str = dspy.InputField(desc="The agent's actual response")
        skill_text: str = dspy.InputField(desc="The skill/instructions the agent was following")
        correctness: float = dspy.OutputField(desc="Score 0.0-1.0: Did the response correctly address the task?")
        procedure_following: float = dspy.OutputField(desc="Score 0.0-1.0: Did it follow the expected procedure?")
        conciseness: float = dspy.OutputField(desc="Score 0.0-1.0: Appropriately concise?")
        feedback: str = dspy.OutputField(desc="Specific, actionable feedback on what could be improved")

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.judge = dspy.ChainOfThought(self.JudgeSignature)

    def score(
        self,
        task_input: str,
        expected_behavior: str,
        agent_output: str,
        skill_text: str,
        artifact_size: Optional[int] = None,
        max_size: Optional[int] = None,
    ) -> FitnessScore:
        """Score an agent output using LLM-as-judge."""

        lm = dspy.LM(self.config.eval_model)

        with dspy.context(lm=lm):
            result = self.judge(
                task_input=task_input,
                expected_behavior=expected_behavior,
                agent_output=agent_output,
                skill_text=skill_text,
            )

        # Parse scores (clamp to 0-1)
        correctness = _parse_score(result.correctness)
        procedure_following = _parse_score(result.procedure_following)
        conciseness = _parse_score(result.conciseness)

        # Length penalty
        length_penalty = 0.0
        if artifact_size is not None and max_size is not None:
            ratio = artifact_size / max_size
            if ratio > 0.9:
                # Penalty ramps from 0 at 90% to 0.3 at 100%+
                length_penalty = min(0.3, (ratio - 0.9) * 3.0)

        return FitnessScore(
            correctness=correctness,
            procedure_following=procedure_following,
            conciseness=conciseness,
            length_penalty=length_penalty,
            feedback=str(result.feedback),
        )


def skill_fitness_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
    pred_name=None,
    pred_trace=None,
) -> float:
    """DSPy-compatible metric function for skill optimization.

    Accepts 5 args so it satisfies dspy.GEPA signature requirement
    (gold, pred, trace, pred_name, pred_trace) - extra args ignored
    for backward compatibility with MIPROv2 (which only passes 3).
    Returns a float 0-1 score.
    """
    # The prediction should have an 'output' field with the agent's response
    agent_output = getattr(prediction, "output", "") or ""
    expected = getattr(example, "expected_behavior", "") or ""

    if not agent_output.strip():
        return 0.0

    # Quick heuristic scoring (for speed during optimization)
    # Full LLM-as-judge scoring is expensive — use it selectively
    score = 0.5  # Base score for non-empty output

    # Check if key phrases from expected behavior appear
    expected_lower = expected.lower()
    output_lower = agent_output.lower()

    # Simple keyword overlap as a fast proxy
    expected_words = set(expected_lower.split())
    output_words = set(output_lower.split())
    if expected_words:
        overlap = len(expected_words & output_words) / len(expected_words)
        score = 0.3 + (0.7 * overlap)

    return min(1.0, max(0.0, score))


def _parse_score(value) -> float:
    """Parse a score value, handling various LLM output formats."""
    if isinstance(value, (int, float)):
        return min(1.0, max(0.0, float(value)))
    try:
        return min(1.0, max(0.0, float(str(value).strip())))
    except (ValueError, TypeError):
        return 0.5  # Default to neutral on parse failure


# ============================================================
# GEPA-aware metrics
#
# GEPA's reflective mutation needs *feedback*, not just a scalar. A metric that
# returns a bare float gives GEPA nothing to reason about ("got 0.4" → mutate
# blindly). The helpers below return ``dspy.Prediction(score=..., feedback=...)``
# so GEPA can read why a candidate scored the way it did.
# ============================================================


def _expected_keywords(example) -> set:
    expected = (getattr(example, "expected_behavior", "") or "").lower()
    return {w for w in expected.split() if len(w) > 3}


def skill_overlap_feedback(example, prediction) -> str:
    """Concrete, actionable feedback for the fast keyword-overlap metric."""
    output = (getattr(prediction, "output", "") or "").strip()
    if not output:
        return "The response was empty. Make the instructions force a concrete answer to the task."
    expected_words = _expected_keywords(example)
    if not expected_words:
        return "No rubric keywords to check; response was non-empty."
    output_words = set(output.lower().split())
    missing = sorted(expected_words - output_words)
    if not missing:
        return "Response covered all expected concepts from the rubric."
    return (
        "Response is missing expected concepts: "
        + ", ".join(missing[:12])
        + ". Revise the skill instructions so the agent reliably addresses these."
    )


def make_gepa_metric(float_metric, feedback_fn):
    """Adapt a plain ``float`` metric into a GEPA feedback metric.

    Returns a callable matching the GEPA 5-arg signature that yields
    ``dspy.Prediction(score, feedback)``.
    """

    def _gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        score = float_metric(gold, pred)
        return dspy.Prediction(score=score, feedback=feedback_fn(gold, pred))

    return _gepa_metric


class _SkillJudgeSignature(dspy.Signature):
    """Judge an agent response against the skill's expected behavior.

    Return a score from 0.0 to 1.0 and concrete, actionable feedback that
    explains the score and tells the prompt optimizer what the skill
    instructions should change to score higher next time.
    """

    task_input: str = dspy.InputField(desc="The task the agent was given")
    expected_behavior: str = dspy.InputField(desc="Rubric describing a good response")
    agent_output: str = dspy.InputField(desc="The agent's actual response")
    skill_text: str = dspy.InputField(desc="The skill/instructions the agent followed")
    score: str = dspy.OutputField(desc="A single floating point number between 0.0 and 1.0")
    feedback: str = dspy.OutputField(desc="Actionable feedback for improving the instructions")


class SkillJudge:
    """LLM-as-judge scorer for skills that yields both score and feedback.

    Exposes three call shapes:
      * ``evaluate`` → ``(score, feedback)`` tuple (the core)
      * ``score``    → ``float`` (for holdout eval and the MIPROv2 fallback)
      * ``gepa``     → ``dspy.Prediction(score, feedback)`` (for GEPA)
    """

    def __init__(self, judge_lm, *, fallback_weight: float = 0.0):
        self.judge = dspy.Predict(_SkillJudgeSignature)
        self.judge_lm = judge_lm
        self.fallback_weight = max(0.0, min(1.0, fallback_weight))

    def evaluate(self, example, prediction) -> tuple[float, str]:
        output = (getattr(prediction, "output", "") or "").strip()
        if not output:
            return 0.0, "The response was empty; make the instructions force a concrete answer."
        try:
            with dspy.context(lm=self.judge_lm):
                result = self.judge(
                    task_input=getattr(example, "task_input", "") or "",
                    expected_behavior=getattr(example, "expected_behavior", "") or "",
                    agent_output=output,
                    skill_text=getattr(example, "skill_text", "") or "",
                )
            score = _parse_score(getattr(result, "score", ""))
            feedback = str(getattr(result, "feedback", "") or "").strip() or "No feedback provided."
        except Exception as exc:  # judge LM unavailable → degrade, don't crash
            score = skill_fitness_metric(example, prediction)
            feedback = skill_overlap_feedback(example, prediction) + f" (judge error: {exc})"
        if self.fallback_weight > 0:
            overlap = skill_fitness_metric(example, prediction)
            score = (1.0 - self.fallback_weight) * score + self.fallback_weight * overlap
        return score, feedback

    def score(self, example, prediction, trace=None, pred_name=None, pred_trace=None) -> float:
        return self.evaluate(example, prediction)[0]

    def gepa(self, gold, pred, trace=None, pred_name=None, pred_trace=None) -> dspy.Prediction:
        score, feedback = self.evaluate(gold, pred)
        return dspy.Prediction(score=score, feedback=feedback)
