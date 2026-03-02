"""
Scoring engine for the AI Job Application Coach evaluation framework.

Defines per-agent scoring dimensions, rubrics, and a generic scorer that
can evaluate any agent output against a ground-truth test case.

Usage::

    from evaluation.scoring import AgentScorer
    scorer = AgentScorer()
    result = scorer.score("resume", agent_output, test_case)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Data classes
# ------------------------------------------------------------------ #


@dataclass
class DimensionScore:
    """Score for a single evaluation dimension."""

    name: str
    score: float               # 0.0 – 10.0
    max_score: float = 10.0
    notes: str = ""
    passed: bool = True        # True if score >= min required


@dataclass
class TestCaseResult:
    """Complete evaluation result for a single test case."""

    test_id: str
    agent: str
    passed: bool
    overall_score: float
    dimension_scores: List[DimensionScore] = field(default_factory=list)
    execution_time: float = 0.0
    error: Optional[str] = None
    raw_output: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "agent": self.agent,
            "passed": self.passed,
            "overall_score": round(self.overall_score, 2),
            "dimension_scores": [
                {
                    "name": d.name,
                    "score": round(d.score, 2),
                    "max_score": d.max_score,
                    "passed": d.passed,
                    "notes": d.notes,
                }
                for d in self.dimension_scores
            ],
            "execution_time": round(self.execution_time, 3),
            "error": self.error,
            "tags": self.tags,
        }


@dataclass
class AgentEvalSummary:
    """Aggregated results for an entire agent dataset."""

    agent: str
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    error_cases: int = 0
    avg_overall_score: float = 0.0
    avg_dimension_scores: Dict[str, float] = field(default_factory=dict)
    results: List[TestCaseResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed_cases / max(self.total_cases, 1) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "error_cases": self.error_cases,
            "pass_rate": round(self.pass_rate, 1),
            "avg_overall_score": round(self.avg_overall_score, 2),
            "avg_dimension_scores": {
                k: round(v, 2) for k, v in self.avg_dimension_scores.items()
            },
            "results": [r.to_dict() for r in self.results],
        }


# ------------------------------------------------------------------ #
#  Scoring rubrics
# ------------------------------------------------------------------ #

# Dimensions for each agent — used to validate expected keys in datasets
# and to weight the overall score computation.
AGENT_DIMENSIONS: Dict[str, Dict[str, float]] = {
    "resume": {
        "analysis_accuracy": 1.0,
        "feedback_quality": 1.0,
        "ats_assessment": 0.8,
        "improvement_relevance": 1.0,
    },
    "interview": {
        "question_relevance": 1.0,
        "difficulty_match": 0.8,
        "evaluation_fairness": 1.0,
        "feedback_quality": 1.0,
    },
    "knowledge": {
        "answer_accuracy": 1.0,
        "rag_grounding": 0.9,
        "answer_completeness": 1.0,
        "recommendation_quality": 0.7,
    },
    "job_search": {
        "result_relevance": 1.0,
        "match_accuracy": 1.0,
        "location_accuracy": 0.8,
        "diversity": 0.6,
    },
    "router": {
        "intent_accuracy": 1.0,
        "confidence_calibration": 0.8,
    },
}


# ------------------------------------------------------------------ #
#  Automatic checkers
# ------------------------------------------------------------------ #

def _check_json_structure(output: Any, required_keys: List[str]) -> float:
    """Return 1.0 if all keys present, partial credit otherwise."""
    if not isinstance(output, dict):
        return 0.0
    present = sum(1 for k in required_keys if k in output)
    return present / max(len(required_keys), 1)


def _check_score_range(
    output: Dict[str, Any],
    key: str,
    expected_min: float,
    expected_max: float,
) -> float:
    """Return 1.0 if value within range, scaled penalty outside."""
    val = output.get(key)
    if val is None:
        return 0.0
    val = float(val)
    if expected_min <= val <= expected_max:
        return 1.0
    # Distance-based penalty
    if val < expected_min:
        return max(0.0, 1.0 - (expected_min - val) / expected_max)
    return max(0.0, 1.0 - (val - expected_max) / expected_max)


def _check_contains_concepts(text: str, concepts: List[str]) -> float:
    """Return fraction of concepts found (case-insensitive substring)."""
    if not concepts:
        return 1.0
    text_lower = text.lower()
    found = sum(1 for c in concepts if c.lower() in text_lower)
    return found / len(concepts)


def _check_must_not_contain(text: str, forbidden: List[str]) -> float:
    """Return 1.0 if none of the forbidden strings appear."""
    text_lower = text.lower()
    for phrase in forbidden:
        if phrase.lower() in text_lower:
            return 0.0
    return 1.0


def _check_list_length(output: Any, key: str, expected: int) -> float:
    """Return 1.0 if list has expected length, scaled otherwise."""
    if not isinstance(output, (dict, list)):
        return 0.0
    items = output.get(key, output) if isinstance(output, dict) else output
    if not isinstance(items, list):
        return 0.0
    actual = len(items)
    if actual == expected:
        return 1.0
    return max(0.0, 1.0 - abs(actual - expected) / max(expected, 1))


# ------------------------------------------------------------------ #
#  Agent-specific scorers
# ------------------------------------------------------------------ #

class AgentScorer:
    """Multi-agent scorer that delegates to per-agent scoring logic."""

    def score(
        self,
        agent: str,
        output: Any,
        test_case: Dict[str, Any],
        execution_time: float = 0.0,
    ) -> TestCaseResult:
        """Score an agent output against a test case.

        Parameters
        ----------
        agent:
            Agent name (``resume``, ``interview``, ``router``, etc.)
        output:
            Raw output from the agent call.
        test_case:
            The dataset test entry (with ``expected``).
        execution_time:
            Wall-clock seconds for the agent call.

        Returns
        -------
        TestCaseResult
        """
        scoring_fn = {
            "resume": self._score_resume,
            "interview": self._score_interview,
            "knowledge": self._score_knowledge,
            "job_search": self._score_job_search,
            "router": self._score_router,
        }.get(agent)

        if scoring_fn is None:
            return TestCaseResult(
                test_id=test_case.get("id", "?"),
                agent=agent,
                passed=False,
                overall_score=0.0,
                error=f"Unknown agent: {agent}",
                tags=test_case.get("tags", []),
            )

        try:
            return scoring_fn(output, test_case, execution_time)
        except Exception as exc:
            logger.exception("Scoring error for %s / %s", agent, test_case.get("id"))
            return TestCaseResult(
                test_id=test_case.get("id", "?"),
                agent=agent,
                passed=False,
                overall_score=0.0,
                error=str(exc),
                tags=test_case.get("tags", []),
            )

    # ----- helpers ---------------------------------------------------- #

    @staticmethod
    def _aggregate_dimensions(
        dims: List[DimensionScore],
        weights: Dict[str, float],
    ) -> float:
        """Weighted average of dimension scores (0-10 scale)."""
        total_weight = 0.0
        weighted_sum = 0.0
        for d in dims:
            w = weights.get(d.name, 1.0)
            weighted_sum += d.score * w
            total_weight += w
        return weighted_sum / max(total_weight, 1e-9)

    @staticmethod
    def _dim(
        name: str,
        score: float,
        min_required: float = 0.0,
        notes: str = "",
    ) -> DimensionScore:
        return DimensionScore(
            name=name,
            score=min(max(score, 0.0), 10.0),
            passed=score >= min_required,
            notes=notes,
        )

    # ================================================================== #
    #  Resume scorer
    # ================================================================== #

    def _score_resume(
        self, output: Any, tc: Dict[str, Any], elapsed: float,
    ) -> TestCaseResult:
        expected = tc.get("expected", {})
        dims_spec = expected.get("dimensions", {})
        dim_list: List[DimensionScore] = []

        # ----- analysis_accuracy ----------------------------------------
        if isinstance(output, dict):
            struct_score = _check_json_structure(
                output,
                ["overall_score", "strengths", "weaknesses", "recommendations",
                 "ats_compatibility", "keyword_analysis", "section_feedback"],
            )
            score_val = struct_score * 10
        else:
            score_val = 0.0
        aa_spec = dims_spec.get("analysis_accuracy", {})
        dim_list.append(self._dim(
            "analysis_accuracy", score_val,
            min_required=aa_spec.get("min_score", 0.0),
            notes=aa_spec.get("notes", ""),
        ))

        # ----- feedback_quality -----------------------------------------
        fq_score = 0.0
        if isinstance(output, dict):
            recs = output.get("recommendations", [])
            strengths = output.get("strengths", [])
            fq_score = min(len(recs), 5) / 5 * 5 + min(len(strengths), 3) / 3 * 5
        fq_spec = dims_spec.get("feedback_quality", {})
        dim_list.append(self._dim(
            "feedback_quality", fq_score,
            min_required=fq_spec.get("min_score", 0.0),
            notes=fq_spec.get("notes", ""),
        ))

        # ----- ats_assessment -------------------------------------------
        ats_score = 0.0
        if isinstance(output, dict) and isinstance(output.get("ats_compatibility"), dict):
            ats = output["ats_compatibility"]
            has_keys = _check_json_structure(ats, ["score", "issues", "suggestions"])
            ats_score = has_keys * 10
        ats_spec = dims_spec.get("ats_assessment", {})
        dim_list.append(self._dim(
            "ats_assessment", ats_score,
            min_required=ats_spec.get("min_score", 0.0),
            notes=ats_spec.get("notes", ""),
        ))

        # ----- score range check ----------------------------------------
        score_range = expected.get("score_range")
        if score_range and isinstance(output, dict):
            overall = output.get("overall_score", 5.0)
            range_ok = _check_score_range(
                output, "overall_score", score_range[0], score_range[1],
            )
            ir_score = range_ok * 10
        else:
            ir_score = 5.0  # neutral when no score_range specified
        ir_spec = dims_spec.get("improvement_relevance", {})
        dim_list.append(self._dim(
            "improvement_relevance", ir_score,
            min_required=ir_spec.get("min_score", 0.0),
            notes=ir_spec.get("notes", ""),
        ))

        # ----- must_contain check ----------------------------------------
        must_contain = expected.get("must_contain", [])
        if must_contain and isinstance(output, dict):
            text = json.dumps(output)
            contain_frac = _check_contains_concepts(text, must_contain)
            # Blend into feedback_quality
            dim_list[1] = self._dim(
                "feedback_quality",
                (dim_list[1].score + contain_frac * 10) / 2,
                min_required=fq_spec.get("min_score", 0.0),
                notes=f"must_contain match: {contain_frac:.0%}",
            )

        overall = self._aggregate_dimensions(dim_list, AGENT_DIMENSIONS["resume"])
        all_pass = all(d.passed for d in dim_list)

        return TestCaseResult(
            test_id=tc["id"],
            agent="resume",
            passed=all_pass,
            overall_score=overall,
            dimension_scores=dim_list,
            execution_time=elapsed,
            raw_output=output if isinstance(output, dict) else {"raw": str(output)},
            tags=tc.get("tags", []),
        )

    # ================================================================== #
    #  Interview scorer
    # ================================================================== #

    def _score_interview(
        self, output: Any, tc: Dict[str, Any], elapsed: float,
    ) -> TestCaseResult:
        expected = tc.get("expected", {})
        dims_spec = expected.get("dimensions", {})
        dim_list: List[DimensionScore] = []

        # ----- question generation case --------------------------------
        expected_count = expected.get("question_count")
        if expected_count is not None:
            if isinstance(output, list):
                count_score = _check_list_length(output, "", expected_count) * 10
                # Check each question has required keys
                required_q_keys = ["id", "question", "type", "difficulty", "key_points"]
                key_scores = [
                    _check_json_structure(q, required_q_keys)
                    for q in output if isinstance(q, dict)
                ]
                struct_avg = (sum(key_scores) / max(len(key_scores), 1)) * 10
            else:
                count_score = 0.0
                struct_avg = 0.0

            qr_spec = dims_spec.get("question_relevance", {})
            dim_list.append(self._dim(
                "question_relevance", struct_avg,
                min_required=qr_spec.get("min_score", 0.0),
                notes=qr_spec.get("notes", ""),
            ))
            dm_spec = dims_spec.get("difficulty_match", {})
            dim_list.append(self._dim(
                "difficulty_match", count_score,
                min_required=dm_spec.get("min_score", 0.0),
                notes=f"Expected {expected_count} questions",
            ))

        # ----- evaluation case -----------------------------------------
        score_range = expected.get("score_range")
        if score_range is not None and isinstance(output, dict):
            range_score = _check_score_range(
                output, "overall_score", score_range[0], score_range[1],
            ) * 10
            ef_spec = dims_spec.get("evaluation_fairness", {})
            dim_list.append(self._dim(
                "evaluation_fairness", range_score,
                min_required=ef_spec.get("min_score", 0.0),
                notes=ef_spec.get("notes", ""),
            ))

            # Feedback quality based on presence of key fields
            feedback_keys = [
                "strength_areas", "improvement_areas",
                "specific_feedback", "suggested_improvement",
            ]
            fb_score = _check_json_structure(output, feedback_keys) * 10
            fq_spec = dims_spec.get("feedback_quality", {})
            dim_list.append(self._dim(
                "feedback_quality", fb_score,
                min_required=fq_spec.get("min_score", 0.0),
                notes=fq_spec.get("notes", ""),
            ))

        # ----- summary case -------------------------------------------
        sa_spec = dims_spec.get("summary_accuracy")
        if sa_spec is not None and isinstance(output, dict):
            summary_keys = [
                "overall_score", "total_questions", "performance_level",
                "strongest_areas", "weakest_areas", "key_recommendations",
            ]
            sa_score = _check_json_structure(output, summary_keys) * 10
            dim_list.append(self._dim(
                "summary_accuracy", sa_score,
                min_required=sa_spec.get("min_score", 0.0),
                notes=sa_spec.get("notes", ""),
            ))
            rq_spec = dims_spec.get("recommendation_quality", {})
            recs = output.get("key_recommendations", [])
            rq_score = min(len(recs), 5) / 5 * 10
            dim_list.append(self._dim(
                "recommendation_quality", rq_score,
                min_required=rq_spec.get("min_score", 0.0),
                notes=rq_spec.get("notes", ""),
            ))

        # Fallback: if no dimensions matched, still return something
        if not dim_list:
            dim_list.append(self._dim("general", 5.0))

        overall = self._aggregate_dimensions(dim_list, AGENT_DIMENSIONS["interview"])
        return TestCaseResult(
            test_id=tc["id"],
            agent="interview",
            passed=all(d.passed for d in dim_list),
            overall_score=overall,
            dimension_scores=dim_list,
            execution_time=elapsed,
            raw_output=output if isinstance(output, (dict, list)) else {"raw": str(output)},
            tags=tc.get("tags", []),
        )

    # ================================================================== #
    #  Knowledge scorer
    # ================================================================== #

    def _score_knowledge(
        self, output: Any, tc: Dict[str, Any], elapsed: float,
    ) -> TestCaseResult:
        expected = tc.get("expected", {})
        dims_spec = expected.get("dimensions", {})
        dim_list: List[DimensionScore] = []

        answer_text = ""
        if isinstance(output, dict):
            answer_text = output.get("answer", "")
        elif isinstance(output, str):
            answer_text = output

        # ----- answer_accuracy ------------------------------------------
        aa_spec = dims_spec.get("answer_accuracy", {})
        concepts = expected.get("must_contain_concepts", [])
        if concepts:
            concept_score = _check_contains_concepts(answer_text, concepts) * 10
        else:
            # If no concepts specified, give credit for non-empty answer
            concept_score = 7.0 if len(answer_text) > 50 else 3.0
        dim_list.append(self._dim(
            "answer_accuracy", concept_score,
            min_required=aa_spec.get("min_score", 0.0),
            notes=aa_spec.get("notes", ""),
        ))

        # ----- rag_grounding -------------------------------------------
        rg_spec = dims_spec.get("rag_grounding")
        if rg_spec is not None and isinstance(output, dict):
            sources = output.get("sources", output.get("sources_used", []))
            confidence = output.get("relevance_score", output.get("confidence", 0.0))
            grounding_score = 0.0
            if sources:
                grounding_score += min(len(sources), 3) / 3 * 5
            grounding_score += float(confidence) * 5
            dim_list.append(self._dim(
                "rag_grounding", grounding_score,
                min_required=rg_spec.get("min_score", 0.0),
                notes=rg_spec.get("notes", ""),
            ))

        # ----- answer_completeness -------------------------------------
        ac_spec = dims_spec.get("answer_completeness", {})
        if ac_spec:
            # Length heuristic + concept coverage
            length_score = min(len(answer_text) / 200, 1.0) * 5
            comp_score = length_score + (concept_score / 2)
            dim_list.append(self._dim(
                "answer_completeness", comp_score,
                min_required=ac_spec.get("min_score", 0.0),
                notes=ac_spec.get("notes", ""),
            ))

        # ----- must_not_contain ----------------------------------------
        forbidden = expected.get("must_not_contain", [])
        if forbidden:
            safety_score = _check_must_not_contain(answer_text, forbidden) * 10
            dim_list.append(self._dim(
                "safety", safety_score,
                min_required=8.0,
                notes="Must not contain restricted content",
            ))

        overall = self._aggregate_dimensions(dim_list, AGENT_DIMENSIONS["knowledge"])
        return TestCaseResult(
            test_id=tc["id"],
            agent="knowledge",
            passed=all(d.passed for d in dim_list),
            overall_score=overall,
            dimension_scores=dim_list,
            execution_time=elapsed,
            raw_output=output if isinstance(output, dict) else {"raw": str(output)},
            tags=tc.get("tags", []),
        )

    # ================================================================== #
    #  Job search scorer
    # ================================================================== #

    def _score_job_search(
        self, output: Any, tc: Dict[str, Any], elapsed: float,
    ) -> TestCaseResult:
        expected = tc.get("expected", {})
        dims_spec = expected.get("dimensions", {})
        dim_list: List[DimensionScore] = []

        jobs = []
        if isinstance(output, dict):
            jobs = output.get("jobs", output.get("results", []))
            if isinstance(output.get("matched_jobs"), list):
                jobs = output["matched_jobs"]
        elif isinstance(output, list):
            jobs = output

        # ----- result_relevance ----------------------------------------
        rr_spec = dims_spec.get("result_relevance", {})
        min_count = expected.get("result_count_min", 0)
        if isinstance(jobs, list) and len(jobs) >= min_count and len(jobs) > 0:
            # Check each job has essential keys
            needed_keys = ["title", "company"]
            key_checks = [_check_json_structure(j, needed_keys) for j in jobs if isinstance(j, dict)]
            rel_score = (sum(key_checks) / max(len(key_checks), 1)) * 10
        elif isinstance(jobs, list) and len(jobs) == 0:
            rel_score = 2.0  # Returned something, but empty
        else:
            rel_score = 0.0
        dim_list.append(self._dim(
            "result_relevance", rel_score,
            min_required=rr_spec.get("min_score", 0.0),
            notes=rr_spec.get("notes", ""),
        ))

        # ----- match_accuracy ------------------------------------------
        ma_spec = dims_spec.get("match_accuracy", {})
        if expected.get("should_include_match_score") and isinstance(jobs, list):
            scores = [
                j.get("match_score", 0) for j in jobs
                if isinstance(j, dict) and "match_score" in j
            ]
            if scores:
                match_score = (sum(scores) / len(scores)) * 10
            else:
                match_score = 3.0
        elif ma_spec:
            match_score = 5.0  # neutral when no explicit matching expected
        else:
            match_score = 5.0
        if ma_spec:
            dim_list.append(self._dim(
                "match_accuracy", match_score,
                min_required=ma_spec.get("min_score", 0.0),
                notes=ma_spec.get("notes", ""),
            ))

        overall = self._aggregate_dimensions(dim_list, AGENT_DIMENSIONS["job_search"])
        return TestCaseResult(
            test_id=tc["id"],
            agent="job_search",
            passed=all(d.passed for d in dim_list),
            overall_score=overall,
            dimension_scores=dim_list,
            execution_time=elapsed,
            raw_output=output if isinstance(output, (dict, list)) else {"raw": str(output)},
            tags=tc.get("tags", []),
        )

    # ================================================================== #
    #  Router scorer
    # ================================================================== #

    def _score_router(
        self, output: Any, tc: Dict[str, Any], elapsed: float,
    ) -> TestCaseResult:
        expected = tc.get("expected", {})
        dims_spec = expected.get("dimensions", {})
        dim_list: List[DimensionScore] = []

        out_intent = None
        out_confidence = None
        if isinstance(output, dict):
            out_intent = output.get("intent")
            out_confidence = output.get("confidence")

        # ----- intent_accuracy ------------------------------------------
        expected_intent = expected.get("intent")
        acceptable = expected.get("acceptable_intents", [])

        if out_intent == expected_intent:
            intent_score = 10.0
        elif out_intent in acceptable:
            intent_score = 7.0
        elif out_intent is not None:
            intent_score = 0.0
        else:
            intent_score = 0.0

        ia_spec = dims_spec.get("intent_accuracy", {})
        dim_list.append(self._dim(
            "intent_accuracy", intent_score,
            min_required=ia_spec.get("min_score", 5.0),
            notes=f"expected={expected_intent}, got={out_intent}",
        ))

        # ----- confidence_calibration -----------------------------------
        min_conf = expected.get("min_confidence", 0.0)
        if out_confidence is not None:
            conf_score = 10.0 if float(out_confidence) >= min_conf else 5.0
        else:
            conf_score = 0.0
        cc_spec = dims_spec.get("confidence_calibration", {})
        dim_list.append(self._dim(
            "confidence_calibration", conf_score,
            min_required=cc_spec.get("min_score", 0.0),
            notes=f"min_required={min_conf}, got={out_confidence}",
        ))

        overall = self._aggregate_dimensions(dim_list, AGENT_DIMENSIONS["router"])
        return TestCaseResult(
            test_id=tc["id"],
            agent="router",
            passed=all(d.passed for d in dim_list),
            overall_score=overall,
            dimension_scores=dim_list,
            execution_time=elapsed,
            raw_output=output if isinstance(output, dict) else {"raw": str(output)},
            tags=tc.get("tags", []),
        )
