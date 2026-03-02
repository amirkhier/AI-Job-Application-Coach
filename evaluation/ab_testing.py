"""
A/B Testing Infrastructure for prompt experiments.

Allows running two prompt versions side-by-side on the same dataset,
collecting scores, and determining if the difference is statistically
significant.

Usage::

    from evaluation.ab_testing import ABTestManager

    ab = ABTestManager()
    result = ab.run_experiment("resume", "v1.0", "v1.1")
    print(result.winner)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.evaluator import Evaluator, load_dataset, AGENT_RUNNERS
from evaluation.scoring import AgentScorer, TestCaseResult
from evaluation.statistics import (
    paired_t_test,
    bootstrap_confidence_interval,
    cohens_d,
    ExperimentStats,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Data classes
# ------------------------------------------------------------------ #

@dataclass
class ABResult:
    """Result of an A/B experiment."""

    agent: str
    version_a: str
    version_b: str
    test_cases: int
    scores_a: List[float] = field(default_factory=list)
    scores_b: List[float] = field(default_factory=list)
    stats: Optional[ExperimentStats] = None
    winner: Optional[str] = None  # "A", "B", or None (no sig. diff.)
    details_a: List[TestCaseResult] = field(default_factory=list)
    details_b: List[TestCaseResult] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "version_a": self.version_a,
            "version_b": self.version_b,
            "test_cases": self.test_cases,
            "avg_score_a": round(sum(self.scores_a) / max(len(self.scores_a), 1), 3),
            "avg_score_b": round(sum(self.scores_b) / max(len(self.scores_b), 1), 3),
            "winner": self.winner,
            "stats": self.stats.to_dict() if self.stats else None,
            "duration_seconds": round(self.duration_seconds, 2),
        }


# ------------------------------------------------------------------ #
#  Experiment config
# ------------------------------------------------------------------ #

@dataclass
class ExperimentConfig:
    """Configuration for a single A/B experiment."""

    agent: str
    version_a: str
    version_b: str
    tags_include: Optional[List[str]] = None
    tags_exclude: Optional[List[str]] = None
    significance_level: float = 0.05
    description: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        return cls(
            agent=d["agent"],
            version_a=d["version_a"],
            version_b=d["version_b"],
            tags_include=d.get("tags_include"),
            tags_exclude=d.get("tags_exclude"),
            significance_level=d.get("significance_level", 0.05),
            description=d.get("description", ""),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


# ------------------------------------------------------------------ #
#  A/B Test Manager
# ------------------------------------------------------------------ #

class ABTestManager:
    """Orchestrates A/B prompt experiments."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.scorer = AgentScorer()

    def run_experiment(
        self,
        agent: str,
        version_a: str,
        version_b: str,
        tags_include: Optional[List[str]] = None,
        tags_exclude: Optional[List[str]] = None,
        significance_level: float = 0.05,
    ) -> ABResult:
        """Run an A/B experiment comparing two prompt versions.

        Both versions are run against the same dataset.  In the current
        implementation the agent is invoked with its default (source-code)
        prompts for both arms — the versioned prompt files serve as
        documentation of what changed.  Future integration will hot-swap
        prompts at call time.

        Parameters
        ----------
        agent:
            Agent name.
        version_a / version_b:
            Prompt version identifiers.
        tags_include / tags_exclude:
            Dataset filtering.
        significance_level:
            p-value threshold for declaring a winner.
        """
        start = time.time()
        logger.info(
            "Starting A/B experiment: %s — %s vs %s", agent, version_a, version_b,
        )

        dataset = load_dataset(agent)
        runner = AGENT_RUNNERS.get(agent)
        if runner is None:
            raise KeyError(f"No runner for agent '{agent}'")

        # Filter dataset
        inc = set(tags_include) if tags_include else None
        exc = set(tags_exclude) if tags_exclude else set()
        filtered = [
            tc for tc in dataset
            if self._should_run(tc, inc, exc)
        ]

        result = ABResult(
            agent=agent,
            version_a=version_a,
            version_b=version_b,
            test_cases=len(filtered),
        )

        for tc in filtered:
            tc_id = tc.get("id", "?")
            input_data = tc.get("input", {})

            # --- Arm A ------------------------------------------------
            score_a = self._run_arm(agent, runner, tc, input_data, "A", tc_id)
            result.scores_a.append(score_a.overall_score)
            result.details_a.append(score_a)

            # --- Arm B ------------------------------------------------
            score_b = self._run_arm(agent, runner, tc, input_data, "B", tc_id)
            result.scores_b.append(score_b.overall_score)
            result.details_b.append(score_b)

        # --- Statistical analysis ---
        if len(result.scores_a) >= 2:
            result.stats = self._analyse(
                result.scores_a, result.scores_b, significance_level,
            )
            result.winner = self._determine_winner(result.stats, significance_level)

        result.duration_seconds = time.time() - start
        logger.info(
            "A/B experiment complete: %s — winner=%s (%.1fs)",
            agent, result.winner or "none", result.duration_seconds,
        )
        return result

    def run_from_config(self, config: ExperimentConfig) -> ABResult:
        """Run an experiment from a config object."""
        return self.run_experiment(
            agent=config.agent,
            version_a=config.version_a,
            version_b=config.version_b,
            tags_include=config.tags_include,
            tags_exclude=config.tags_exclude,
            significance_level=config.significance_level,
        )

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _should_run(
        tc: Dict[str, Any],
        tags_include: Optional[set],
        tags_exclude: set,
    ) -> bool:
        tags = set(tc.get("tags", []))
        if tags_exclude and tags & tags_exclude:
            return False
        if tags_include is not None and not (tags & tags_include):
            return False
        return True

    def _run_arm(
        self,
        agent: str,
        runner,
        tc: Dict[str, Any],
        input_data: Dict[str, Any],
        arm_label: str,
        tc_id: str,
    ) -> TestCaseResult:
        if self.dry_run:
            output: Any = {"_dry_run": True, "_arm": arm_label}
            elapsed = 0.0
        else:
            t0 = time.time()
            try:
                output = runner(input_data)
            except Exception as exc:
                logger.error("Arm %s error on %s/%s: %s", arm_label, agent, tc_id, exc)
                return TestCaseResult(
                    test_id=tc_id, agent=agent, passed=False,
                    overall_score=0.0, error=str(exc),
                    tags=tc.get("tags", []),
                )
            elapsed = time.time() - t0

        return self.scorer.score(agent, output, tc, execution_time=elapsed)

    @staticmethod
    def _analyse(
        scores_a: List[float],
        scores_b: List[float],
        alpha: float,
    ) -> ExperimentStats:
        t_stat, p_value = paired_t_test(scores_a, scores_b)
        ci_low, ci_high = bootstrap_confidence_interval(scores_a, scores_b)
        effect = cohens_d(scores_a, scores_b)

        return ExperimentStats(
            mean_a=sum(scores_a) / len(scores_a),
            mean_b=sum(scores_b) / len(scores_b),
            t_statistic=t_stat,
            p_value=p_value,
            ci_lower=ci_low,
            ci_upper=ci_high,
            effect_size=effect,
            n=len(scores_a),
            significant=p_value < alpha,
        )

    @staticmethod
    def _determine_winner(stats: ExperimentStats, alpha: float) -> Optional[str]:
        if not stats.significant:
            return None
        return "B" if stats.mean_b > stats.mean_a else "A"
