"""
Evaluator — orchestrates agent execution against test datasets.

Loads JSON test cases, invokes the appropriate agent, collects output,
scores each result via ``evaluation.scoring``, and returns structured
``AgentEvalSummary`` objects.

Example
-------
::

    from evaluation.evaluator import Evaluator

    ev = Evaluator()
    summary = ev.run_agent("resume")   # evaluates all resume test cases
    print(summary.pass_rate)

    # Or run everything:
    all_summaries = ev.run_all()
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from evaluation.scoring import AgentScorer, AgentEvalSummary, TestCaseResult

logger = logging.getLogger(__name__)

# Resolve dataset root relative to this file
_DATASET_ROOT = Path(__file__).resolve().parent / "datasets"


# ------------------------------------------------------------------ #
#  Agent wrappers
# ------------------------------------------------------------------ #
# Each wrapper accepts a single test-case ``input`` dict and returns
# the agent's raw output.  These are thin adapters — they import at
# call-time so the framework can be loaded without every dependency.
# ------------------------------------------------------------------ #

def _run_resume(input_data: Dict[str, Any]) -> Any:
    """Invoke ResumeAgent.analyze_resume."""
    from app.agents.resume import ResumeAgent

    agent = ResumeAgent()
    resume_text = input_data.get("resume_text", "")
    job_description = input_data.get("job_description")
    return agent.analyze_resume(resume_text, job_description)


def _run_interview(input_data: Dict[str, Any]) -> Any:
    """Invoke InterviewAgent — dispatches to generate or evaluate."""
    from app.agents.interview import InterviewAgent

    agent = InterviewAgent()

    # Generation case
    if "role" in input_data and "count" in input_data:
        return agent.generate_questions(
            role=input_data["role"],
            level=input_data.get("level", "mid"),
            count=input_data["count"],
        )

    # Evaluation case
    if "question" in input_data and "answer" in input_data:
        return agent.evaluate_answer(
            question=input_data["question"],
            answer=input_data["answer"],
            role=input_data.get("role", "Software Engineer"),
            level=input_data.get("level", "mid"),
        )

    # Session summary case
    if "session_data" in input_data:
        return agent.generate_session_summary(
            session_data=input_data["session_data"],
        )

    logger.warning("Interview test case has unrecognised input shape: %s", list(input_data.keys()))
    return None


def _run_knowledge(input_data: Dict[str, Any]) -> Any:
    """Invoke KnowledgeAgent.answer_question."""
    from app.agents.knowledge import KnowledgeAgent

    agent = KnowledgeAgent()
    return agent.answer_question(input_data.get("query", ""))


def _run_job_search(input_data: Dict[str, Any]) -> Any:
    """Invoke JobSearchAgent.search_jobs (with optional matching)."""
    from app.agents.job_search import JobSearchAgent

    agent = JobSearchAgent()

    resume_text = input_data.get("resume_text")
    if resume_text:
        return agent.search_jobs_with_matching(
            query=input_data.get("role", ""),
            location=input_data.get("location", ""),
            experience_level=input_data.get("experience_level", "mid"),
            resume_text=resume_text,
        )
    return agent.search_jobs(
        query=input_data.get("role", ""),
        location=input_data.get("location", ""),
        experience_level=input_data.get("experience_level", "mid"),
    )


def _run_router(input_data: Dict[str, Any]) -> Any:
    """Invoke RouterAgent.classify_intent."""
    from app.agents.router import RouterAgent

    agent = RouterAgent()
    return agent.classify_intent(
        user_query=input_data.get("user_query", ""),
        has_active_session=input_data.get("has_active_session", False),
    )


AGENT_RUNNERS: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    "resume": _run_resume,
    "interview": _run_interview,
    "knowledge": _run_knowledge,
    "job_search": _run_job_search,
    "router": _run_router,
}


# ------------------------------------------------------------------ #
#  Dataset loader
# ------------------------------------------------------------------ #

def load_dataset(agent: str) -> List[Dict[str, Any]]:
    """Load JSON test cases for *agent* from the datasets directory."""
    ds_path = _DATASET_ROOT / agent / f"{agent}_dataset.json"
    if not ds_path.exists():
        raise FileNotFoundError(f"No dataset found at {ds_path}")
    with open(ds_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Dataset {ds_path} must be a JSON array")
    logger.info("Loaded %d test cases from %s", len(data), ds_path)
    return data


# ------------------------------------------------------------------ #
#  Evaluator
# ------------------------------------------------------------------ #

class Evaluator:
    """Top-level evaluation orchestrator.

    Parameters
    ----------
    agents : list[str] | None
        Agents to evaluate.  ``None`` means all available.
    tags_include : list[str] | None
        If set, only run test cases whose tags intersect with these.
    tags_exclude : list[str] | None
        If set, skip test cases with any of these tags.
    dry_run : bool
        If ``True``, skip the actual LLM call and use a dummy output.
    """

    def __init__(
        self,
        agents: Optional[List[str]] = None,
        tags_include: Optional[List[str]] = None,
        tags_exclude: Optional[List[str]] = None,
        dry_run: bool = False,
    ):
        self.agents = agents or list(AGENT_RUNNERS.keys())
        self.tags_include = set(tags_include) if tags_include else None
        self.tags_exclude = set(tags_exclude) if tags_exclude else set()
        self.dry_run = dry_run
        self.scorer = AgentScorer()

    # ------------------------------------------------------------------ #
    #  Run single agent
    # ------------------------------------------------------------------ #

    def run_agent(self, agent: str) -> AgentEvalSummary:
        """Run all test cases for a single agent and return the summary."""
        dataset = load_dataset(agent)
        runner = AGENT_RUNNERS.get(agent)
        if runner is None:
            raise KeyError(f"No runner registered for agent '{agent}'")

        summary = AgentEvalSummary(agent=agent)
        for tc in dataset:
            if not self._should_run(tc):
                continue

            summary.total_cases += 1
            result = self._execute_case(agent, runner, tc)
            summary.results.append(result)

            if result.error:
                summary.error_cases += 1
            elif result.passed:
                summary.passed_cases += 1
            else:
                summary.failed_cases += 1

        # Aggregate scores
        self._compute_averages(summary)
        return summary

    # ------------------------------------------------------------------ #
    #  Run all agents
    # ------------------------------------------------------------------ #

    def run_all(self) -> Dict[str, AgentEvalSummary]:
        """Run evaluation for all configured agents."""
        results: Dict[str, AgentEvalSummary] = {}
        for agent in self.agents:
            try:
                results[agent] = self.run_agent(agent)
            except FileNotFoundError:
                logger.warning("No dataset for agent '%s' — skipping.", agent)
            except Exception as exc:
                logger.exception("Error evaluating agent '%s'", agent)
                results[agent] = AgentEvalSummary(agent=agent)
        return results

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _should_run(self, tc: Dict[str, Any]) -> bool:
        """Determine if a test case should be executed based on tag filters."""
        tags = set(tc.get("tags", []))
        if self.tags_exclude and tags & self.tags_exclude:
            return False
        if self.tags_include is not None and not (tags & self.tags_include):
            return False
        return True

    def _execute_case(
        self,
        agent: str,
        runner: Callable,
        tc: Dict[str, Any],
    ) -> TestCaseResult:
        """Execute a single test case and score the result."""
        tc_id = tc.get("id", "?")
        input_data = tc.get("input", {})

        if self.dry_run:
            logger.info("[DRY-RUN] %s / %s — skipping LLM call", agent, tc_id)
            output: Any = {"_dry_run": True}
            elapsed = 0.0
        else:
            logger.info("Running %s / %s …", agent, tc_id)
            start = time.time()
            try:
                output = runner(input_data)
            except Exception as exc:
                logger.error("Agent error %s / %s: %s", agent, tc_id, exc)
                return TestCaseResult(
                    test_id=tc_id,
                    agent=agent,
                    passed=False,
                    overall_score=0.0,
                    error=str(exc),
                    tags=tc.get("tags", []),
                )
            elapsed = time.time() - start

        return self.scorer.score(agent, output, tc, execution_time=elapsed)

    @staticmethod
    def _compute_averages(summary: AgentEvalSummary) -> None:
        """Compute aggregate score averages for the summary."""
        if not summary.results:
            return

        valid_results = [r for r in summary.results if r.error is None]
        if not valid_results:
            return

        summary.avg_overall_score = (
            sum(r.overall_score for r in valid_results) / len(valid_results)
        )

        # Per-dimension averages
        dim_totals: Dict[str, List[float]] = {}
        for r in valid_results:
            for d in r.dimension_scores:
                dim_totals.setdefault(d.name, []).append(d.score)
        summary.avg_dimension_scores = {
            name: sum(vals) / len(vals) for name, vals in dim_totals.items()
        }
