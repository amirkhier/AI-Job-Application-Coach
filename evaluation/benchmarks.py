"""
Performance benchmarks for agent response latency and throughput.

Measures wall-clock time for each agent operation and produces a baseline
report.  Supports both live (with LLM calls) and mock benchmarks.

Usage::

    # Mock benchmark (no LLM calls)
    python -m evaluation.benchmarks --mock

    # Live benchmark (requires OPENAI_API_KEY)
    python -m evaluation.benchmarks --live --iterations 3
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Data classes
# ------------------------------------------------------------------ #

@dataclass
class BenchmarkResult:
    """Single benchmark measurement."""

    operation: str
    agent: str
    iterations: int
    times: List[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return statistics.mean(self.times) if self.times else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.times) if self.times else 0.0

    @property
    def p95(self) -> float:
        if len(self.times) < 2:
            return self.mean
        sorted_t = sorted(self.times)
        idx = int(0.95 * len(sorted_t))
        return sorted_t[min(idx, len(sorted_t) - 1)]

    @property
    def min_time(self) -> float:
        return min(self.times) if self.times else 0.0

    @property
    def max_time(self) -> float:
        return max(self.times) if self.times else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "agent": self.agent,
            "iterations": self.iterations,
            "mean_ms": round(self.mean * 1000, 1),
            "median_ms": round(self.median * 1000, 1),
            "p95_ms": round(self.p95 * 1000, 1),
            "min_ms": round(self.min_time * 1000, 1),
            "max_ms": round(self.max_time * 1000, 1),
        }


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""

    results: List[BenchmarkResult] = field(default_factory=list)
    mode: str = "mock"
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "mode": self.mode,
            "results": [r.to_dict() for r in self.results],
        }


# ------------------------------------------------------------------ #
#  Benchmark definitions
# ------------------------------------------------------------------ #

SAMPLE_RESUME = """
John Doe
Software Engineer | john@example.com | (555) 123-4567

EXPERIENCE
Senior Software Engineer, TechCorp (2020-2024)
- Led migration of monolithic app to microservices, reducing deploy time by 60%
- Designed REST APIs serving 10M+ requests/day
- Mentored 3 junior developers

Software Engineer, StartupXYZ (2018-2020)
- Built real-time data pipeline processing 1TB/day
- Reduced API latency by 40% through caching optimization

EDUCATION
B.S. Computer Science, MIT (2018)

SKILLS
Python, Java, AWS, Docker, Kubernetes, PostgreSQL, Redis, React
"""

SAMPLE_JD = """
Senior Backend Engineer at CloudScale Inc.
Requirements: 5+ years Python, distributed systems, AWS, CI/CD.
"""


def _make_mock_llm_response(content: str):
    """Create a mock LLM response object with a real ``.content`` string."""
    mock = MagicMock()
    mock.content = content
    return mock


def _make_mock_chain(response_json: str):
    """Return a mock chain whose ``.invoke()`` yields a proper LLM response."""
    chain = MagicMock()
    chain.invoke.return_value = _make_mock_llm_response(response_json)
    return chain


def _patch_prompt_pipe(agent, prompt_attr: str, mock_chain):
    """Replace a prompt attribute with a MagicMock whose ``|`` returns *mock_chain*.

    Python looks up ``__or__`` on the **type**, so patching an instance
    attribute of a ChatPromptTemplate won't affect the ``|`` operator.
    Replacing the attribute with a MagicMock works because MagicMock
    supports the ``__or__`` dunder natively.
    """
    mock_prompt = MagicMock()
    mock_prompt.__or__.return_value = mock_chain
    setattr(agent, prompt_attr, mock_prompt)


# ---- Canned response payloads ----------------------------------------

def _get_mock_resume_response() -> str:
    return json.dumps({
        "overall_score": 7.5,
        "strengths": ["Strong technical background", "Quantified achievements"],
        "weaknesses": ["Could add more leadership examples"],
        "recommendations": ["Add a professional summary"],
        "ats_compatibility": {"score": 8.0, "issues": [], "suggestions": ["Add keywords"]},
        "keyword_analysis": {"present_keywords": ["Python", "AWS"], "missing_keywords": ["CI/CD"], "keyword_density_notes": "Good"},
        "section_feedback": {"contact_info": "Good", "summary": "Missing", "experience": "Strong", "skills": "Good", "education": "Good"},
    })


def _get_mock_improvement_response() -> str:
    return json.dumps({
        "improved_summary": "Experienced software engineer with 6+ years...",
        "improved_bullets": [
            {"original": "Led migration", "improved": "Led cloud migration of monolith to 12 microservices, reducing deploy time 60%"}
        ],
        "additional_suggestions": ["Add metrics to every bullet"],
        "priority_actions": ["Rewrite professional summary", "Quantify impact"],
    })


def _get_mock_router_response() -> str:
    return json.dumps({
        "intent": "resume_analysis",
        "confidence": 0.95,
        "reasoning": "User wants resume reviewed",
    })


def _get_mock_interview_response() -> str:
    return json.dumps([
        {"id": "q1", "question": "Tell me about yourself.", "type": "behavioral", "difficulty": "easy", "key_points": ["Background"]},
        {"id": "q2", "question": "System design question.", "type": "technical", "difficulty": "medium", "key_points": ["Scalability"]},
    ])


def _get_mock_knowledge_response() -> str:
    return json.dumps({
        "answer": "When negotiating salary, research market rates first.",
        "sources_used": ["salary_negotiation.md"],
        "confidence": 0.85,
        "related_topics": ["benefits", "equity"],
    })


# ------------------------------------------------------------------ #
#  Benchmark runner
# ------------------------------------------------------------------ #

class BenchmarkRunner:
    """Runs performance benchmarks for all agents."""

    def __init__(self, iterations: int = 5, mode: str = "mock"):
        self.iterations = iterations
        self.mode = mode

    def run_all(self) -> BenchmarkSuite:
        suite = BenchmarkSuite(
            mode=self.mode,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        benchmarks = [
            ("resume.analyze", "resume", self._bench_resume_analyze),
            ("resume.improve", "resume", self._bench_resume_improve),
            ("router.classify", "router", self._bench_router_classify),
            ("interview.generate", "interview", self._bench_interview_generate),
            ("knowledge.answer", "knowledge", self._bench_knowledge_answer),
        ]

        for op_name, agent, bench_fn in benchmarks:
            result = BenchmarkResult(
                operation=op_name,
                agent=agent,
                iterations=self.iterations,
            )
            for i in range(self.iterations):
                elapsed = bench_fn()
                result.times.append(elapsed)
                logger.debug("%s iteration %d: %.3fs", op_name, i + 1, elapsed)

            suite.results.append(result)
            print(f"  {op_name:30s} mean={result.mean*1000:.1f}ms  "
                  f"p95={result.p95*1000:.1f}ms  ({self.iterations} iters)")

        return suite

    # ---- Individual benchmarks ----

    def _bench_resume_analyze(self) -> float:
        if self.mode == "mock":
            with patch("app.agents.resume.ChatOpenAI"):
                from app.agents.resume import ResumeAgent
                agent = ResumeAgent()
                mock_chain = _make_mock_chain(_get_mock_resume_response())
                _patch_prompt_pipe(agent, "_analysis_prompt", mock_chain)
                t0 = time.time()
                agent.analyze_resume(SAMPLE_RESUME, SAMPLE_JD)
                return time.time() - t0
        else:
            from app.agents.resume import ResumeAgent
            agent = ResumeAgent()
            t0 = time.time()
            agent.analyze_resume(SAMPLE_RESUME, SAMPLE_JD)
            return time.time() - t0

    def _bench_resume_improve(self) -> float:
        if self.mode == "mock":
            with patch("app.agents.resume.ChatOpenAI"):
                from app.agents.resume import ResumeAgent
                agent = ResumeAgent()
                mock_analysis = json.loads(_get_mock_resume_response())
                mock_chain = _make_mock_chain(_get_mock_improvement_response())
                _patch_prompt_pipe(agent, "_improvement_prompt", mock_chain)
                t0 = time.time()
                agent.suggest_improvements(SAMPLE_RESUME, SAMPLE_JD, mock_analysis)
                return time.time() - t0
        else:
            from app.agents.resume import ResumeAgent
            agent = ResumeAgent()
            analysis = agent.analyze_resume(SAMPLE_RESUME, SAMPLE_JD)
            t0 = time.time()
            agent.suggest_improvements(SAMPLE_RESUME, SAMPLE_JD, analysis)
            return time.time() - t0

    def _bench_router_classify(self) -> float:
        if self.mode == "mock":
            with patch("app.agents.router.ChatOpenAI"):
                from app.agents.router import RouterAgent
                agent = RouterAgent()
                mock_chain = _make_mock_chain(_get_mock_router_response())
                agent.classification_chain = mock_chain
                t0 = time.time()
                agent.classify_intent("Help me improve my resume")
                return time.time() - t0
        else:
            from app.agents.router import RouterAgent
            agent = RouterAgent()
            t0 = time.time()
            agent.classify_intent("Help me improve my resume")
            return time.time() - t0

    def _bench_interview_generate(self) -> float:
        if self.mode == "mock":
            with patch("app.agents.interview.ChatOpenAI"):
                from app.agents.interview import InterviewAgent
                agent = InterviewAgent()
                mock_chain = _make_mock_chain(_get_mock_interview_response())
                _patch_prompt_pipe(agent, "_question_prompt", mock_chain)
                t0 = time.time()
                agent.generate_questions("Backend Engineer", "senior", 2)
                return time.time() - t0
        else:
            from app.agents.interview import InterviewAgent
            agent = InterviewAgent()
            t0 = time.time()
            agent.generate_questions("Backend Engineer", "senior", 2)
            return time.time() - t0

    def _bench_knowledge_answer(self) -> float:
        if self.mode == "mock":
            with patch("app.agents.knowledge.ChatOpenAI"), \
                 patch("app.agents.knowledge.query_knowledge_base", return_value=[
                     {"content": "Research market rates.", "source": "salary_negotiation.md", "score": 0.9}
                 ]), \
                 patch("app.agents.knowledge.get_formatted_context",
                       return_value="Context: Research market rates before negotiating."):
                from app.agents.knowledge import KnowledgeAgent
                agent = KnowledgeAgent()
                mock_chain = _make_mock_chain(_get_mock_knowledge_response())
                agent.chain = mock_chain  # pre-built chain; replace directly
                t0 = time.time()
                agent.answer_question("How do I negotiate salary?")
                return time.time() - t0
        else:
            from app.agents.knowledge import KnowledgeAgent
            agent = KnowledgeAgent()
            t0 = time.time()
            agent.answer_question("How do I negotiate salary?")
            return time.time() - t0


# ------------------------------------------------------------------ #
#  Report writer
# ------------------------------------------------------------------ #

def write_benchmark_report(suite: BenchmarkSuite, path: str | Path) -> Path:
    """Write benchmark results to JSON and Markdown."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = p.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(suite.to_dict(), fh, indent=2)

    # Markdown
    md_path = p.with_suffix(".md")
    lines = [
        f"# Performance Benchmark Report\n",
        f"*{suite.timestamp}* | Mode: **{suite.mode}**\n",
        "| Operation | Mean (ms) | Median (ms) | P95 (ms) | Min (ms) | Max (ms) |",
        "|-----------|-----------|-------------|----------|----------|----------|",
    ]
    for r in suite.results:
        d = r.to_dict()
        lines.append(
            f"| {d['operation']} | {d['mean_ms']} | {d['median_ms']} | "
            f"{d['p95_ms']} | {d['min_ms']} | {d['max_ms']} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("*Generated by AI Job Application Coach Benchmark Runner*\n")

    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    return md_path


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agent Performance Benchmarks")
    parser.add_argument("--mock", action="store_true", default=True,
                        help="Run with mocked LLM (default)")
    parser.add_argument("--live", action="store_true",
                        help="Run with real LLM calls (requires OPENAI_API_KEY)")
    parser.add_argument("--iterations", "-n", type=int, default=5)
    parser.add_argument("--output", "-o", type=str, default="evaluation/reports/benchmarks")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    mode = "live" if args.live else "mock"
    print(f"\nRunning {mode} benchmarks ({args.iterations} iterations)...\n")

    runner = BenchmarkRunner(iterations=args.iterations, mode=mode)
    suite = runner.run_all()

    report_path = write_benchmark_report(suite, args.output)
    print(f"\nReports written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
