"""
Reporters — generate evaluation reports in JSON (machine-readable) and
Markdown (human-readable) formats.

Usage::

    from evaluation.evaluator import Evaluator
    from evaluation.reporters import JSONReporter, MarkdownReporter

    ev = Evaluator()
    all_summaries = ev.run_all()

    JSONReporter().write(all_summaries, "evaluation/reports/latest.json")
    MarkdownReporter().write(all_summaries, "evaluation/reports/latest.md")
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from evaluation.scoring import AgentEvalSummary

# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def _ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------ #
#  JSON Reporter
# ------------------------------------------------------------------ #

class JSONReporter:
    """Writes evaluation results as a structured JSON file."""

    def write(
        self,
        summaries: Dict[str, AgentEvalSummary],
        path: str | Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Serialise *summaries* to *path* and return the resolved path.

        Parameters
        ----------
        summaries : dict[str, AgentEvalSummary]
            Output from ``Evaluator.run_all()``.
        path : str | Path
            Destination file.
        metadata : dict | None
            Optional extra info (git sha, config, etc.).
        """
        out = _ensure_parent(path)

        report: Dict[str, Any] = {
            "generated_at": _timestamp(),
            "metadata": metadata or {},
            "summary": {},
            "agents": {},
        }

        # Aggregate summary
        total = passed = failed = errors = 0
        for s in summaries.values():
            total += s.total_cases
            passed += s.passed_cases
            failed += s.failed_cases
            errors += s.error_cases

        report["summary"] = {
            "total_agents": len(summaries),
            "total_cases": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "pass_rate": round(passed / max(total, 1) * 100, 1),
        }

        # Per-agent detail
        for name, s in summaries.items():
            report["agents"][name] = s.to_dict()

        with open(out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)

        return out


# ------------------------------------------------------------------ #
#  Markdown Reporter
# ------------------------------------------------------------------ #

class MarkdownReporter:
    """Writes evaluation results as a human-readable Markdown file."""

    def write(
        self,
        summaries: Dict[str, AgentEvalSummary],
        path: str | Path,
        title: str = "Evaluation Report",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        out = _ensure_parent(path)
        lines: list[str] = []
        _a = lines.append

        _a(f"# {title}\n")
        _a(f"*Generated: {_timestamp()}*\n")

        if metadata:
            _a("## Metadata\n")
            for k, v in metadata.items():
                _a(f"- **{k}**: {v}")
            _a("")

        # ------ Overall summary ----------------------------------------
        total = passed = failed = errors = 0
        for s in summaries.values():
            total += s.total_cases
            passed += s.passed_cases
            failed += s.failed_cases
            errors += s.error_cases
        pass_rate = round(passed / max(total, 1) * 100, 1)

        _a("## Overall Summary\n")
        _a(f"| Metric | Value |")
        _a(f"|--------|-------|")
        _a(f"| Total agents evaluated | {len(summaries)} |")
        _a(f"| Total test cases | {total} |")
        _a(f"| Passed | {passed} |")
        _a(f"| Failed | {failed} |")
        _a(f"| Errors | {errors} |")
        _a(f"| **Pass rate** | **{pass_rate}%** |")
        _a("")

        # ------ Per-agent breakdown ------------------------------------
        _a("## Agent Breakdown\n")
        _a("| Agent | Cases | Passed | Failed | Errors | Pass Rate | Avg Score |")
        _a("|-------|-------|--------|--------|--------|-----------|-----------|")
        for name, s in summaries.items():
            _a(
                f"| {name} | {s.total_cases} | {s.passed_cases} | "
                f"{s.failed_cases} | {s.error_cases} | "
                f"{s.pass_rate:.1f}% | {s.avg_overall_score:.2f} |"
            )
        _a("")

        # ------ Per-agent detailed results -----------------------------
        for name, s in summaries.items():
            _a(f"## {name.replace('_', ' ').title()} Agent\n")

            # Dimension averages
            if s.avg_dimension_scores:
                _a("### Dimension Averages\n")
                _a("| Dimension | Avg Score |")
                _a("|-----------|-----------|")
                for dim, avg in s.avg_dimension_scores.items():
                    _a(f"| {dim} | {avg:.2f} |")
                _a("")

            # Failed / error cases detail
            problem_results = [r for r in s.results if not r.passed or r.error]
            if problem_results:
                _a("### Issues\n")
                for r in problem_results:
                    status = "ERROR" if r.error else "FAILED"
                    _a(f"- **{r.test_id}** [{status}] — score {r.overall_score:.2f}")
                    if r.error:
                        _a(f"  - Error: `{r.error}`")
                    for d in r.dimension_scores:
                        if not d.passed:
                            _a(f"  - {d.name}: {d.score:.2f} (min: needed to pass)")
                _a("")

        # ------ Footer -------------------------------------------------
        _a("---")
        _a(f"*Report generated by AI Job Application Coach Evaluation Framework — Phase 5*")
        _a("")

        with open(out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

        return out


# ------------------------------------------------------------------ #
#  Console reporter (for CLI usage)
# ------------------------------------------------------------------ #

class ConsoleReporter:
    """Print evaluation results to stdout."""

    def report(self, summaries: Dict[str, AgentEvalSummary]) -> None:
        total = sum(s.total_cases for s in summaries.values())
        passed = sum(s.passed_cases for s in summaries.values())
        pr = round(passed / max(total, 1) * 100, 1)

        print(f"\n{'='*60}")
        print(f"  EVALUATION REPORT — {_timestamp()}")
        print(f"{'='*60}")
        print(f"  Total cases: {total}  |  Passed: {passed}  |  Pass rate: {pr}%\n")

        for name, s in summaries.items():
            icon = "✓" if s.pass_rate == 100 else "✗"
            print(f"  {icon} {name:15s}  {s.passed_cases}/{s.total_cases} passed "
                  f"({s.pass_rate:.0f}%)  avg={s.avg_overall_score:.2f}")

            if s.avg_dimension_scores:
                dims = "  ".join(f"{k}={v:.1f}" for k, v in s.avg_dimension_scores.items())
                print(f"    dims: {dims}")

            errors = [r for r in s.results if r.error]
            for r in errors[:3]:
                print(f"    ! {r.test_id}: {r.error}")

        print(f"\n{'='*60}\n")
