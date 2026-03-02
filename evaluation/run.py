"""
CLI entry-point for running evaluations.

Usage
-----
::

    # Evaluate all agents (dry-run — no LLM calls)
    python -m evaluation.run --dry-run

    # Evaluate a specific agent
    python -m evaluation.run --agent resume

    # Full evaluation with reports
    python -m evaluation.run --report-dir evaluation/reports

    # Filter by tags
    python -m evaluation.run --include-tags positive --exclude-tags adversarial
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from evaluation.evaluator import Evaluator
from evaluation.reporters import JSONReporter, MarkdownReporter, ConsoleReporter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="AI Job Application Coach — Evaluation Runner",
    )
    parser.add_argument(
        "--agent", "-a",
        type=str,
        default=None,
        help="Evaluate a single agent (resume, interview, knowledge, job_search, router). "
             "Omit to run all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip LLM calls — validate dataset loading, scoring logic, and reporting.",
    )
    parser.add_argument(
        "--report-dir", "-o",
        type=str,
        default=None,
        help="Directory for JSON and Markdown reports. If omitted, only console output is produced.",
    )
    parser.add_argument(
        "--include-tags",
        nargs="*",
        default=None,
        help="Only run test cases whose tags include at least one of these.",
    )
    parser.add_argument(
        "--exclude-tags",
        nargs="*",
        default=None,
        help="Skip test cases with any of these tags.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )

    args = parser.parse_args(argv)

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    agents = [args.agent] if args.agent else None

    evaluator = Evaluator(
        agents=agents,
        tags_include=args.include_tags,
        tags_exclude=args.exclude_tags,
        dry_run=args.dry_run,
    )

    # Run
    summaries = evaluator.run_all()

    # Console output
    ConsoleReporter().report(summaries)

    # File reports
    if args.report_dir:
        rd = Path(args.report_dir)
        json_path = rd / "latest.json"
        md_path = rd / "latest.md"

        metadata = {
            "dry_run": args.dry_run,
            "include_tags": args.include_tags,
            "exclude_tags": args.exclude_tags,
        }

        JSONReporter().write(summaries, json_path, metadata=metadata)
        MarkdownReporter().write(summaries, md_path, metadata=metadata)

        logging.getLogger(__name__).info(
            "Reports written to %s and %s", json_path, md_path,
        )

    # Exit code: 0 if all passed, 1 otherwise
    total = sum(s.total_cases for s in summaries.values())
    passed = sum(s.passed_cases for s in summaries.values())
    return 0 if total > 0 and total == passed else 1


if __name__ == "__main__":
    sys.exit(main())
