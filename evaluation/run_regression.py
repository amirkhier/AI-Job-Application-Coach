"""
Run prompt regression suite — validates all prompt versions and reports results.

Usage::

    python -m evaluation.run_regression
    python -m evaluation.run_regression --verbose
"""

from __future__ import annotations

import argparse
import sys

from evaluation.prompt_manager import PromptManager


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prompt Regression Checker")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    pm = PromptManager()
    all_issues = pm.validate_all()

    total = len(all_issues)
    failed = sum(1 for v in all_issues.values() if v)

    print(f"\nPrompt Regression Check — {total} files scanned\n")

    for key, issues in sorted(all_issues.items()):
        status = "FAIL" if issues else "OK"
        print(f"  [{status}] {key}")
        if issues and args.verbose:
            for issue in issues:
                print(f"        ! {issue}")

    print(f"\n  Result: {total - failed}/{total} passed\n")

    if failed:
        print("  Some prompt files have issues — see above.\n")
        return 1

    print("  All prompt files are well-formed.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
