"""
Statistical utilities for A/B experiment analysis.

Provides:
- Paired t-test for comparing two sets of scores
- Bootstrap confidence intervals for mean difference
- Cohen's d effect size

Uses pure Python + math stdlib to avoid a hard scipy dependency; falls
back gracefully when scipy is not installed.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class ExperimentStats:
    """Statistical summary for an A/B experiment."""

    mean_a: float
    mean_b: float
    t_statistic: float
    p_value: float
    ci_lower: float         # 95% CI lower bound on (mean_b - mean_a)
    ci_upper: float         # 95% CI upper bound
    effect_size: float      # Cohen's d
    n: int
    significant: bool

    @property
    def mean_diff(self) -> float:
        return self.mean_b - self.mean_a

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_a": round(self.mean_a, 4),
            "mean_b": round(self.mean_b, 4),
            "mean_diff": round(self.mean_diff, 4),
            "t_statistic": round(self.t_statistic, 4),
            "p_value": round(self.p_value, 6),
            "ci_95": [round(self.ci_lower, 4), round(self.ci_upper, 4)],
            "effect_size_cohens_d": round(self.effect_size, 4),
            "n": self.n,
            "significant": self.significant,
        }


# ------------------------------------------------------------------ #
#  Paired t-test
# ------------------------------------------------------------------ #

def paired_t_test(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Paired two-tailed t-test.

    Returns ``(t_statistic, p_value)``.

    Falls back to a pure-Python implementation if scipy is not available.
    """
    n = len(a)
    if n != len(b):
        raise ValueError("Sample sizes must match")
    if n < 2:
        return (0.0, 1.0)

    # Check if differences have zero variance (identical pairs)
    diffs = [ai - bi for ai, bi in zip(a, b)]
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    if var_d < 1e-15:
        # No variance — samples have identical differences
        if abs(mean_d) < 1e-15:
            return (0.0, 1.0)  # truly identical
        else:
            # All diffs are the same non-zero value — "infinitely" significant
            return (float("inf") if mean_d > 0 else float("-inf"), 0.0)

    try:
        from scipy.stats import ttest_rel
        result = ttest_rel(a, b)
        t_val = float(result.statistic)
        p_val = float(result.pvalue)
        # Guard against scipy returning nan
        if math.isnan(t_val) or math.isnan(p_val):
            raise ValueError("scipy returned nan")
        return (t_val, p_val)
    except (ImportError, ValueError):
        pass

    # Pure-Python fallback
    se = math.sqrt(var_d / n)
    t_stat = mean_d / se

    # Approximate p-value using the normal distribution for large n,
    # or a rough t-distribution approximation
    p_value = _approx_two_tailed_p(t_stat, n - 1)
    return (t_stat, p_value)


def _approx_two_tailed_p(t: float, df: int) -> float:
    """Rough two-tailed p-value from t-statistic and degrees of freedom.

    Uses the normal CDF approximation for large df.  For small df, applies
    a correction factor.  This is intentionally simple — install scipy for
    accurate results.
    """
    abs_t = abs(t)
    # Normal CDF approximation (Abramowitz & Stegun)
    p_one_tail = 0.5 * math.erfc(abs_t / math.sqrt(2))
    p_two_tail = 2 * p_one_tail
    # Rough correction for small degrees of freedom
    if df < 30:
        correction = 1 + (1 / (4 * max(df, 1)))
        p_two_tail = min(1.0, p_two_tail * correction)
    return max(0.0, min(1.0, p_two_tail))


# ------------------------------------------------------------------ #
#  Bootstrap confidence interval
# ------------------------------------------------------------------ #

def bootstrap_confidence_interval(
    a: List[float],
    b: List[float],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Bootstrap 95% CI for the mean difference (b - a).

    Returns ``(lower, upper)```.
    """
    n = len(a)
    if n < 2:
        return (0.0, 0.0)

    rng = random.Random(seed)
    diffs: List[float] = []
    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        sample_a = [a[i] for i in indices]
        sample_b = [b[i] for i in indices]
        mean_diff = (sum(sample_b) / n) - (sum(sample_a) / n)
        diffs.append(mean_diff)

    diffs.sort()
    alpha = 1 - ci
    lower_idx = int(alpha / 2 * n_bootstrap)
    upper_idx = int((1 - alpha / 2) * n_bootstrap) - 1
    return (diffs[lower_idx], diffs[upper_idx])


# ------------------------------------------------------------------ #
#  Effect size
# ------------------------------------------------------------------ #

def cohens_d(a: List[float], b: List[float]) -> float:
    """Cohen's d effect size for paired samples."""
    n = len(a)
    if n < 2:
        return 0.0

    diffs = [bi - ai for ai, bi in zip(a, b)]
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    sd = math.sqrt(var_d) if var_d > 0 else 1e-9
    return mean_d / sd
