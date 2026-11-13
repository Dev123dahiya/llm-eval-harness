"""Is the difference real? Confidence intervals and paired significance tests."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .runner import Run


def bootstrap_ci(values: list[float], confidence: float = 0.95, resamples: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean.

    An eval score is an estimate from a sample of cases. With 50 cases, "82%"
    can easily mean anything from 70% to 91%; the interval says so.
    """
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(resamples))
    lo = means[int((1 - confidence) / 2 * resamples)]
    hi = means[min(int((1 + confidence) / 2 * resamples), resamples - 1)]
    return (lo, hi)


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar test for paired pass/fail outcomes.

    Only cases where the two systems disagree carry information. Under the null
    hypothesis each disagreement is a fair coin flip, so the p-value is a
    binomial tail. Works for small samples, unlike the chi-squared version.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


@dataclass
class Comparison:
    scorer: str
    baseline_mean: float
    candidate_mean: float
    fixed: list[str]       # failed in baseline, pass in candidate
    broken: list[str]      # passed in baseline, fail in candidate
    p_value: float
    diff_ci: tuple[float, float]

    @property
    def delta(self) -> float:
        return self.candidate_mean - self.baseline_mean

    def is_regression(self, alpha: float = 0.05) -> bool:
        """Worse, and unlikely to be noise."""
        return self.delta < 0 and self.p_value < alpha


def compare_runs(baseline: Run, candidate: Run, scorer: str, seed: int = 0) -> Comparison:
    """Paired comparison on the cases both runs share.

    Pairing matters: both systems saw the same cases, so comparing case by case
    removes the variance that comes from some cases simply being harder.
    """
    base = {r.case_id: r.scores[scorer] for r in baseline.results}
    cand = {r.case_id: r.scores[scorer] for r in candidate.results}
    shared = [i for i in base if i in cand]
    if not shared:
        raise ValueError("the runs share no cases")
    fixed = [i for i in shared if base[i] < 1.0 <= cand[i]]
    broken = [i for i in shared if cand[i] < 1.0 <= base[i]]
    diffs = [cand[i] - base[i] for i in shared]
    return Comparison(scorer, sum(base[i] for i in shared) / len(shared), sum(cand[i] for i in shared) / len(shared),
                      fixed, broken, mcnemar_exact(len(fixed), len(broken)), bootstrap_ci(diffs, seed=seed))
