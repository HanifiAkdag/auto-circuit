"""Statistics for the port fidelity study.

Pure numpy, no torch and no I/O, so the inference can be unit-tested independently
of the experiment that feeds it.

Two things the design turns on:

**Equivalence is not the absence of a difference.** A non-significant difference
test is consistent with an underpowered study, so it cannot support a claim of
fidelity. The positive claim comes from TOST: the confidence interval for the
shift must lie entirely inside a pre-registered margin. Both are reported.

**Pairwise similarities are not independent observations.** Each seed appears in
many pairs, so a test that permutes *pairs* has the wrong null. The seed-label
permutation below permutes seeds and recomputes the within/between split.
"""

from itertools import combinations
from math import comb
from typing import Dict, List, Sequence, Tuple

import numpy as np

MAX_EXACT = 400_000  # enumerate below this many label assignments, else sample


def exact_permutation_test(
    a: Sequence[float], b: Sequence[float], n_mc: int = 100_000, seed: int = 0
) -> Dict[str, float]:
    """Two-sided permutation test on the difference of means.

    With 10 observations per arm there are ``C(20, 10) = 184,756`` assignments, so
    the test is enumerated exactly and the p-value has no Monte Carlo error.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.concatenate([a, b])
    n, n_a = pooled.size, a.size
    observed = a.mean() - b.mean()
    total = pooled.sum()

    n_assign = comb(n, n_a)
    if n_assign <= MAX_EXACT:
        idx = np.arange(n)
        count = 0
        for pick in combinations(idx, n_a):
            s = pooled[list(pick)].sum()
            diff = s / n_a - (total - s) / (n - n_a)
            count += abs(diff) >= abs(observed) - 1e-12
        return {"observed": float(observed), "p_value": count / n_assign,
                "n_permutations": float(n_assign), "exact": 1.0}

    rng = np.random.default_rng(seed)
    count = 1  # include the observed assignment
    for _ in range(n_mc):
        perm = rng.permutation(pooled)
        diff = perm[:n_a].mean() - perm[n_a:].mean()
        count += abs(diff) >= abs(observed) - 1e-12
    return {"observed": float(observed), "p_value": count / (n_mc + 1),
            "n_permutations": float(n_mc), "exact": 0.0}


def bootstrap_diff_ci(
    a: Sequence[float], b: Sequence[float], level: float = 0.90,
    n_boot: int = 10_000, seed: int = 0,
) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for ``mean(a) - mean(b)``, resampling **seeds**.

    Returns ``(point_estimate, low, high)``.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        draws[i] = (rng.choice(a, a.size, replace=True).mean()
                    - rng.choice(b, b.size, replace=True).mean())
    tail = (1.0 - level) / 2.0
    return float(a.mean() - b.mean()), *np.quantile(draws, [tail, 1.0 - tail])


def tost_equivalent(low: float, high: float, delta: float) -> bool:
    """TOST at alpha=0.05 via a 90% CI: equivalence iff the CI sits inside +/-delta."""
    return bool(low > -delta and high < delta)


def seed_label_permutation_test(
    similarity: np.ndarray, n_a: int, n_mc: int = 100_000, seed: int = 0
) -> Dict[str, float]:
    """Test whether within-arm similarity exceeds between-arm similarity.

    ``similarity`` is the full symmetric pairwise matrix over ``n_a + n_b`` seeds,
    the first ``n_a`` rows belonging to arm A. The statistic is
    ``mean(within-arm pairs) - mean(between-arm pairs)``.

    Labels are permuted over seeds, not over pairs: a pair-level permutation would
    treat the ``C(n, 2)`` similarities as independent when each seed contributes to
    ``n - 1`` of them.
    """
    similarity = np.asarray(similarity, float)
    n = similarity.shape[0]
    assert similarity.shape == (n, n), "similarity must be square"
    iu = np.triu_indices(n, k=1)
    values = similarity[iu]

    def statistic(members: np.ndarray) -> float:
        same = members[iu[0]] == members[iu[1]]
        # Undefined when either group of pairs is empty, e.g. a single seed per arm.
        if not same.any() or same.all():
            return np.nan
        return values[same].mean() - values[~same].mean()

    labels = np.zeros(n, dtype=bool)
    labels[:n_a] = True
    observed = statistic(labels)

    # The statistic is invariant under swapping the two arm names, so enumerating
    # assignments that contain seed 0 covers every distinct split exactly once.
    n_assign = comb(n - 1, n_a - 1)
    if n_assign <= MAX_EXACT:
        count = 0
        rest = np.arange(1, n)
        for pick in combinations(rest, n_a - 1):
            members = np.zeros(n, dtype=bool)
            members[0] = True
            members[list(pick)] = True
            count += statistic(members) >= observed - 1e-12
        return {"observed": float(observed), "p_value": count / n_assign,
                "n_permutations": float(n_assign), "exact": 1.0}

    rng = np.random.default_rng(seed)
    count = 1
    for _ in range(n_mc):
        members = np.zeros(n, dtype=bool)
        members[rng.choice(n, n_a, replace=False)] = True
        count += statistic(members) >= observed - 1e-12
    return {"observed": float(observed), "p_value": count / (n_mc + 1),
            "n_permutations": float(n_mc), "exact": 0.0}


def within_between_means(similarity: np.ndarray, n_a: int) -> Dict[str, float]:
    """Mean similarity within arm A, within arm B, and across arms."""
    similarity = np.asarray(similarity, float)
    n = similarity.shape[0]
    iu = np.triu_indices(n, k=1)
    in_a = iu[0] < n_a
    in_b = iu[1] >= n_a
    values = similarity[iu]

    def mean_or_nan(selected: np.ndarray) -> float:
        # An arm with a single seed contributes no within-arm pairs.
        return float(selected.mean()) if selected.size else float("nan")

    return {
        "within_a": mean_or_nan(values[in_a & ~in_b]),
        "within_b": mean_or_nan(values[~in_a & in_b]),
        "between": mean_or_nan(values[in_a & in_b]),
    }


def holm(p_values: Dict[str, float], alpha: float = 0.05) -> Dict[str, dict]:
    """Holm-Bonferroni step-down correction over a family of tests."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out: Dict[str, dict] = {}
    previous = 0.0
    for i, (key, p) in enumerate(ordered):
        adjusted = max(previous, min(1.0, (m - i) * p))
        previous = adjusted
        out[key] = {"p_raw": p, "p_holm": adjusted, "reject": adjusted < alpha}
    return out


def interpret(diff_p: float, equivalent: bool) -> str:
    """The reading grid fixed in the pre-registration.

    The (not significant, not equivalent) cell is the one that must never be
    reported as support: it means the study lacked the resolution to decide.
    """
    significant = diff_p < 0.05
    if not significant and equivalent:
        return "fidelity supported"
    if not significant and not equivalent:
        return "inconclusive (underpowered)"
    if significant and equivalent:
        return "real but negligible shift"
    return "genuine discrepancy"


def summarise_endpoint(
    a: Sequence[float], b: Sequence[float], delta: float, seed: int = 0
) -> Dict[str, object]:
    """Full report for one scalar endpoint: difference test plus equivalence test."""
    perm = exact_permutation_test(a, b, seed=seed)
    point, low, high = bootstrap_diff_ci(a, b, level=0.90, seed=seed)
    equivalent = tost_equivalent(low, high, delta)
    return {
        "n_a": len(a), "n_b": len(b),
        "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
        "sd_a": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "sd_b": float(np.std(b, ddof=1)) if len(b) > 1 else 0.0,
        "diff": point, "ci90_low": float(low), "ci90_high": float(high),
        "delta": delta, "equivalent": equivalent,
        "p_permutation": perm["p_value"], "exact": bool(perm["exact"]),
        "reading": interpret(perm["p_value"], equivalent),
    }


def detected(control: Sequence[float], reference: Sequence[float], delta: float,
             seed: int = 0) -> bool:
    """Did an endpoint flag a negative control? Used to evidence achieved power."""
    report = summarise_endpoint(control, reference, delta, seed=seed)
    return bool(report["p_permutation"] < 0.05 or not report["equivalent"])


def pairwise_matrix(vectors: List[np.ndarray], metric) -> np.ndarray:
    """Symmetric pairwise matrix from a list of per-seed vectors."""
    n = len(vectors)
    out = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            out[i, j] = out[j, i] = metric(vectors[i], vectors[j])
    return out
