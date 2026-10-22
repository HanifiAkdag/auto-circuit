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


def bootstrap_tost_p(draws: np.ndarray, delta: float) -> float:
    """Invert bootstrap tails at the two equivalence bounds.

    The maximum one-sided tail probability is the TOST p-value. At alpha=0.05,
    rejecting non-equivalence is equivalent to the 90% percentile interval lying
    entirely inside ``[-delta, delta]``.
    """
    draws = np.asarray(draws, float)
    draws = draws[np.isfinite(draws)]
    if not draws.size:
        return float("nan")
    n = draws.size
    lower = (np.count_nonzero(draws <= -delta) + 1) / (n + 1)
    upper = (np.count_nonzero(draws >= delta) + 1) / (n + 1)
    return float(max(lower, upper))


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

    # Complementary allocations belong to the same fixed-size null only when
    # both groups have equal size. Otherwise enumerate every C(n, n_a) allocation.
    balanced = n_a == n - n_a
    n_assign = comb(n - 1, n_a - 1) if balanced else comb(n, n_a)
    if n_assign <= MAX_EXACT:
        count = 0
        picks = (combinations(range(1, n), n_a - 1) if balanced
                 else combinations(range(n), n_a))
        for pick in picks:
            members = np.zeros(n, dtype=bool)
            if balanced:
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


def _similarity_gap(similarity: np.ndarray, n_a: int) -> float:
    """Mean within-arm similarity minus mean between-arm similarity."""
    similarity = np.asarray(similarity, float)
    n = similarity.shape[0]
    iu = np.triu_indices(n, k=1)
    labels = np.arange(n) < n_a
    same = labels[iu[0]] == labels[iu[1]]
    values = similarity[iu]
    if not same.any() or same.all():
        return float("nan")
    return float(values[same].mean() - values[~same].mean())


def _resampled_similarity_gap(
    similarity: np.ndarray, selected: np.ndarray, n_a: int
) -> float:
    """Evaluate a seed bootstrap draw without counting duplicate seeds as pairs."""
    sampled = similarity[np.ix_(selected, selected)]
    iu = np.triu_indices(selected.size, k=1)
    distinct = selected[iu[0]] != selected[iu[1]]
    labels = np.arange(selected.size) < n_a
    same = labels[iu[0]] == labels[iu[1]]
    values = sampled[iu]
    within = values[distinct & same]
    between = values[distinct & ~same]
    if not within.size or not between.size:
        return float("nan")
    return float(within.mean() - between.mean())


def bootstrap_similarity_gap_ci(
    similarity: np.ndarray, n_a: int, level: float = 0.90,
    n_boot: int = 10_000, seed: int = 0,
) -> Tuple[float, float, float]:
    """Bootstrap the within-minus-between gap by resampling seeds within each arm."""
    similarity = np.asarray(similarity, float)
    n = similarity.shape[0]
    if not 1 < n_a < n - 1:
        return float("nan"), float("nan"), float("nan")

    draws = _bootstrap_similarity_gap_draws(similarity, n_a, n_boot, seed)
    tail = (1.0 - level) / 2.0
    low, high = np.nanquantile(draws, [tail, 1.0 - tail])
    return _similarity_gap(similarity, n_a), float(low), float(high)


def _bootstrap_similarity_gap_draws(
    similarity: np.ndarray, n_a: int, n_boot: int, seed: int,
) -> np.ndarray:
    """Generate seed-level bootstrap draws for a structural similarity contrast."""
    n = similarity.shape[0]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx_a = rng.choice(n_a, n_a, replace=True)
        idx_b = rng.choice(np.arange(n_a, n), n - n_a, replace=True)
        selected = np.concatenate([idx_a, idx_b])
        draws[i] = _resampled_similarity_gap(similarity, selected, n_a)
    return draws


def summarise_similarity_endpoint(
    similarity: np.ndarray, n_a: int, delta: float, seed: int = 0,
) -> Dict[str, object]:
    """Difference and equivalence report for a seed-dependent similarity matrix."""
    test = seed_label_permutation_test(similarity, n_a=n_a, seed=seed)
    point, low, high = bootstrap_similarity_gap_ci(similarity, n_a=n_a, seed=seed)
    draws = _bootstrap_similarity_gap_draws(
        np.asarray(similarity, float), n_a, n_boot=10_000, seed=seed
    )

    equivalent = tost_equivalent(low, high, delta)
    p_tost = bootstrap_tost_p(draws, delta)
    return {
        **within_between_means(similarity, n_a=n_a),
        "n_a": n_a,
        "n_b": similarity.shape[0] - n_a,
        "n_permutations": test["n_permutations"],
        "gap_within_minus_between": point,
        "ci90_low": low,
        "ci90_high": high,
        "delta": delta,
        "equivalent": equivalent,
        "p_tost": p_tost,
        "p_permutation": test["p_value"],
        "exact": bool(test["exact"]),
        "reading": interpret(test["p_value"], equivalent),
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


def apply_holm_to_reports(reports: Dict[str, dict]) -> Dict[str, dict]:
    """Correct difference and equivalence tests over one endpoint family."""
    diff = holm({key: row["p_permutation"] for key, row in reports.items()})
    tost = holm({key: row["p_tost"] for key, row in reports.items()})
    out: Dict[str, dict] = {}
    for key, row in reports.items():
        corrected = dict(row)
        corrected["p_difference_holm"] = diff[key]["p_holm"]
        corrected["p_tost_holm"] = tost[key]["p_holm"]
        corrected["equivalent"] = bool(tost[key]["reject"])
        corrected["reading"] = interpret(
            corrected["p_difference_holm"], corrected["equivalent"]
        )
        out[key] = corrected
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
    rng = np.random.default_rng(seed)
    a_arr, b_arr = np.asarray(a, float), np.asarray(b, float)
    draws = np.empty(10_000)
    for i in range(draws.size):
        draws[i] = (rng.choice(a_arr, a_arr.size, replace=True).mean()
                    - rng.choice(b_arr, b_arr.size, replace=True).mean())
    point = float(a_arr.mean() - b_arr.mean())
    low, high = np.quantile(draws, [0.05, 0.95])
    equivalent = tost_equivalent(low, high, delta)
    return {
        "n_a": len(a), "n_b": len(b),
        "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
        "sd_a": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "sd_b": float(np.std(b, ddof=1)) if len(b) > 1 else 0.0,
        "diff": point, "ci90_low": float(low), "ci90_high": float(high),
        "delta": delta, "equivalent": equivalent,
        "p_tost": bootstrap_tost_p(draws, delta),
        "p_permutation": perm["p_value"], "exact": bool(perm["exact"]),
        "n_permutations": perm["n_permutations"],
        "reading": interpret(perm["p_value"], equivalent),
    }


def control_outcomes(report: dict) -> Dict[str, bool]:
    """Separate demonstrated detection from equivalence and an unresolved result."""
    p_difference = report.get("p_difference_holm", report["p_permutation"])
    difference = bool(p_difference < 0.05)
    equivalent = bool(report["equivalent"])
    return {
        "detected": difference,
        "difference_detected": difference,
        "equivalence_established": equivalent,
        "inconclusive": not difference and not equivalent,
    }


def detected(control: Sequence[float], reference: Sequence[float], delta: float,
             seed: int = 0) -> bool:
    """Did the difference test detect a negative control at alpha=0.05?"""
    report = summarise_endpoint(control, reference, delta, seed=seed)
    return control_outcomes(report)["detected"]


def pairwise_matrix(vectors: List[np.ndarray], metric) -> np.ndarray:
    """Symmetric pairwise matrix from a list of per-seed vectors."""
    n = len(vectors)
    out = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            out[i, j] = out[j, i] = metric(vectors[i], vectors[j])
    return out
