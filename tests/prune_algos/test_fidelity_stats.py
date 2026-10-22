"""Tests for the port fidelity inference.

The statistics decide whether the thesis can claim the port is faithful, so they
are tested against cases with known answers rather than trusted by inspection.
"""

import sys
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.port_comparison import fidelity_stats  # noqa: E402
from experiments.port_comparison.fidelity_stats import (  # noqa: E402
    apply_holm_to_reports,
    bootstrap_similarity_gap_ci,
    bootstrap_diff_ci,
    detected,
    exact_permutation_test,
    holm,
    interpret,
    pairwise_matrix,
    seed_label_permutation_test,
    summarise_endpoint,
    summarise_similarity_endpoint,
    tost_equivalent,
    within_between_means,
)


# --- permutation test --------------------------------------------------------


def test_permutation_is_exact_at_the_study_size():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=10), rng.normal(size=10)
    res = exact_permutation_test(a, b)
    assert res["exact"] == 1.0
    assert res["n_permutations"] == comb(20, 10) == 184756


def test_permutation_detects_a_large_shift():
    a = np.arange(10, dtype=float)
    b = np.arange(10, dtype=float) + 50
    assert exact_permutation_test(a, b)["p_value"] < 0.001


def test_permutation_finds_nothing_in_identical_arms():
    a = np.arange(10, dtype=float)
    assert exact_permutation_test(a, a.copy())["p_value"] == 1.0


def test_permutation_is_symmetric_in_its_arguments():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=8), rng.normal(loc=0.4, size=8)
    assert exact_permutation_test(a, b)["p_value"] == pytest.approx(
        exact_permutation_test(b, a)["p_value"]
    )


def test_permutation_p_value_is_uniform_under_the_null():
    """A valid test rejects at roughly the nominal rate when nothing differs."""
    rng = np.random.default_rng(11)
    ps = [
        exact_permutation_test(rng.normal(size=6), rng.normal(size=6))["p_value"]
        for _ in range(200)
    ]
    assert 0.0 <= np.mean(np.array(ps) < 0.05) <= 0.12


# --- equivalence -------------------------------------------------------------


def test_tost_requires_the_whole_interval_inside_the_margin():
    assert tost_equivalent(-0.4, 0.4, delta=0.5)
    assert not tost_equivalent(-0.6, 0.4, delta=0.5)
    assert not tost_equivalent(-0.4, 0.6, delta=0.5)


def test_bootstrap_ci_brackets_a_known_difference():
    rng = np.random.default_rng(5)
    a = rng.normal(loc=5.0, scale=0.1, size=10)
    b = rng.normal(loc=3.0, scale=0.1, size=10)
    point, low, high = bootstrap_diff_ci(a, b)
    assert low < point < high
    assert 1.8 < point < 2.2
    assert not tost_equivalent(low, high, delta=0.5)


def test_tight_arms_are_declared_equivalent():
    rng = np.random.default_rng(7)
    a = rng.normal(loc=10.0, scale=0.05, size=10)
    b = rng.normal(loc=10.0, scale=0.05, size=10)
    _, low, high = bootstrap_diff_ci(a, b)
    assert tost_equivalent(low, high, delta=1.0)


def test_wide_arms_are_not_equivalent_even_with_equal_means():
    """The trap the pre-registration guards against: a non-significant difference
    test on noisy data must not be read as evidence of fidelity."""
    rng = np.random.default_rng(9)
    a = rng.normal(loc=10.0, scale=8.0, size=10)
    b = rng.normal(loc=10.0, scale=8.0, size=10)
    report = summarise_endpoint(a, b, delta=0.5)
    assert report["p_permutation"] > 0.05
    assert not report["equivalent"]
    assert report["reading"] == "inconclusive (underpowered)"


def test_scalar_report_exposes_tost_p_value_consistent_with_interval():
    rng = np.random.default_rng(17)
    a = rng.normal(loc=0.01, scale=0.02, size=10)
    b = rng.normal(loc=0.00, scale=0.02, size=10)
    report = summarise_endpoint(a, b, delta=0.2)
    assert report["equivalent"]
    assert report["p_tost"] < 0.05
    assert report["n_permutations"] == comb(20, 10)


def test_interpretation_grid():
    assert interpret(0.9, True) == "fidelity supported"
    assert interpret(0.9, False) == "inconclusive (underpowered)"
    assert interpret(0.001, True) == "real but negligible shift"
    assert interpret(0.001, False) == "genuine discrepancy"


# --- seed-label permutation --------------------------------------------------


def _exhaustive_seed_test(matrix, n_a):
    """Independent oracle: partition explicit unordered pairs for every allocation."""
    n = len(matrix)
    pairs = list(combinations(range(n), 2))

    def gap(group):
        within, between = [], []
        for i, j in pairs:
            destination = within if (i in group) == (j in group) else between
            destination.append(matrix[i, j])
        return np.average(within) - np.average(between)

    observed = gap(set(range(n_a)))
    distribution = [gap(set(group)) for group in combinations(range(n), n_a)]
    exceedances = np.count_nonzero(np.asarray(distribution) >= observed - 1e-12)
    return observed, exceedances / len(distribution)


@pytest.mark.parametrize("n_a,n_b", [(3, 3), (2, 5), (5, 2), (5, 10), (10, 5)])
def test_seed_test_agrees_with_independent_full_enumeration(n_a, n_b):
    rng = np.random.default_rng(31)
    matrix = rng.uniform(-1, 1, size=(n_a + n_b, n_a + n_b))
    matrix = (matrix + matrix.T) / 2
    observed, p_value = _exhaustive_seed_test(matrix, n_a)
    result = seed_label_permutation_test(matrix, n_a)
    assert result["observed"] == pytest.approx(observed)
    assert result["p_value"] == pytest.approx(p_value)
    allocations = comb(n_a + n_b, n_a)
    expected_count = allocations // 2 if n_a == n_b else allocations
    assert result["n_permutations"] == expected_count


@pytest.mark.parametrize("n_a,n_b", [(3, 3), (2, 5), (5, 10)])
def test_seed_test_is_invariant_to_swapping_groups(n_a, n_b):
    rng = np.random.default_rng(41)
    matrix = rng.normal(size=(n_a + n_b, n_a + n_b))
    matrix = (matrix + matrix.T) / 2
    order = list(range(n_a, n_a + n_b)) + list(range(n_a))
    result = seed_label_permutation_test(matrix, n_a)
    swapped = seed_label_permutation_test(matrix[np.ix_(order, order)], n_b)
    assert swapped["observed"] == pytest.approx(result["observed"])
    assert swapped["p_value"] == pytest.approx(result["p_value"])


def test_unequal_group_exact_cutoff_uses_all_allocations(monkeypatch):
    monkeypatch.setattr(fidelity_stats, "MAX_EXACT", 10)
    matrix = _blocked_similarity(2, 5, within=0.8, between=0.2)
    result = seed_label_permutation_test(matrix, n_a=2, n_mc=100)
    # C(7, 2)=21 exceeds the cutoff; conditioning on seed 0 would give only 6.
    assert result["exact"] == 0.0
    assert result["n_permutations"] == 100


def _blocked_similarity(n_a, n_b, within, between, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    n = n_a + n_b
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            same = (i < n_a) == (j < n_a)
            m[i, j] = m[j, i] = (within if same else between) + noise * rng.normal()
    np.fill_diagonal(m, 1.0)
    return m


def test_seed_label_permutation_is_exact_at_the_study_size():
    m = _blocked_similarity(10, 10, 0.3, 0.3)
    res = seed_label_permutation_test(m, n_a=10)
    assert res["exact"] == 1.0
    # Enumerating splits that contain seed 0 covers each distinct split once.
    assert res["n_permutations"] == comb(19, 9)


def test_similarity_report_exposes_group_sizes_and_permutation_count():
    report = summarise_similarity_endpoint(_blocked_similarity(10, 10, 0.3, 0.3),
                                           n_a=10, delta=0.1)
    assert report["n_a"] == report["n_b"] == 10
    assert report["n_permutations"] == comb(19, 9)


def test_seed_label_permutation_detects_arm_structure():
    m = _blocked_similarity(8, 8, within=0.8, between=0.2, noise=0.01, seed=1)
    res = seed_label_permutation_test(m, n_a=8)
    assert res["observed"] > 0.5
    assert res["p_value"] < 0.01


def test_seed_label_permutation_finds_nothing_without_arm_structure():
    m = _blocked_similarity(8, 8, within=0.5, between=0.5, noise=0.05, seed=2)
    assert seed_label_permutation_test(m, n_a=8)["p_value"] > 0.05


def test_within_between_means_reads_the_blocks():
    m = _blocked_similarity(5, 6, within=0.75, between=0.25)
    means = within_between_means(m, n_a=5)
    assert means["within_a"] == pytest.approx(0.75)
    assert means["within_b"] == pytest.approx(0.75)
    assert means["between"] == pytest.approx(0.25)


def test_within_between_means_is_nan_free_of_warnings_with_one_seed_per_arm():
    """A partly-failed arm must degrade to nan, not to a RuntimeWarning."""
    m = _blocked_similarity(1, 1, within=0.5, between=0.3)
    means = within_between_means(m, n_a=1)
    assert np.isnan(means["within_a"]) and np.isnan(means["within_b"])
    assert means["between"] == pytest.approx(0.3)


def test_seed_label_statistic_is_nan_when_a_group_of_pairs_is_empty():
    m = _blocked_similarity(1, 1, within=0.5, between=0.3)
    assert np.isnan(seed_label_permutation_test(m, n_a=1)["observed"])


def test_structural_bootstrap_resamples_seed_labels_not_similarity_pairs():
    m = _blocked_similarity(6, 6, within=0.52, between=0.50, noise=0.01, seed=4)
    point, low, high = bootstrap_similarity_gap_ci(m, n_a=6, seed=9)
    observed = seed_label_permutation_test(m, n_a=6)["observed"]
    assert point == pytest.approx(observed)
    assert low < point < high


def test_structural_report_requires_interval_inside_margin():
    m = _blocked_similarity(6, 6, within=0.51, between=0.50, noise=0.01, seed=8)
    report = summarise_similarity_endpoint(m, n_a=6, delta=0.2)
    assert report["equivalent"]
    assert report["p_tost"] < 0.05
    assert report["ci90_low"] > -0.2
    assert report["ci90_high"] < 0.2


def test_pairwise_matrix_is_symmetric_with_unit_diagonal():
    vectors = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([1.0, 1.0, 0])]
    m = pairwise_matrix(vectors, lambda x, y: float(np.dot(x, y)))
    assert np.allclose(m, m.T)
    assert np.allclose(np.diag(m), 1.0)


# --- multiplicity ------------------------------------------------------------


def test_holm_is_monotone_and_conservative():
    out = holm({"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.5})
    assert out["a"]["p_holm"] == pytest.approx(0.004)
    assert out["b"]["p_holm"] == pytest.approx(0.06)
    # Step-down: no adjusted p may fall below an earlier one.
    adjusted = [out[k]["p_holm"] for k in sorted(out, key=lambda k: out[k]["p_raw"])]
    assert adjusted == sorted(adjusted)
    assert out["a"]["reject"] and not out["b"]["reject"]


def test_holm_never_reduces_a_p_value():
    raw = {"a": 0.01, "b": 0.2, "c": 0.9}
    out = holm(raw)
    assert all(out[k]["p_holm"] >= raw[k] for k in raw)


def test_holm_correction_updates_both_tests_and_the_reading():
    reports = {
        "a": {"p_permutation": 0.01, "p_tost": 0.01},
        "b": {"p_permutation": 0.04, "p_tost": 0.04},
    }
    corrected = apply_holm_to_reports(reports)
    assert corrected["a"]["p_difference_holm"] == pytest.approx(0.02)
    assert corrected["a"]["p_tost_holm"] == pytest.approx(0.02)
    assert corrected["a"]["reading"] == "real but negligible shift"
    assert corrected["b"]["p_difference_holm"] == pytest.approx(0.04)
    assert corrected["b"]["p_tost_holm"] == pytest.approx(0.04)


@pytest.mark.parametrize("p_value,equivalent,difference,inconclusive", [
    (0.01, False, True, False),
    (0.01, True, True, False),
    (0.5, True, False, False),
    (0.5, False, False, True),
])
def test_control_outcomes_separate_detection_from_equivalence(
    p_value, equivalent, difference, inconclusive
):
    result = fidelity_stats.control_outcomes({
        "p_permutation": p_value, "equivalent": equivalent,
    })
    assert result["detected"] == difference
    assert result["difference_detected"] == difference
    assert result["equivalence_established"] == equivalent
    assert result["inconclusive"] == inconclusive


def test_control_outcomes_respect_existing_multiplicity_adjustment():
    result = fidelity_stats.control_outcomes({
        "p_permutation": 0.01, "p_difference_holm": 0.06, "equivalent": False,
    })
    assert not result["detected"]
    assert result["inconclusive"]


def test_inconclusive_control_does_not_demonstrate_detection_power():
    values = [-100.0, 0.0, 100.0]
    assert not detected(values, values, delta=0.01)
