import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from run_cell import _cache_stem  # noqa: E402


def test_joint_stem_matches_the_legacy_name():
    """The 195 existing cells must keep resolving, so a joint cell carries no init tag."""
    assert _cache_stem("ioi", "ep", "resample", 42, None) == "cell_ioi_ep_resample_seed42"
    assert _cache_stem("ioi", "ep", "resample", 42, 42) == "cell_ioi_ep_resample_seed42"


def test_decoupled_init_gets_a_tag():
    assert _cache_stem("ioi", "ep", "resample", 43, 42) == "cell_ioi_ep_resample_seed43_init42"
    assert _cache_stem("ioi", "ep", "resample", 42, 43) == "cell_ioi_ep_resample_seed42_init43"


def test_stem_slugs_algo_and_keeps_other_tags():
    assert _cache_stem("gt", "discogp-ep", "zero", 42, 43, null_test=True, partial_tag="a") == (
        "cell_gt_discogp_ep_zero_seed42_init43_null__a"
    )


def test_joint_glob_does_not_match_a_decoupled_cell():
    """--skip-if-exists globs '<stem>-*.pkl'; the separator after the seed keeps the two
    apart, which is what stops a decomposition cell being mistaken for a joint one."""
    joint = _cache_stem("ioi", "ep", "resample", 43, 43)
    decoupled = _cache_stem("ioi", "ep", "resample", 43, 42)
    assert not f"{decoupled}-2026.pkl".startswith(f"{joint}-")
