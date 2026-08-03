"""Iteration order must not depend on set ordering.

`model.edges` is a `set` and `model.dest_wrappers` is a set of `nn.Module`s
(identity-hashed), so their iteration order varies between processes. Any code that
draws random masks, or builds an autograd graph, in that order produces a different
result on every run from the same seed. These tests pin the orders that matter.
"""
import transformer_lens as tl

from auto_circuit.utils.graph_utils import patchable_model


def _acdc_order(model):
    """Mirror of the sort in `ACDC.acdc_prune_scores`."""
    return sorted(
        model.edges,
        key=lambda e: (-e.dest.layer, e.name, -1 if e.seq_idx is None else e.seq_idx),
    )


def test_ordered_dest_wrappers_is_canonical(mini_tl_transformer: tl.HookedTransformer):
    """EP and DiscoGP build their mask dicts from this property."""
    model = patchable_model(mini_tl_transformer, factorized=True, separate_qkv=True)
    names = [w.module_name for w in model.ordered_dest_wrappers]

    assert names == sorted(names)
    assert len(names) == len(set(names)), "module_name is not a total order"
    assert set(names) == {w.module_name for w in model.dest_wrappers}


def test_patch_masks_keys_are_sorted(mini_tl_transformer: tl.HookedTransformer):
    """EP, DiscoGP and the Random baseline draw their masks in `patch_masks` order."""
    model = patchable_model(mini_tl_transformer, factorized=True, separate_qkv=True)
    assert list(model.patch_masks) == sorted(model.patch_masks)


def test_prune_scores_keys_are_sorted(mini_tl_transformer: tl.HookedTransformer):
    """Every algo must emit PruneScores in one canonical key order.

    Values are keyed by module name, so a per-process order is harmless to the scores
    themselves, but it silently breaks any analysis that flattens the dict into a
    single index space (e.g. concatenating tensors to compare two runs).
    """
    model = patchable_model(mini_tl_transformer, factorized=True, separate_qkv=True)
    expected = sorted(model.patch_masks)

    assert list(model.new_prune_scores()) == expected
    assert [w.module_name for w in model.ordered_dest_wrappers] == expected


def test_acdc_edge_order_is_a_total_order(mini_tl_transformer: tl.HookedTransformer):
    """Layer alone leaves within-layer ties in set order; the key must break them."""
    model = patchable_model(mini_tl_transformer, factorized=True, separate_qkv=True)
    order = _acdc_order(model)

    assert len(order) == model.n_edges
    keys = [
        (-e.dest.layer, e.name, -1 if e.seq_idx is None else e.seq_idx) for e in order
    ]
    assert len(set(keys)) == len(keys), "sort key does not uniquely order the edges"
    assert keys == sorted(keys)
    # Layers must still descend, as ACDC's greedy sweep requires.
    layers = [e.dest.layer for e in order]
    assert layers == sorted(layers, reverse=True)
