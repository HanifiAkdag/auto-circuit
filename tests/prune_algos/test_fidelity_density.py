"""Density scope must not change the circuits produced by mask extraction."""

import pytest
import torch

from experiments.port_comparison.fidelity_analysis import edge_density


def test_common_density_excludes_first_source_for_every_destination():
    scores = {
        "heads": torch.tensor([[0.9, 0.9, 0.1], [0.1, 0.9, 0.1]]),
        "residual": torch.tensor([0.9, 0.1]),
    }
    assert edge_density(scores, "threshold") == pytest.approx(4 / 8)
    assert edge_density(scores, "threshold", exclude_embedding_edges=True) == pytest.approx(2 / 5)


def test_common_density_extracts_full_tensor_before_excluding_embeddings():
    # Expected zeros round to one for both tensors. The embedding is smallest,
    # so extraction on the full tensor keeps both common edges, not just one.
    scores = {"residual": torch.tensor([0.1, 0.2, 0.2])}
    assert edge_density(scores, "expected_l0") == pytest.approx(2 / 3)
    assert edge_density(scores, "expected_l0", exclude_embedding_edges=True) == 1.0
    assert edge_density({"residual": scores["residual"][1:]}, "expected_l0") == 0.5
