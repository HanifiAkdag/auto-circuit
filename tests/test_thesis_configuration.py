"""Portable input resolution and complete final discovery identities."""
import hashlib
import json
from pathlib import Path

import pytest


def test_discovery_matrix_contains_each_final_identity_once():
    from experiments.config import discovery_cells
    cells = discovery_cells()
    assert len(cells) == 342
    identities = {(c['task'], c['algo'], c['ablation'], c['seed'], c['init_seed']) for c in cells}
    assert len(identities) == 342
    assert sum(c['split_acdc'] for c in cells) == 30
    assert sum(c['algo'] == 'gt' for c in cells) == 3


def test_published_paths_resolve_original_absolute_provenance(tmp_path):
    from experiments.config import resolve_path
    target = tmp_path / 'artifacts/final/example.json'
    target.parent.mkdir(parents=True)
    target.write_text('{}')
    registry = {'path_map': {'results/old/example.json': 'artifacts/final/example.json'}}
    actual = resolve_path('/mnt/beegfs/work/akdag/thesis/auto-circuit/results/old/example.json', registry, root=tmp_path)
    assert actual == target


def test_absolute_path_inside_fresh_checkout_is_not_rebased(tmp_path):
    from experiments.config import resolve_path
    target = tmp_path / 'artifacts/final/input.json'
    assert resolve_path(str(target), {}, root=tmp_path) == target


def test_conflicting_input_identity_rejected():
    from experiments.config import validate_inventory
    item = {'cell': {'task': 'ioi', 'algo': 'ep', 'ablation': 'zero', 'seed': 42},
            'scores': 'one.pkl', 'manifest': 'one.json'}
    with pytest.raises(ValueError, match='Duplicate'):
        validate_inventory([item, {**item, 'scores': 'two.pkl'}])


def test_cache_directory_selection_rejects_duplicate_identities():
    from experiments.config import require_unique_caches
    manifest = {'cell': {'task': 'ioi', 'algo': 'ep', 'ablation': 'zero', 'seed': 42}}
    with pytest.raises(ValueError, match='Duplicate discovery cache identity'):
        require_unique_caches([(manifest, {}), (manifest, {})])


def test_resume_rejects_changed_scientific_configuration(tmp_path):
    from experiments.config import freeze_run
    freeze_run(tmp_path, {'protocol': 'thesis', 'smoke': False})
    with pytest.raises(ValueError, match='configuration'):
        freeze_run(tmp_path, {'protocol': 'thesis', 'smoke': True})
