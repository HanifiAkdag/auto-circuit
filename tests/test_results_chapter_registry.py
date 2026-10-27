"""Results exports consume exact published inputs and contain their output paths."""

import copy
import csv
import json
import shutil
import statistics
from pathlib import Path

import pytest

from experiments.config import ROOT, load_registry, resolve_path, sha256
from experiments.results_chapter_sources import bundle, frozen_manifest_index


def test_default_sources_are_published_registry_bundles():
    registry = load_registry()
    for name in ('main', 'docstring'):
        selected = bundle(name, registry)
        assert Path(selected['source_dir']) == resolve_path(registry['benchmark']['bundles'][name]['source_dir'], registry)
        assert Path(selected['selected_inputs']).is_file()
        assert 'artifacts/final' in selected['source_dir']


def test_frozen_selection_rejects_duplicate_joint_identity():
    registry = load_registry()
    item = next(item for item in registry['benchmark']['inputs']
                if item['cell']['task'] == 'ioi' and item['cell']['algo'] == 'eap'
                and item['cell']['ablation'] == 'mean' and item['cell']['seed'] == 42
                and item['cell'].get('init_seed', 42) == 42)
    with pytest.raises(ValueError, match='Duplicate frozen'):
        frozen_manifest_index([item, copy.deepcopy(item)], {'ioi'}, {'eap'}, {'mean'}, {42}, registry=registry)


def test_native_roc_uses_exact_frozen_inputs_without_cache_search(monkeypatch):
    from experiments import results_chapter_native_roc as roc
    registry = load_registry()
    def forbid_search(*args, **kwargs):
        raise AssertionError('A frozen export must not search for newer caches')
    monkeypatch.setattr(Path, 'glob', forbid_search)
    index = roc.cache_index(registry)
    assert len(index) == 183
    expected = {resolve_path(item['scores'], registry) for item in registry['benchmark']['inputs']}
    assert {value[1] for value in index.values()} <= expected


def test_native_summary_combines_regenerated_and_published_tasks(tmp_path):
    from experiments import results_chapter_native_roc as roc
    registry = load_registry()
    output = tmp_path / 'exports'
    fresh = tmp_path / 'evaluation/main/summary.csv'
    fresh.parent.mkdir(parents=True)
    fresh.write_text('fresh main summary')
    paths = roc.summary_sources(registry, output)
    assert paths[0] == fresh
    assert paths[1] == resolve_path(registry['benchmark']['bundles']['docstring']['source_dir'], registry) / 'summary.csv'


def test_native_auc_and_acdc_boundaries_match_frozen_sources(tmp_path, monkeypatch):
    from experiments import results_chapter_native_roc as roc
    rendered = []
    monkeypatch.setattr(roc, 'render_native_figures', lambda rows, means, output_root: rendered.append(output_root))
    registry = load_registry()
    roc.main(registry=registry, output_root=tmp_path)
    assert rendered == [tmp_path]
    with (tmp_path / 'native_roc/manual_circuit_auc.csv').open(newline='') as handle:
        actual = list(csv.DictReader(handle))
    assert len(actual) == 36
    reference = resolve_path(registry['benchmark']['bundles']['main']['exhibits_dir'], registry) / 'native_roc/manual_circuit_auc.csv'
    with reference.open(newline='') as handle:
        expected = list(csv.DictReader(handle))
    keys = ('task', 'method', 'discovery_ablation')
    expected = {tuple(row[key] for key in keys): row for row in expected}
    with reference.with_name('manual_circuit_auc_per_seed.csv').open(newline='') as handle:
        per_seed = list(csv.DictReader(handle))
    for row in actual:
        identity = tuple(row[key] for key in keys)
        source = expected[identity]
        group = [r for r in per_seed if tuple(r[key] for key in keys) == identity]
        assert sorted(int(r['seed']) for r in group) == [42, 43, 44, 45, 46]
        assert float(row['auc_mean']) == pytest.approx(float(source['auc_mean']), abs=1e-12)
        assert row['sd_ddof'] == '0'
        assert float(row['auc_sd']) == pytest.approx(
            statistics.pstdev(float(r['auc']) for r in group), abs=1e-12)
    meta = json.loads((tmp_path / 'native_roc/manual_native_roc_manifest.json').read_text())
    assert meta['sd_ddof'] == 0 and meta['sd_convention'] == 'population'
    assert len(meta['acdc_selected_boundary_check']) == 45
    assert len(meta['summary_sha256']) == 2


def test_no_argument_manual_export_copies_published_measurements(tmp_path, monkeypatch):
    from experiments import results_chapter_manual_baseline as manual
    monkeypatch.setattr(manual, 'EXPORT_ROOT', tmp_path)
    monkeypatch.setattr(manual, 'measure_prune_metrics', lambda **kwargs: pytest.fail('Published baseline export must not run a model'))
    manual.main()
    registry = load_registry()
    for name in ('main', 'docstring'):
        source = resolve_path(registry['benchmark']['bundles'][name]['exhibits_dir'], registry) / 'size_curves/manual_baselines.csv'
        assert (tmp_path / name / 'size_curves/manual_baselines.csv').read_bytes() == source.read_bytes()


def test_paired_analysis_checks_regenerated_manifest_instead_of_golden_hashes(tmp_path):
    from experiments import results_chapter_structure_function as paired
    registry = load_registry()
    main = registry['benchmark']['bundles']['main']
    published = resolve_path(main['source_dir'], registry)
    source = tmp_path / 'evaluation/main'
    files = ('functional.csv', 'structural.csv', 'summary.csv',
             'exhibits/stability/within_ablation_selected.csv', 'exhibits/stability/exhibit_manifest.json',
             'exhibits/controlled_comparison/matched_quality.csv',
             'exhibits/controlled_comparison/matched_quality_manifest.json')
    for name in files:
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(published / name, destination)
    selected = resolve_path(main['selected_inputs'], registry)
    derived = {'complete': True, 'seeds': [42, 43, 44, 45, 46],
               'evaluation_protocol_id': main['evaluation_protocol_id'],
               'selected_inputs': str(selected), 'selected_inputs_sha256': sha256(selected)}
    (source / 'manifest.json').write_text(json.dumps(derived))
    exhibit_path = source / 'exhibits/stability/exhibit_manifest.json'
    exhibit = json.loads(exhibit_path.read_text())
    exhibit['source_sha256']['manifest.json'] = sha256(source / 'manifest.json')
    exhibit_path.write_text(json.dumps(exhibit))
    for name in ('functional', 'structural', 'manifest'):
        main[name + '_sha256'] = 'superseded golden hash must not apply to regenerated inputs'
    output = tmp_path / 'paired'
    paired.run(ROOT, output, registry=registry, source_dir=source)
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['n_circuits'] == 40
    assert manifest['n_metric_pairs'] == 120
    assert manifest['source_hashes'][str(source / 'manifest.json')] == sha256(source / 'manifest.json')
