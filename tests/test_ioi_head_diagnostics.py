import numpy as np
import pytest

from experiments.thesis.ioi_head_diagnostics.protocol import (
    native_mask, exact_mask, positions, make_interventions, normalized_margin,
    control_sets, replacement_sets,
)


def edge(src, dest):
    layer = int(dest.split('.')[0][1:]) if dest.startswith('A') else 12
    kind = 'attention' if src.startswith('A') else 'mlp' if src.startswith('MLP') else 'input'
    return dict(name=f'{src}->{dest}', src=src, dest=dest, layer=layer,
                channel=dest.rsplit('.', 1)[-1] if dest.startswith('A') else 'output', kind=kind)


def test_signed_scores_and_threshold_ties():
    scores = np.array([-5., 4., -4., 1.])
    assert native_mask(scores, 2).tolist() == [True, True, True, False]
    masks = [exact_mask(scores, 2, i) for i in range(10)]
    assert all(m.sum() == 2 and m[0] for m in masks)
    assert len({tuple(m) for m in masks}) == 2
    for i in range(10):
        assert np.all(exact_mask(scores, 2, i) <= exact_mask(scores, 3, i))


def test_acdc_infinite_scores_are_valid_rankings_but_nan_is_not():
    scores = np.array([np.inf, np.inf, 2., 1.])
    assert native_mask(scores, 1).tolist() == [True, True, False, False]
    assert exact_mask(scores, 1, 'acdc').sum() == 1
    with pytest.raises(ValueError, match='NaN'):
        native_mask(np.array([np.nan, 1.]), 1)


def test_positions_with_left_padding_and_subject_order():
    assert positions(np.array([0, 0, 9, 31, 21, 31, 8]), 21, 31) == (4, 3, 5)
    assert positions(np.array([0, 9, 21, 31, 31, 8]), 21, 31) == (2, 3, 4)
    with pytest.raises(ValueError, match='occurrences'):
        positions(np.array([21, 31, 8]), 21, 31)


def test_ratio_of_averages():
    assert normalized_margin(np.array([1., 9.]), np.array([1., 3.])) == 250.


def test_intervention_masks_restore_only_target_channels_and_keep_output_paths_distinct():
    edges = [edge('A9.9', 'A10.7.Q'), edge('A9.9', 'A10.7.K'),
             edge('A7.3', 'A11.10.Q'), edge('A10.7', 'A11.10.Q'),
             edge('A10.7', 'Resid End'), edge('A11.10', 'Resid End'),
             edge('A9.9', 'Resid End'), edge('A10.2', 'Resid End')]
    base = np.array([False, False, True, True, True, True, True, True])
    variants = make_interventions(edges, base)
    assert variants['restore_q_both'].tolist() == [True, False, True, True, True, True, True, True]
    assert variants['drop_nnm_out'].tolist() == [False, False, True, True, False, False, True, True]
    assert variants['drop_nm_out'][4] and not variants['drop_nm_out'][6]
    assert variants['interhead_drop'][4] and not variants['interhead_drop'][3]
    assert np.array_equal(base, [False, False, True, True, True, True, True, True])


def test_control_matching_and_fixed_budget_pairs_are_reproducible():
    edges = [edge('A9.9', 'A10.7.Q'), edge('MLP 0', 'A11.10.Q')]
    edges += [edge('A9.9', f'A10.{h}.Q') for h in range(6)]
    edges += [edge('MLP 0', f'A11.{h}.Q') for h in range(6)]
    edges += [edge('Resid Start', f'A9.{h}.K') for h in range(6)]
    base = np.zeros(len(edges), dtype=bool)
    base[-6:] = True
    add = np.zeros(len(edges), dtype=bool)
    add[:2] = True
    controls = control_sets(edges, base, add, 'case')
    assert len(controls) == 20
    assert all(len(c) == 2 and not (set(c) & {0, 1}) for c in controls)
    assert controls == control_sets(edges, base, add, 'case')
    pairs = replacement_sets(edges, np.arange(len(edges)), base, add, 'case')
    assert len(pairs) == 10
    for removal, restored in pairs:
        assert removal.sum() == base.sum() - 2
        assert restored.sum() == base.sum()
        assert np.array_equal(restored & ~add, removal)


def test_metrics_preserve_kl_direction_and_strict_accuracy():
    import torch
    from experiments.thesis.ioi_head_diagnostics.evaluation import output_metrics
    full = torch.tensor([[2., 0., -1.], [0., 0., -1.]])
    changed = torch.tensor([[0., 2., -1.], [0., 0., -1.]])
    values = output_metrics(changed, full, torch.tensor([0, 0]), torch.tensor([1, 1]))
    assert values['margin'].tolist() == [-2., 0.]
    assert values['accuracy'].tolist() == [0., 0.]
    expected = (full[0].softmax(-1) * (full[0].log_softmax(-1) - changed[0].log_softmax(-1))).sum()
    assert torch.allclose(values['kl'][0], expected)
    assert values['kl'][1] == 0


def test_crossed_bootstrap_keeps_paired_constant_effect():
    from experiments.thesis.ioi_head_diagnostics.analysis import crossed_interval
    data = np.full((5, 16), 2.5)
    assert crossed_interval(data, draws=200) == (2.5, 2.5)
    denominator = np.arange(1, 17, dtype=float)
    paired = np.broadcast_to(denominator * .5, (5, 16))
    assert crossed_interval(paired, draws=200, denominator=denominator) == (50., 50.)


def test_unavailable_controls_do_not_relax_strata():
    edges = [edge('A9.9', 'A10.7.Q'), edge('MLP 0', 'A10.0.Q')]
    with pytest.raises(ValueError, match='Unavailable'):
        control_sets(edges, np.array([False, False]), np.array([True, False]), 'missing')


def test_frozen_inventory_identity_ignores_file_container_but_rejects_content_changes():
    from experiments.thesis.ioi_head_diagnostics.protocol import fingerprint, validate_identity
    content = dict(schema=1, conditions=[dict(id=0, attained=3)])
    saved = dict(content, study_id=fingerprint(content), masks_file_hash='container-sha')
    validate_identity(saved)
    saved['conditions'][0]['attained'] = 4
    with pytest.raises(ValueError, match='identity'):
        validate_identity(saved)


def test_reference_checksum_rejects_changed_denominator(tmp_path):
    import json
    from experiments.thesis.ioi_head_diagnostics.protocol import file_hash, validate_reference
    reference = tmp_path / 'full_reference.npz'
    prompts = tmp_path / 'prompt_positions.json'
    np.savez(reference, margin=np.array([2., 3.]))
    prompts.write_text('{}')
    (tmp_path / 'reference.json').write_text(json.dumps(dict(study_id='study', complete=True,
        checksums={p.name: file_hash(p) for p in (reference, prompts)})))
    validate_reference(tmp_path, dict(study_id='study'))
    np.savez(reference, margin=np.array([20., 30.]))
    with pytest.raises(ValueError, match='checksum'):
        validate_reference(tmp_path, dict(study_id='study'))


def test_failed_observed_forward_cleans_hooks_masks_and_patch_state():
    import torch
    from types import SimpleNamespace
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    from auto_circuit.utils.graph_utils import patchable_model
    from auto_circuit.utils.ablation_activations import src_ablations
    from auto_circuit.utils.misc import module_by_name
    from experiments.thesis.ioi_head_diagnostics.protocol import graph_records
    from experiments.thesis.ioi_head_diagnostics.evaluation import forward
    cfg = HookedTransformerConfig(n_layers=12, n_heads=12, d_model=24, d_head=2,
        d_mlp=48, d_vocab=17, n_ctx=8, act_fn='relu', device='cpu',
        use_attn_result=True, use_attn_in=True, use_split_qkv_input=True, use_hook_mlp_in=True)
    model = patchable_model(HookedTransformer(cfg), factorized=True, separate_qkv=True, slice_output='last_seq')
    batch = SimpleNamespace(clean=torch.tensor([[1, 2, 3]]))
    replacement = src_ablations(model, torch.tensor([[1, 4, 3]]))
    before_masks = {k: v.detach().clone() for k, v in model.patch_masks.items()}
    def fail(module, inputs, output):
        raise RuntimeError('injected observation failure')
    handle = module_by_name(model, 'blocks.10.attn.hook_pattern').register_forward_hook(fail)
    hook_counts = {name: len(module._forward_hooks) for name, module in model.named_modules()}
    try:
        with pytest.raises(RuntimeError, match='injected'):
            forward(model, batch, graph_records(model), np.zeros(model.n_edges, dtype=bool), replacement)
        assert all(torch.equal(v, before_masks[k]) for k, v in model.patch_masks.items())
        assert all(not w.patch_mode and w.curr_src_outs is None for w in model.wrappers)
        assert all(w.patch_src_outs is None for w in model.dest_wrappers)
        assert hook_counts == {name: len(module._forward_hooks) for name, module in model.named_modules()}
    finally:
        handle.remove()


def test_mean_case_plans_controls_and_exact_interventions_at_each_budget():
    # Catches silently restricting causal conditions to 3000 or sharing control identities across budgets.
    from experiments.thesis.ioi_head_diagnostics import protocol
    edges = [edge('A9.9', 'A10.7.Q'), edge('MLP 0', 'A11.10.Q')]
    edges += [edge('A9.9', f'A10.{h}.Q') for h in range(6)]
    edges += [edge('MLP 0', f'A11.{h}.Q') for h in range(6)]
    edges += [edge('Resid Start', f'A9.{h}.K') for h in range(12)]
    scores = np.zeros(len(edges))
    scores[-12:] = [1, 2, 3, 4, 5, 6, 7, 9, 9, 9, 10, 11]
    conditions, masks, status = protocol.plan_case(edges, scores, 2, ['eap', 'mean', 42], budgets=(4, 6, 8))
    assert set(status) == {'4', '6', '8'}
    assert all(s == 'available' for s in status.values())
    for budget, attained in [(4, 5), (6, 6), (8, 8)]:
        selected = [c for c in conditions if c['budget'] == budget]
        baseline = next(c for c in selected if c['label'] == 'baseline')
        assert baseline['attained'] == attained
        assert sum(c['label'].startswith('control_') for c in selected) == 20
        assert sum(c['label'].startswith('fixed_q_') for c in selected) == 10
        q = next(c for c in selected if c['label'] == 'restore_q_both')
        assert q['added'] == [0, 1]
        assert q['removed'] == []
        for c in selected:
            assert c['case'] == 2
            assert int(masks[c['id']].sum()) == c['attained']
        assert sum(c['label'] == 'exact_baseline' for c in selected) == (10 if budget == 4 else 0)
        assert sum(c['label'] == 'exact_restore_q_both' for c in selected) == (10 if budget == 4 else 0)


def test_primary_effects_do_not_overwrite_discovery_runs_across_budgets():
    # Catches a budget omitted from inference keys, which used to pool/overwrite different circuits.
    from experiments.thesis.ioi_head_diagnostics import analysis
    conditions, arrays, results = [], {}, {}
    for case, seed in enumerate((42, 43)):
        for budget, baseline, effect in ((739, 2., 1.), (2000, 4., 3.)):
            for label, phase, value in [('baseline', 'screen', baseline), ('restore_q_both', 'causal', baseline + effect)]:
                idx = len(conditions)
                conditions.append(dict(id=idx, case=case, budget=budget, label=label, phase=phase))
                arrays[idx] = {key: np.full(3, value) for key in ('margin', 'correct', 'wrong', 'kl', 'accuracy')}
                arrays[idx]['attention'] = np.zeros((3, 2, 3))
                results[idx] = dict(cell=dict(algo='eap', seed=seed, ablation='mean'))
    primary = analysis.collect_primary(conditions, arrays, results)
    assert set(primary[('eap', 739, 'restore_q_both', 'margin')]) == {42, 43}
    assert set(primary[('eap', 2000, 'restore_q_both', 'margin')]) == {42, 43}
    assert primary[('eap', 739, 'restore_q_both', 'margin')][42].tolist() == [1., 1., 1.]
    assert primary[('eap', 2000, 'restore_q_both', 'margin')][42].tolist() == [3., 3., 3.]


def test_kl_partition_detects_wrong_pair_mass_even_with_unchanged_answer_margin():
    # Catches conflating within-name preference with full-vocabulary faithfulness.
    import torch
    from experiments.thesis.ioi_head_diagnostics.evaluation import output_metrics
    full = torch.log(torch.tensor([[.3, .1, .2, .4]], dtype=torch.float64))
    changed = torch.log(torch.tensor([[.6, .2, .05, .15]], dtype=torch.float64))
    values = output_metrics(changed, full, torch.tensor([0]), torch.tensor([1]))
    assert torch.allclose(values['kl_pair'], torch.zeros(1, dtype=torch.float64), atol=1e-12)
    assert values['kl_partition'][0] > .3
    assert values['kl_other'][0] > 0
    assert torch.allclose(values['kl'], values['kl_partition'] + values['kl_pair'] + values['kl_other'], atol=1e-12)
    assert values['pair_probability'].tolist() == pytest.approx([.8])
    assert values['correct_probability'].tolist() == pytest.approx([.6])


def test_mean_output_reference_selects_actual_source_indices():
    # Catches attributing output replacement to individual corrupt activations or the wrong source order.
    import torch
    from types import SimpleNamespace
    from experiments.thesis.ioi_head_diagnostics import evaluation
    model = SimpleNamespace(srcs=[SimpleNamespace(name='A11.10', src_idx=0),
                                 SimpleNamespace(name='A10.7', src_idx=2)])
    replacement = torch.arange(3 * 2 * 4 * 5).reshape(3, 2, 4, 5)
    reference = evaluation.output_reference(model, replacement)
    assert torch.equal(reference['result_10_7'], replacement[2, :, -1])
    assert torch.equal(reference['result_11_10'], replacement[0, :, -1])


def test_exact_tie_effects_are_averaged_within_discovery_runs():
    # Catches dropping exact-size probes or treating tie resolutions as independent runs.
    from experiments.thesis.ioi_head_diagnostics.analysis import collect_primary
    conditions, arrays, results = [], {}, {}
    for seed, case in [(42, 0), (43, 1)]:
        for realization, effect in [(0, 1.), (1, 3.)]:
            for label, value in [('exact_baseline', 4.), ('exact_restore_q_both', 4. + effect)]:
                idx = len(conditions)
                conditions.append(dict(id=idx, case=case, budget=2000, label=label,
                                       phase='size-sensitivity', realization=realization))
                arrays[idx] = {key: np.full(3, value) for key in ('margin', 'correct', 'wrong', 'kl', 'accuracy')}
                arrays[idx]['attention'] = np.zeros((3, 2, 3))
                results[idx] = dict(cell=dict(algo='acdc', seed=seed, ablation='mean'))
    for label, phase, value in [('baseline', 'screen', 4.), ('restore_q_both', 'causal', 8.)]:
        idx = len(conditions)
        conditions.append(dict(id=idx, case=2, budget=2000, attained=2000, label=label, phase=phase))
        arrays[idx] = {key: np.full(3, value) for key in ('margin', 'correct', 'wrong', 'kl', 'accuracy')}
        arrays[idx]['attention'] = np.zeros((3, 2, 3))
        results[idx] = dict(cell=dict(algo='acdc', seed=44, ablation='mean'))
    primary = collect_primary(conditions, arrays, results)
    values = primary[('acdc', 2000, 'exact_restore_q_both', 'margin')]
    assert set(values) == {42, 43, 44}
    assert values[42].tolist() == [2., 2., 2.]
    assert values[43].tolist() == [2., 2., 2.]
    assert values[44].tolist() == [4., 4., 4.]


def test_exact_endpoint_table_includes_untied_native_runs_once():
    from experiments.thesis.ioi_head_diagnostics.analysis import exact_run_rows
    from experiments.thesis.ioi_head_diagnostics.evaluation import METRICS
    rows = []
    for seed, phase, label, value in [(42, 'size-sensitivity', 'exact_baseline', 2.),
                                      (42, 'size-sensitivity', 'exact_baseline', 4.),
                                      (43, 'screen', 'baseline', 7.),
                                      (44, 'screen', 'baseline', 9.)]:
        rows.append(dict(algo='ep', seed=seed, budget=3000, attained=3001 if seed == 44 else 3000,
                         phase=phase, label=label, logit_difference_percent=value,
                         **{key: value for key in METRICS}))
    exact = exact_run_rows(rows)
    assert {r['seed'] for r in exact} == {42, 43}
    tied = next(r for r in exact if r['seed'] == 42)
    native = next(r for r in exact if r['seed'] == 43)
    assert tied['n_realizations'] == 2 and tied['margin'] == 3.
    assert native['n_realizations'] == 1 and native['margin'] == 7.


def test_exact_endpoint_export_reports_actual_intervention_sizes():
    from experiments.thesis.ioi_head_diagnostics.analysis import exact_run_rows
    from experiments.thesis.ioi_head_diagnostics.evaluation import METRICS
    rows = [dict(algo='acdc', seed=42, budget=2000, attained=n, phase='size-sensitivity',
                 label='exact_restore_q_both', logit_difference_percent=100.,
                 **{key: 1. for key in METRICS}) for n in (2200, 2210)]
    row = exact_run_rows(rows)[0]
    assert row['attained'] == 2205
    assert row['attained_min'] == 2200 and row['attained_max'] == 2210
    assert row['baseline_attained'] == 2000
