"""Observed component-level forwards, numerical checks, and resumable workers."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import fcntl
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch
import torch.nn.functional as F

from auto_circuit.utils.ablation_activations import src_ablations
from auto_circuit.utils.graph_utils import patch_mode
from auto_circuit.utils.misc import module_by_name
from auto_circuit.types import AblationType, PatchType

from .protocol import ROOT, atomic_json, file_hash, fingerprint, load_inventory, unwrap_scores, normalized_margin, validate_reference

HEADS = ((10, 7), (11, 10))
ABLATION = AblationType.TOKENWISE_MEAN_CORRUPT
METRICS = ('correct', 'wrong', 'margin', 'kl', 'accuracy', 'kl_partition', 'kl_pair', 'kl_other',
           'correct_probability', 'wrong_probability', 'pair_probability', 'entropy')


def output_metrics(logits, full, correct, wrong):
    rows = torch.arange(len(logits), device=logits.device)
    good, bad = logits[rows, correct], logits[rows, wrong]
    full_lp = F.log_softmax(full, dim=-1)
    lp = F.log_softmax(logits, dim=-1)
    full_p = full_lp.exp()
    kl = (full_p * (full_lp - lp)).sum(-1)
    pair_idx = torch.stack([correct, wrong], dim=-1)
    f_pair = full_lp.gather(-1, pair_idx)
    c_pair = lp.gather(-1, pair_idx)
    f_mass = f_pair.logsumexp(-1)
    c_mass = c_pair.logsumexp(-1)
    # Compute rest mass from its own logsumexp, not 1 - pair_mass (unstable near one).
    full_rest, rest = full_lp.clone(), lp.clone()
    full_rest.scatter_(-1, pair_idx, -torch.inf)
    rest.scatter_(-1, pair_idx, -torch.inf)
    f_rest_mass, c_rest_mass = full_rest.logsumexp(-1), rest.logsumexp(-1)
    partition = f_mass.exp() * (f_mass - c_mass) + f_rest_mass.exp() * (f_rest_mass - c_rest_mass)
    pair_kl = (f_pair.exp() * ((f_pair - f_mass[:, None]) - (c_pair - c_mass[:, None]))).sum(-1)
    rest_diff = (full_lp - f_rest_mass[:, None]) - (lp - c_rest_mass[:, None])
    rest_diff.scatter_(-1, pair_idx, 0.)
    other_kl = (full_p * rest_diff).sum(-1)
    if not torch.allclose(kl, partition + pair_kl + other_kl, atol=1e-4, rtol=1e-4):
        raise ValueError('KL partition reconstruction failed')
    return dict(correct=good, wrong=bad, margin=good - bad,
                kl=kl.clamp_min(0), accuracy=(good > bad).float(),
                kl_partition=partition.clamp_min(0), kl_pair=pair_kl.clamp_min(0), kl_other=other_kl.clamp_min(0),
                correct_probability=lp[rows, correct].exp(), wrong_probability=lp[rows, wrong].exp(),
                pair_probability=c_mass.exp(), entropy=-(lp.exp() * lp).sum(-1))


def output_reference(model, replacement):
    indices = {src.name: src.src_idx for src in model.srcs}
    return {f'result_{layer}_{head}': replacement[indices[f'A{layer}.{head}'], :, -1]
            for layer, head in HEADS}


def answer_ids(batch):
    def single(value):
        if isinstance(value, list):
            value = torch.stack(value)
        if value.ndim != 2 or value.shape[1] != 1:
            raise ValueError('Diagnostic requires one-token IOI answers')
        return value[:, 0]
    return single(batch.answers), single(batch.wrong_answers)


@contextmanager
def observe(model, hybrid=None, full_capture=None):
    captured, handles = {}, []
    def collect(key, transform=lambda x: x):
        def hook(module, inputs, output):
            captured[key] = transform(output).detach().clone()
        return hook
    try:
        for layer, head in HEADS:
            tag = f'{layer}_{head}'
            q_module = module_by_name(model, f'blocks.{layer}.hook_q_input')
            def q_hook(module, inputs, output, h=head, t=tag):
                if hybrid in (t, 'both'):
                    output = output.clone()
                    output[:, :, h] = full_capture['q_input_' + t]
                captured['q_input_' + t] = output[:, :, h].detach().clone()
                return output
            handles.append(q_module.register_forward_hook(q_hook))
            for channel in ('q', 'k', 'v'):
                module = module_by_name(model, f'blocks.{layer}.attn.hook_{channel}')
                handles.append(module.register_forward_hook(collect(channel + '_' + tag, lambda x, h=head: x[:, :, h])))
            handles.append(module_by_name(model, f'blocks.{layer}.attn.hook_pattern').register_forward_hook(
                collect('pattern_' + tag, lambda x, h=head: x[:, h, -1])))
            handles.append(module_by_name(model, f'blocks.{layer}.attn.hook_result').register_forward_hook(
                collect('result_' + tag, lambda x, h=head: x[:, -1, h])))
        handles.append(module_by_name(model, 'ln_final.hook_scale').register_forward_hook(
            collect('scale', lambda x: x[:, -1])))
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def apply_mask(model, records, mask):
    with torch.no_grad():
        for value in model.patch_masks.values():
            value.fill_(1)
        # Construct on CPU and transfer once per wrapper rather than per edge.
        buffers = {name: np.ones(tuple(value.shape), dtype=np.float32) for name, value in model.patch_masks.items()}
        for e, retained in zip(records, mask):
            buffers[e['module']][tuple(e['index'])] = 0. if retained else 1.
        for name, value in model.patch_masks.items():
            value.copy_(torch.from_numpy(buffers[name]).to(value.device))


def forward(model, batch, records=None, mask=None, replacements=None, hybrid=None, full_capture=None):
    previous = {name: value.detach().clone() for name, value in model.patch_masks.items()}
    try:
        with torch.inference_mode(), observe(model, hybrid, full_capture) as captured:
            if mask is None:
                logits = model(batch.clean)[model.out_slice]
            else:
                apply_mask(model, records, mask)
                with patch_mode(model, replacements):
                    logits = model(batch.clean)[model.out_slice]
        return logits.detach().clone(), captured
    finally:
        with torch.no_grad():
            for name, value in previous.items():
                model.patch_masks[name].copy_(value)


def diagnostics(model, logits, full_logits, captured, replacement_outputs, correct, wrong, positions_tensor):
    values = output_metrics(logits, full_logits, correct, wrong)
    rows = torch.arange(len(logits), device=logits.device)
    attention, scores, outputs, dla, contrasts = [], [], [], [], []
    gain = getattr(model.wrapped_model.ln_final, 'w', torch.ones(model.cfg.d_model, device=logits.device))
    scale = captured['scale']
    if torch.any(scale <= 0):
        raise ValueError('Invalid final LayerNorm scale')
    directions = torch.stack([model.wrapped_model.W_U[:, correct].T,
                              model.wrapped_model.W_U[:, wrong].T], dim=1)
    for layer, head in HEADS:
        tag = f'{layer}_{head}'
        pattern, v = captured['pattern_' + tag], captured['v_' + tag]
        reconstructed = torch.einsum('bp,bpd,dm->bm', pattern, v,
                                     model.wrapped_model.blocks[layer].attn.W_O[head])
        output = captured['result_' + tag]
        if not torch.allclose(reconstructed, output, atol=1e-4, rtol=1e-4):
            raise ValueError(f'Attention/value/output reconstruction failed for {tag}: {(reconstructed-output).abs().max().item()}')
        attention.append(torch.stack([pattern[rows, positions_tensor[:, j]] for j in range(3)], dim=-1))
        q, k = captured['q_' + tag][:, -1], captured['k_' + tag]
        selected_k = torch.stack([k[rows, positions_tensor[:, j]] for j in range(3)], dim=1)
        scores.append(torch.einsum('bd,bnd->bn', q, selected_k) / model.cfg.d_head ** .5)
        def attribution(vector):
            normalized = (vector - vector.mean(-1, keepdim=True)) * gain / scale
            return torch.einsum('bd,bnd->bn', normalized, directions)
        outputs.append(output)
        dla.append(attribution(output))
        contrasts.append(attribution(output - replacement_outputs['result_' + tag]))
    values.update(attention=torch.stack(attention, 1), scores=torch.stack(scores, 1),
                  vectors=torch.stack(outputs, 1), dla=torch.stack(dla, 1),
                  contrast_dla=torch.stack(contrasts, 1), final_scale=scale[:, 0])
    return {key: value.detach().cpu().numpy() for key, value in values.items()}


def build_task(manifest):
    from experiments.thesis.evaluation_protocol import load_protocol, build_evaluation_tasks, verify_evaluation_task
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = load_protocol(Path(manifest['evaluation_manifest']))
    task = next(iter(build_evaluation_tasks(protocol, ['ioi']).values()))
    verify_evaluation_task(task, protocol, 'ioi')
    model = task.model
    import importlib.metadata
    if any(importlib.metadata.version(p) != v for p, v in manifest['runtime']['packages'].items()):
        raise ValueError('Runtime packages differ from frozen scientific configuration')
    if str(next(model.parameters()).dtype) != manifest['runtime']['dtype']:
        raise ValueError('Runtime model dtype differs from frozen configuration')
    configuration = json.loads(json.dumps({k: v for k, v in model.cfg.to_dict().items() if k != 'device'}, default=str))
    if configuration != manifest['runtime']['model_config']:
        raise ValueError('Runtime model configuration differs from frozen scientific configuration')
    if fingerprint(model.tokenizer.get_vocab()) != manifest['runtime']['tokenizer_vocab_hash']:
        raise ValueError('Runtime tokenizer differs from frozen configuration')
    model.eval()
    if model.seq_len is not None or not model.is_factorized or not model.separate_qkv:
        raise ValueError('Incorrect component graph configuration')
    if task.test_loader.diverge_idx != 0:
        raise ValueError('Unexpected cached prefix in the variable-length evaluation set')
    if manifest['evaluation_ablation'] != 'mean' or any(c['cell']['ablation'] != 'mean' for c in manifest['cases']):
        raise ValueError('Mean-only study cannot use other discovery/evaluation ablations')
    from .protocol import graph_records
    if graph_records(model) != manifest['graph']:
        raise ValueError('Frozen graph differs from runtime graph')
    if any(w.mask_fn is not None or w.dropout_layer.p != 0 for w in model.dest_wrappers):
        raise ValueError('Stochastic masks/dropout unexpectedly enabled')
    return task


def prompt_positions(batch):
    from .protocol import positions
    correct, wrong = answer_ids(batch)
    result = [positions(row.cpu().numpy(), int(c), int(w))
              for row, c, w in zip(batch.clean, correct, wrong)]
    return torch.tensor(result, device=batch.clean.device), correct, wrong


def weights_hash(model):
    mask_ids = {id(p) for p in model.patch_masks.values()}
    digest = hashlib.sha256()
    for name, param in model.named_parameters():
        if id(param) not in mask_ids:
            digest.update(name.encode())
            digest.update(param.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def assert_same(actual, expected, label):
    if not torch.allclose(actual, expected, atol=1e-4, rtol=1e-4):
        raise ValueError(f'{label} parity failed: max error {(actual-expected).abs().max().item()}')


def smoke(output):
    from auto_circuit.prune import run_circuits
    manifest, masks = load_inventory(output)
    task = build_task(manifest)
    model = task.model
    initial = weights_hash(model)
    checks = []
    replacement = src_ablations(model, task.test_loader, ABLATION)
    replacement_outputs = output_reference(model, replacement)
    for case in manifest['cases']:
        c = case['cell']
        if c['seed'] != 42:
            continue
        scores = {k: v.float().to(task.device) for k, v in unwrap_scores(ROOT / case['scores']).items()}
        production = run_circuits(model, task.test_loader, manifest['budgets'], scores,
                                  PatchType.TREE_PATCH, ABLATION)
        conditions = {cond['budget']: cond for cond in manifest['conditions']
                      if cond['case'] == case['index'] and cond['phase'] == 'screen'}
        variants = {cond['label']: cond for cond in manifest['conditions']
                    if cond['case'] == case['index'] and cond['phase'] == 'causal' and cond['budget'] == manifest['primary_budget']}
        for batch in task.test_loader:
            p, good, bad = prompt_positions(batch)
            with torch.inference_mode():
                plain = model(batch.clean)[model.out_slice]
                full, full_cap = forward(model, batch)
                assert_same(full, plain, 'Observer neutrality')
                intact, intact_cap = forward(model, batch, manifest['graph'], np.ones(len(manifest['graph']), dtype=bool), replacement)
                assert_same(intact, full, 'Full patched graph')
                if float(output_metrics(intact, full, good, bad)['kl'].max()) >= 1e-5:
                    raise ValueError('Full patched graph KL check failed')
                diagnostics(model, intact, full, intact_cap, replacement_outputs, good, bad, p)
                baseline_logits, baseline_cap = forward(model, batch, manifest['graph'], masks[conditions[manifest['primary_budget']]['id']], replacement)
                diagnostics(model, full, full, full_cap, replacement_outputs, good, bad, p)
                for k in manifest['budgets']:
                    cond = conditions[k]
                    observed, cap = forward(model, batch, manifest['graph'], masks[cond['id']], replacement)
                    assert_same(observed, production[cond['attained']][batch.key], 'Production forward')
                    diagnostics(model, observed, full, cap, replacement_outputs, good, bad, p)
                for label, tag in [('restore_q_10_7', '10_7'), ('restore_q_11_10', '11_10')]:
                    cond = variants[label]
                    _, cap = forward(model, batch, manifest['graph'], masks[cond['id']], replacement)
                    for channel in ('k', 'v'):
                        assert_same(cap[channel + '_' + tag], baseline_cap[channel + '_' + tag], 'Individual Q '+channel)
                cond = variants['drop_nnm_out']
                _, cap = forward(model, batch, manifest['graph'], masks[cond['id']], replacement)
                for tag in ('10_7', '11_10'):
                    assert_same(cap['pattern_' + tag], baseline_cap['pattern_' + tag], 'Output-edge attention invariance')
                full_q_cond = variants['full_q_both']
                _, cap = forward(model, batch, manifest['graph'], masks[full_q_cond['id']], replacement,
                                 'both', full_cap)
                for tag in ('10_7', '11_10'):
                    assert_same(cap['q_input_' + tag], full_cap['q_input_' + tag], 'Intact query intervention')
            checks.append(dict(algo=c['algo'], batch_hash=fingerprint(batch.clean.tolist()),
                               production_parity=True, observer_neutral=True, qkv_invariance=True,
                               output_attention_invariance=True, reconstruction=True))
    if weights_hash(model) != initial:
        raise ValueError('Model weights changed during smoke')
    atomic_json(Path(output) / 'smoke.json', dict(study_id=manifest['study_id'], complete=True, checks=checks,
                model_weights_hash=initial, node=platform.node(), torch_version=torch.__version__,
                gpu=torch.cuda.get_device_name(), peak_gpu_bytes=torch.cuda.max_memory_allocated()))
    print('Smoke verified', len(checks), 'method-batch combinations', flush=True)


def atomic_arrays(path, arrays):
    tmp = path.with_name('.' + path.name + '.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def worker(output, phase, case_index):
    started = time.monotonic()
    output = Path(output)
    manifest, masks = load_inventory(output)
    smoke_result = json.loads((output / 'smoke.json').read_text())
    if not smoke_result.get('complete') or smoke_result['study_id'] != manifest['study_id']:
        raise ValueError('Scientific execution requires matching successful smoke checks')
    if not 0 <= case_index < len(manifest['cases']):
        raise ValueError('Invalid frozen case index')
    case = manifest['cases'][case_index]
    conditions = [c for c in manifest['conditions'] if c['case'] == case_index and c['phase'] == phase]
    if not conditions:
        print('No conditions', phase, case_index, flush=True)
        return
    result_dir = output / 'conditions'
    result_dir.mkdir(exist_ok=True)
    pending = []
    for cond in conditions:
        path = result_dir / f'{cond["id"]:05d}.json'
        if path.exists():
            saved = json.loads(path.read_text())
            array_path = result_dir / saved['array_file']
            if saved['study_id'] != manifest['study_id'] or saved['condition'] != cond or not saved['complete']:
                raise ValueError('Resume condition identity mismatch')
            if file_hash(array_path) != saved['arrays_sha256']:
                raise ValueError('Saved per-prompt arrays changed')
        else:
            pending.append(cond)
    if not pending:
        print('Already complete', phase, case_index, flush=True)
        return
    task = build_task(manifest)
    model = task.model
    initial = weights_hash(model)
    if initial != smoke_result['model_weights_hash']:
        raise ValueError('Runtime model weights differ from validated intact model')
    unique, owners = {}, {}
    for cond in pending:
        key = cond['mask_hash'], cond['hybrid']
        owner = unique.setdefault(key, cond)
        owners[cond['id']] = owner['id']
    collected = {c['id']: {} for c in unique.values()}
    refs, full_logits_saved, prompt_rows = {}, [], []
    with torch.inference_mode():
        replacement = src_ablations(model, task.test_loader, ABLATION)
        replacement_outputs = output_reference(model, replacement)
        for batch_number, batch in enumerate(task.test_loader):
            p, good, bad = prompt_positions(batch)
            full, full_cap = forward(model, batch)
            reference = diagnostics(model, full, full, full_cap, replacement_outputs, good, bad, p)
            for key, value in reference.items():
                refs.setdefault(key, []).append(value)
            full_logits_saved.append(full.cpu().numpy())
            for j in range(len(batch.clean)):
                prompt_rows.append(dict(index=batch_number * 64 + j, clean=batch.clean[j].cpu().tolist(),
                                        corrupt=batch.corrupt[j].cpu().tolist(), correct=int(good[j]), wrong=int(bad[j]),
                                        positions=p[j].cpu().tolist()))
            for cond in unique.values():
                logits, cap = forward(model, batch, manifest['graph'], masks[cond['id']], replacement,
                                      cond['hybrid'], full_cap)
                values = diagnostics(model, logits, full, cap, replacement_outputs, good, bad, p)
                for key, value in values.items():
                    collected[cond['id']].setdefault(key, []).append(value)
            print('Batch complete', phase, case_index, batch_number, 'unique conditions', len(unique), flush=True)
            del full_cap
    if weights_hash(model) != initial:
        raise ValueError('Model weights changed during scientific evaluation')
    reference = {k: np.concatenate(v) for k, v in refs.items()}
    reference_path = output / 'full_reference.npz'
    with (output / '.reference.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        prompt_document = dict(study_id=manifest['study_id'], prompts=prompt_rows)
        if reference_path.exists():
            if (output / 'reference.json').exists():
                validate_reference(output, manifest)
            with np.load(reference_path) as previous:
                for key in reference:
                    if not np.allclose(previous[key], reference[key], atol=1e-4, rtol=1e-4):
                        raise ValueError('Intact reference differs between workers: ' + key)
                if not np.allclose(previous['logits'], np.concatenate(full_logits_saved), atol=1e-4, rtol=1e-4):
                    raise ValueError('Intact logits differ between workers')
            if not (output / 'prompt_positions.json').exists():
                atomic_json(output / 'prompt_positions.json', prompt_document)
            elif json.loads((output / 'prompt_positions.json').read_text()) != prompt_document:
                raise ValueError('Prompt order or name positions differ between workers')
        else:
            atomic_json(output / 'prompt_positions.json', prompt_document)
            atomic_arrays(reference_path, {**reference, 'logits': np.concatenate(full_logits_saved)})
        if not (output / 'reference.json').exists():
            atomic_json(output / 'reference.json', dict(study_id=manifest['study_id'], complete=True,
                        checksums={name: file_hash(output / name) for name in ('full_reference.npz', 'prompt_positions.json')}))
        validate_reference(output, manifest)
    if float(reference['kl'].max()) >= 1e-5:
        raise ValueError('Intact graph KL check failed')
    retention = lambda mask: [bool(mask[next(i for i, e in enumerate(manifest['graph'])
                                           if e['src'] == f'A{l}.{h}' and e['dest'] == 'Resid End')]) for l, h in HEADS]
    array_files = {}
    for cond in unique.values():
        arrays = {k: np.concatenate(v) for k, v in collected[cond['id']].items()}
        randomized = cond['label'].startswith(('control_', 'removal_', 'fixed_', 'exact_'))
        if randomized:
            arrays.pop('vectors')
        name = f'{cond["id"]:05d}.npz'
        atomic_arrays(result_dir / name, arrays)
        array_files[cond['id']] = (name, file_hash(result_dir / name), arrays)
    for cond in pending:
        name, checksum, values = array_files[owners[cond['id']]]
        summary = {key: float(values[key].mean()) for key in METRICS}
        summary['logit_difference_percent'] = normalized_margin(values['margin'], reference['margin'])
        summary['io_minus_s1'] = (values['attention'][:, :, 0] - values['attention'][:, :, 1]).mean(0).tolist()
        atomic_json(result_dir / f'{cond["id"]:05d}.json', dict(study_id=manifest['study_id'], complete=True,
                    condition=cond, cell=case['cell'], array_file=name, arrays_sha256=checksum,
                    alias_of=owners[cond['id']] if owners[cond['id']] != cond['id'] else None,
                    output_retained=retention(masks[cond['id']]), summary=summary))
    atomic_json(output / f'worker_{phase}_{case_index:02d}.json', dict(study_id=manifest['study_id'], complete=True,
                elapsed_seconds=time.monotonic() - started, conditions=len(conditions), evaluated_unique=len(unique),
                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                peak_gpu_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                node=platform.node(), slurm_job=os.environ.get('SLURM_JOB_ID'), torch_version=torch.__version__))
    print('Worker complete', phase, case_index, 'seconds', time.monotonic() - started, flush=True)
