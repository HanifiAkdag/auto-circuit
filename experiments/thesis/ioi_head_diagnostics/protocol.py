"""Circuit selection, controls, and immutable input inventory."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
NNM = {'A10.7', 'A11.10'}
NM = {'A9.6', 'A9.9', 'A10.0'}
SI = {'A7.3', 'A7.9', 'A8.6', 'A8.10'}
BNM = {'A10.10', 'A10.6', 'A10.2', 'A10.1', 'A11.2', 'A9.7', 'A9.0', 'A11.9'}
BUDGETS = (739, 2000, 3000)
ALGOS = ('acdc', 'discogp-ep', 'eap', 'ep')
PROTOCOL_FILE = 'docs/writing/ioi_retained_head_mean_protocol_20261011.md'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hash():
    dependencies = list((ROOT / 'auto_circuit').rglob('*.py'))
    dependencies += [ROOT / 'experiments/thesis' / p for p in ('evaluation_protocol.py', 'run_cell.py')]
    dependencies += list(Path(__file__).parent.glob('*.py'))
    return fingerprint({str(p.relative_to(ROOT)): file_hash(p) for p in sorted(dependencies)})


def validate_reference(output, manifest):
    output = Path(output)
    saved = json.loads((output / 'reference.json').read_text())
    if saved.get('study_id') != manifest['study_id'] or not saved.get('complete'):
        raise ValueError('Intact reference identity mismatch')
    for name, checksum in saved['checksums'].items():
        if file_hash(output / name) != checksum:
            raise ValueError('Intact reference checksum mismatch: ' + name)


def validate_identity(manifest):
    content = {k: v for k, v in manifest.items() if k not in ('study_id', 'masks_file_hash')}
    if fingerprint(content) != manifest['study_id']:
        raise ValueError('Frozen inventory content identity mismatch')


def rng(identity):
    seed = int(fingerprint([20261011, identity])[:16], 16)
    return np.random.default_rng(seed)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(tmp, path)


def native_mask(scores, k):
    values = np.abs(np.asarray(scores))
    # ACDC deliberately leaves retained edges at +inf. NaN is never a ranking.
    if np.isnan(values).any() or not 0 < k <= len(values):
        raise ValueError('NaN scores or invalid budget')
    threshold = np.partition(values, len(values) - k)[len(values) - k]
    return values >= threshold


def exact_mask(scores, k, realization):
    values = np.abs(np.asarray(scores))
    base = native_mask(values, k)
    threshold = values[base].min()
    mask = values > threshold
    # A single ordering for every budget preserves nesting within a realization.
    order = rng(['tie', realization]).permutation(len(values))
    ties = order[values[order] == threshold]
    mask[ties[:k - int(mask.sum())]] = True
    return mask


def positions(tokens, io, subject):
    io_pos = np.flatnonzero(np.asarray(tokens) == io)
    s_pos = np.flatnonzero(np.asarray(tokens) == subject)
    if len(io_pos) != 1 or len(s_pos) != 2:
        raise ValueError(f'Ambiguous name occurrences: IO={io_pos.tolist()}, S={s_pos.tolist()}')
    return int(io_pos[0]), int(s_pos[0]), int(s_pos[1])


def normalized_margin(margin, full_margin):
    denominator = float(np.mean(full_margin))
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError('Full-model average margin must be positive')
    return float(100 * np.mean(margin) / denominator)


def head_of(dest):
    return dest.rsplit('.', 1)[0] if dest.startswith('A') else dest


def make_interventions(edges, base):
    out = {}
    def change(label, predicate, keep):
        mask = base.copy()
        mask[[i for i, e in enumerate(edges) if predicate(e)]] = keep
        out[label] = mask
    for head, tag in [('A10.7', '10_7'), ('A11.10', '11_10')]:
        change('restore_q_' + tag, lambda e, h=head: e['dest'] == h + '.Q', True)
        change('drop_out_' + tag, lambda e, h=head: e['src'] == h and e['dest'] == 'Resid End', False)
    for channel in ('Q', 'K', 'V'):
        change('restore_' + channel.lower() + '_both',
               lambda e, c=channel: head_of(e['dest']) in NNM and e['channel'] == c, True)
    change('restore_qkv_both', lambda e: head_of(e['dest']) in NNM, True)
    change('drop_nnm_out', lambda e: e['src'] in NNM and e['dest'] == 'Resid End', False)
    mask = out['restore_q_both'].copy()
    mask[[i for i, e in enumerate(edges) if e['src'] in NNM and e['dest'] == 'Resid End']] = False
    out['restore_q_drop_out'] = mask
    for group, heads in [('nm', NM), ('bnm', BNM), ('nnm_nm', NNM | NM)]:
        change('drop_' + group + '_out', lambda e, hs=heads: e['src'] in hs and e['dest'] == 'Resid End', False)
    for group, predicate in [('nm', lambda s: s in NM), ('si', lambda s: s in SI),
                             ('other', lambda s: s not in NM | SI)]:
        change('restore_q_' + group,
               lambda e, p=predicate: head_of(e['dest']) in NNM and e['channel'] == 'Q' and p(e['src']), True)
    for keep, tag in [(True, 'keep'), (False, 'drop')]:
        change('interhead_' + tag, lambda e: e['src'] == 'A10.7' and e['dest'] == 'A11.10.Q', keep)
    for tag in ('10_7', '11_10', 'both'):
        out['full_q_' + tag] = base.copy()
    return out


def stratum(e):
    return e['layer'], e['channel'], e['kind']


def control_sets(edges, base, added, identity):
    counts = {}
    for i in np.flatnonzero(added):
        key = stratum(edges[i])
        counts[key] = counts.get(key, 0) + 1
    pools = {key: [i for i, e in enumerate(edges) if not base[i]
                  and head_of(e['dest']) not in NNM and stratum(e) == key] for key in counts}
    deficits = {str(key): n - len(pools[key]) for key, n in counts.items() if len(pools[key]) < n}
    if deficits:
        raise ValueError(f'Unavailable exact restoration control: {deficits}')
    gen = rng(['control', identity])
    return [sorted(int(i) for key, n in sorted(counts.items())
                   for i in gen.choice(pools[key], n, replace=False)) for _ in range(20)]


def replacement_sets(edges, scores, base, added, identity):
    m = int(added.sum())
    eligible = [i for i, e in enumerate(edges) if base[i] and e['src'] not in NNM
                and head_of(e['dest']) not in NNM]
    if len(eligible) < m:
        raise ValueError('Insufficient protected fixed-budget removal pool')
    ordered = sorted(eligible, key=lambda i: (abs(scores[i]), edges[i]['name']))
    pool = ordered[:max(m, math.ceil(.2 * len(eligible)))]
    gen = rng(['replacement', identity])
    pairs = []
    for _ in range(10):
        removed = base.copy()
        removed[gen.choice(pool, m, replace=False)] = False
        restored = removed | added
        assert restored.sum() == base.sum()
        pairs.append((removed, restored))
    return pairs


def graph_records(model):
    records = []
    for e in sorted(model.edges, key=lambda e: e.name):
        assert e.seq_idx is None
        dest = e.dest.name
        records.append(dict(name=e.name, src=e.src.name, dest=dest,
                            layer=int(dest.split('.')[0][1:]) if dest.startswith('A') else 12,
                            channel=dest.rsplit('.', 1)[-1] if dest.startswith('A') else 'output',
                            kind='attention' if e.src.name.startswith('A') else 'mlp' if e.src.name.startswith('MLP') else 'input',
                            module=e.dest.module_name, index=list(e.patch_idx)))
    return records


def score_vector(scores, edges):
    return np.array([float(scores[e['module']][tuple(e['index'])]) for e in edges])


def unwrap_scores(path):
    import torch
    value = torch.load(path, map_location='cpu', weights_only=True)
    return next(iter(next(iter(value.values())).values()))


def plan_case(edges, values, case_index, identity, budgets=BUDGETS):
    conditions, masks, status = [], [], {}
    def add(label, mask, base, phase, budget, realization=None):
        idx = len(masks)
        masks.append(mask)
        conditions.append(dict(id=idx, case=case_index, label=label, phase=phase, budget=budget,
                               realization=realization, attained=int(mask.sum()),
                               added=np.flatnonzero(mask & ~base).tolist(),
                               removed=np.flatnonzero(base & ~mask).tolist(),
                               mask_hash=hashlib.sha256(mask.tobytes()).hexdigest(),
                               hybrid=label.removeprefix('full_q_') if label.startswith('full_q_') else None))
    for k in budgets:
        base = native_mask(values, k)
        add('baseline', base, base, 'screen', k)
        variants = make_interventions(edges, base)
        for label, mask in variants.items():
            add(label, mask, base, 'causal', k)
        added = variants['restore_q_both'] & ~base
        control_identity = [identity, k]
        try:
            controls = control_sets(edges, base, added, control_identity)
            status[str(k)] = 'available'
            for j, restored in enumerate(controls):
                mask = base.copy()
                mask[restored] = True
                add(f'control_{j:02d}', mask, base, 'causal', k)
        except ValueError as exc:
            status[str(k)] = str(exc)
        for j, (removed, restored) in enumerate(replacement_sets(edges, values, base, added, control_identity)):
            add(f'removal_{j:02d}', removed, base, 'causal', k)
            add(f'fixed_q_{j:02d}', restored, base, 'causal', k)
        if base.sum() != k:
            for j in range(10):
                tie = exact_mask(values, k, [identity, j])
                add('exact_baseline', tie, tie, 'size-sensitivity', k, j)
                variants = make_interventions(edges, tie)
                for label in ('restore_q_both', 'drop_nnm_out', 'restore_q_drop_out'):
                    add('exact_' + label, variants[label], tie, 'size-sensitivity', k, j)
    return conditions, masks, status


def inventory(output):
    from experiments.thesis.evaluation_protocol import load_protocol, load_frozen_inputs, build_evaluation_tasks
    output = Path(output)
    active_path = ROOT / 'results/results_chapter/active_sources.json'
    active = json.loads(active_path.read_text())['bundles']['main']
    eval_path = Path(active['evaluation_manifest'])
    selected_path = Path(active['selected_inputs'])
    protocol = load_protocol(eval_path)
    items = load_frozen_inputs(selected_path)
    selected = []
    for item in items:
        c = item['cell']
        if c['task'] == 'ioi' and c['algo'] in ALGOS and c['ablation'] == 'mean' and not c.get('null_test'):
            seed = int(c['seed'])
            if seed in range(42, 47) and int(c.get('init_seed', seed)) == seed:
                selected.append(item)
    selected.sort(key=lambda x: (x['cell']['algo'], x['cell']['ablation'], x['cell']['seed']))
    expected = {(a, 'mean', s) for a in ALGOS for s in range(42, 47)}
    identities = [(i['cell']['algo'], i['cell']['ablation'], i['cell']['seed']) for i in selected]
    if len(identities) != 20 or set(identities) != expected:
        raise ValueError('Missing or duplicate frozen discovery identities')
    task = next(iter(build_evaluation_tasks(protocol, ['ioi']).values()))
    model = task.model
    import importlib.metadata
    runtime = dict(dtype=str(next(model.parameters()).dtype),
                   model_config=json.loads(json.dumps({k: v for k, v in model.cfg.to_dict().items() if k != 'device'}, default=str)),
                   packages={p: importlib.metadata.version(p) for p in ('torch', 'transformer-lens', 'numpy', 'transformers')},
                   tokenizer_class=type(model.tokenizer).__name__,
                   tokenizer_vocab_hash=fingerprint(model.tokenizer.get_vocab()),
                   batch_size=64, tf32=False, component_graph=True)
    edges = graph_records(model)
    masks, cases, conditions = [], [], []
    for case_index, item in enumerate(selected):
        scores = unwrap_scores(ROOT / item['scores'])
        if set(scores) != set(model.patch_masks) or any(scores[k].shape != model.patch_masks[k].shape for k in scores):
            raise ValueError('Cache score shape mismatch')
        values = score_vector(scores, edges)
        planned, case_masks, status = plan_case(edges, values, case_index, identities[case_index])
        offset = len(masks)
        conditions.extend(dict(c, id=c['id'] + offset) for c in planned)
        masks.extend(case_masks)
        case = dict(index=case_index, **item, control_status=status)
        cases.append(case)
    manifest = dict(schema=2, discovery_ablation='mean', evaluation_ablation='mean',
                    budgets=list(BUDGETS), primary_budget=2000, attribution_reference='mean_replacement',
                    evaluation_protocol_id=protocol['protocol_id'],
                    active_sources_hash=file_hash(active_path), evaluation_manifest=str(eval_path),
                    evaluation_manifest_hash=file_hash(eval_path), selected_inputs=str(selected_path),
                    selected_inputs_hash=file_hash(selected_path), code_hash=code_hash(),
                    source_dir=active['source_dir'], graph=edges, cases=cases, conditions=conditions,
                    runtime=runtime,
                    protocol_file=PROTOCOL_FILE, protocol_file_hash=file_hash(ROOT / PROTOCOL_FILE))
    manifest['study_id'] = fingerprint(manifest)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'inventory.json'
    if path.exists():
        saved = json.loads(path.read_text())
        validate_identity(saved)
        if saved['study_id'] != manifest['study_id'] or file_hash(output / 'masks.npz') != saved['masks_file_hash']:
            raise ValueError('Refusing to overwrite a different frozen study inventory')
        manifest = saved
    else:
        tmp = output / '.masks.tmp.npz'
        np.savez_compressed(tmp, masks=np.stack(masks))
        os.replace(tmp, output / 'masks.npz')
        manifest['masks_file_hash'] = file_hash(output / 'masks.npz')
        # File integrity is separate from the content-derived scientific identity.
        atomic_json(path, manifest)
    print(json.dumps(dict(study_id=manifest['study_id'], cases=len(cases), conditions=len(conditions),
                          phases={p: sum(c['phase'] == p for c in conditions) for p in ('screen', 'causal', 'size-sensitivity')})))
    return manifest


def load_inventory(output):
    manifest = json.loads((Path(output) / 'inventory.json').read_text())
    validate_identity(manifest)
    if manifest['code_hash'] != code_hash():
        raise ValueError('Code differs from frozen inventory; use a new output directory and record the amendment')
    for path, key in [(manifest['evaluation_manifest'], 'evaluation_manifest_hash'),
                      (manifest['selected_inputs'], 'selected_inputs_hash'),
                      (ROOT / 'results/results_chapter/active_sources.json', 'active_sources_hash'),
                      (ROOT / manifest['protocol_file'], 'protocol_file_hash'),
                      (Path(output) / 'masks.npz', 'masks_file_hash')]:
        if file_hash(path) != manifest[key]:
            raise ValueError(f'Frozen input changed: {path}')
    for item in manifest['cases']:
        for field in ('scores', 'manifest'):
            if file_hash(ROOT / item[field]) != item[field + '_sha256']:
                raise ValueError(f'Frozen source changed: {item[field]}')
    return manifest, np.load(Path(output) / 'masks.npz')['masks']
