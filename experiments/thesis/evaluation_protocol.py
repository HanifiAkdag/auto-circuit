"""Identity checks and explicit Tasks for temporary repair and fixed holdouts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def protocol_id(value: dict) -> str:
    return fingerprint({k: v for k, v in value.items()
                        if k not in ('protocol_id', '_manifest_path', 'created_at')})


def validate_records(records: list[dict], original: list[dict], encode: Callable) -> dict:
    """Require novel unique clean inputs; report corrupt-context reuse separately."""
    original_inputs = {tuple(encode(row[role])) for row in original for role in ('clean', 'corrupt')}
    original_records = {fingerprint(row) for row in original}
    clean = [tuple(encode(row['clean'])) for row in records]
    corrupt = [tuple(encode(row['corrupt'])) for row in records]
    if len(set(clean)) != len(clean):
        raise ValueError('Evaluation contains duplicate clean token inputs')
    overlap = sum(tokens in original_inputs for tokens in clean)
    if overlap:
        raise ValueError(f'{overlap} evaluation clean inputs occurred in discovery inputs')
    record_overlap = sum(fingerprint(row) in original_records for row in records)
    if record_overlap:
        raise ValueError('Evaluation contains complete discovery records')
    return {'examples': len(records), 'clean_overlap_rows': overlap,
            'complete_record_overlap_rows': record_overlap,
            'corrupt_overlap_rows': sum(tokens in original_inputs for tokens in corrupt),
            'ordered_clean_tokens_sha256': fingerprint(clean),
            'ordered_corrupt_tokens_sha256': fingerprint(corrupt)}


def load_protocol(path: Path) -> dict:
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if value.get('purpose') != 'temporary_historical_repair':
        raise ValueError('This manifest must explicitly identify temporary_historical_repair')
    if value.get('protocol_id') != protocol_id(value):
        raise ValueError('Evaluation protocol hash mismatch')
    for item in value['tasks'].values():
        artifact = path.parent / item['dataset']
        if sha256(artifact) != item['sha256']:
            raise ValueError(f'Evaluation artifact hash mismatch: {artifact}')
        if 'source_dataset' in item and sha256(ROOT / item['source_dataset']) != item['source_sha256']:
            raise ValueError('Original discovery pool hash changed')
    value['_manifest_path'] = str(path)
    return value


def build_evaluation_tasks(protocol: dict | None, task_shorts: list[str], split_seed: int = 42) -> dict:
    from .run_cell import _TASK_TEMPLATES
    result = {}
    for short in task_shorts:
        template = _TASK_TEMPLATES[short]
        if protocol is None:
            task = replace(template, seed=42, split_seed=split_seed, __init_complete__=False)
        else:
            item = protocol['tasks'][short]
            path = Path(protocol['_manifest_path']).parent / item['dataset']
            dataset_name = str(path.resolve().relative_to(ROOT / 'datasets').with_suffix(''))
            task = replace(template, _dataset_name=dataset_name, seed=42, split_seed=None,
                           dataset_shuffle=False, batch_size=64, batch_count=(0, 4),
                           __init_complete__=False)
        result[task.key] = task
    return result


def verify_evaluation_task(task, protocol: dict, short: str) -> None:
    """Recheck identities with the tokenizer actually used for model evaluation."""
    item = protocol['tasks'][short]
    records = json.loads((Path(protocol['_manifest_path']).parent / item['dataset']).read_text())['prompts']
    original = json.loads((ROOT / item['source_dataset']).read_text())['prompts']
    tokenizer = task.model.tokenizer
    stats = validate_records(records, original, lambda s: tokenizer.encode(tokenizer.bos_token + s))
    if stats != item['identity_checks'] or stats['examples'] != 256:
        raise ValueError('Runtime evaluation token identities disagree with the frozen protocol')


def require_same_protocol(*manifests: dict) -> None:
    if any(m.get('complete') is False for m in manifests):
        raise ValueError('Incomplete functional source bundle')
    ids = {m.get('evaluation_protocol_id') for m in manifests}
    if len(ids) != 1 or None in ids:
        raise ValueError('Evaluation protocol mismatch between functional sources')


def verify_holdout_identity(manifests: list[dict], expected: dict, budget: dict | None = None) -> None:
    for source in manifests:
        actual = source.get('data_split') or {}
        fields = ('protocol', 'split_seed', 'pool_sha256', 'test_clean_sha256')
        if any(actual.get(k) != expected.get(k) for k in fields):
            raise ValueError('Discovery/evaluation holdout identities disagree')
        if source.get('cell', {}).get('split_seed') != expected.get('split_seed'):
            raise ValueError('Discovery/evaluation holdout seeds disagree')
        if budget is not None and source.get('data_budget') != budget:
            raise ValueError('Discovery/evaluation data budgets disagree')


def fixed_evaluation_id(tasks: dict) -> str:
    """Identify the actual ordered evaluation inputs and loader construction."""
    content = {}
    def serialize(value):
        return [serialize(item) for item in value] if isinstance(value, (list, tuple)) else value.detach().cpu().tolist()
    for key, task in sorted(tasks.items()):
        batches = [{field: serialize(getattr(batch, field))
                    for field in ('clean', 'corrupt', 'answers', 'wrong_answers')}
                   for batch in task.test_loader]
        content[key] = {'split': task.test_loader.split_metadata, 'batches': batches,
                        'batch_size': task.batch_size, 'batch_count': task.batch_count,
                        'tokenizer': fingerprint(task.model.tokenizer.get_vocab())}
        # Training draws vary by seed; only the reserved evaluation membership matters.
        content[key]['split'] = {k: v for k, v in content[key]['split'].items() if k != 'train_row_ids'}
    return 'fixed_holdout_v1_' + fingerprint(content)


def load_frozen_inputs(path: Path) -> list[dict]:
    items = json.loads(Path(path).read_text())['inputs']
    for item in items:
        for field in ('scores', 'manifest'):
            source = ROOT / item[field]
            if sha256(source) != item[field + '_sha256']:
                raise ValueError(f'Frozen {field} hash changed: {source}')
    return items
