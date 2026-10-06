"""Local actual-pool checks; requires the existing cached GPT-2 tokenizer."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from auto_circuit.data import load_datasets_from_json
from experiments.thesis.evaluation_protocol import build_evaluation_tasks, load_protocol, validate_records, fixed_evaluation_id

ROOT = Path(__file__).resolve().parents[1]


def test_greaterthan_default_generation_preserves_actual_pool():
    spec = importlib.util.spec_from_file_location('gt_generator', ROOT / 'datasets/greaterthan_generator.py')
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    tokenizer = AutoTokenizer.from_pretrained('gpt2', local_files_only=True)
    generated = generator.get_year_data(1000, SimpleNamespace(tokenizer=tokenizer))
    assert generated == json.loads((ROOT / 'datasets/greaterthan_gpt2-small_prompts.json').read_text())['prompts']


def test_actual_pools_have_content_disjoint_common_holdouts():
    tokenizer = AutoTokenizer.from_pretrained('gpt2', local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    encode = lambda s: tuple(tokenizer.encode(tokenizer.bos_token + s))
    for source in ('datasets/ioi/ioi_prompts.json', 'datasets/greaterthan_gpt2-small_prompts.json'):
        path = ROOT / source
        original = json.loads(path.read_text())['prompts']
        expected = None
        for seed in range(42, 47):
            train, test = load_datasets_from_json(SimpleNamespace(tokenizer=tokenizer), path,
                torch.device('cpu'), batch_size=64, train_test_size=(704, 256),
                random_seed=seed, split_seed=42, tail_divergence=False)
            meta = test.split_metadata
            clean = [encode(original[i]['clean']) for i in meta['test_row_ids']]
            training_inputs = {encode(original[i][role]) for i in meta['train_row_ids']
                               for role in ('clean', 'corrupt')}
            assert len(set(clean)) == 256
            assert set(clean).isdisjoint(training_inputs)
            if expected is None:
                expected = meta['test_clean_sha256']
            assert meta['test_clean_sha256'] == expected
            assert len(train) == 11 and len(test) == 4
        evaluation = SimpleNamespace(test_loader=test, batch_size=64, batch_count=(11, 4), model=SimpleNamespace(tokenizer=tokenizer))
        assert fixed_evaluation_id({'task': evaluation}).startswith('fixed_holdout_v1_')


def test_actual_temporary_repair_has_no_clean_input_exposure():
    path = ROOT / 'datasets/evaluation/temporary_repair_20261006/manifest.json'
    protocol = load_protocol(path)
    tokenizer = AutoTokenizer.from_pretrained('gpt2', local_files_only=True)
    encode = lambda s: tuple(tokenizer.encode(tokenizer.bos_token + s))
    for item in protocol['tasks'].values():
        records = json.loads((path.parent / item['dataset']).read_text())['prompts']
        original = json.loads((ROOT / item['source_dataset']).read_text())['prompts']
        assert validate_records(records, original, encode) == item['identity_checks']
    tasks = build_evaluation_tasks(protocol, ['ioi', 'gt'])
    assert all(task.split_seed is None and not task.dataset_shuffle and task.batch_count == (0, 4) for task in tasks.values())
    from experiments.thesis.run_cell import _build_task
    for short in ('ioi', 'gt'):
        future = _build_task(short, 43, False)
        assert future.split_seed == 42
        assert 'temporary_repair' not in future._dataset_name
