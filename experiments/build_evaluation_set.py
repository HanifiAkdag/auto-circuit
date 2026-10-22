"""Create a temporary unseen-clean evaluation artifact for historical score caches."""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoTokenizer

from .evaluation_protocol import ROOT, fingerprint, protocol_id, sha256, validate_records


def _generator(relative: str):
    spec = importlib.util.spec_from_file_location('evaluation_generator', ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ioi_candidates(original, generator, seed):
    """Preserve empirical non-name geometry and use the released XYZ corruption."""
    rng = random.Random(seed)
    for attempt in range(100000):
        source = rng.choice(original)
        old_io = source['answers'][0].strip()
        old_subject = source['wrong_answers'][0].strip()
        io, subject = rng.sample(generator.NAMES, 2)
        words = source['clean'].split(' ')
        words = [io if word == old_io else subject if word == old_subject else word for word in words]
        clean = ' '.join(words)
        prompt = {'text': clean + ' ' + io, 'IO': io, 'S': subject}
        name_order = [word for word in words if word in (io, subject)]
        if name_order not in ([io, subject, subject], [subject, io, subject]):
            raise ValueError('Unrecognized IOI name/template construction')
        order = 'ABBA' if name_order[0] == io else 'BABA'
        corrupt = generator.gen_flipped_prompts([prompt], [order], 'ABB->XYZ, BAB->XYZ', generator.NAMES, seed + attempt + 1)[0]['text']
        yield {'clean': clean, 'corrupt': corrupt.rsplit(' ', 1)[0],
               'answers': [' ' + io], 'wrong_answers': [' ' + subject]}


def build(output: Path, seed: int = 20261006, count: int = 256, tasks=("ioi", "gt")) -> dict:
    if count != 256:
        raise ValueError('The historical repair preserves the 256-example evaluation budget')
    if (output / 'manifest.json').exists():
        raise ValueError('Refusing to overwrite an existing frozen evaluation artifact')
    resources = {
        'ioi': ('datasets/ioi/ioi_prompts.json', 'datasets/ioi/ioi_generator.py'),
        'gt': ('datasets/greaterthan_gpt2-small_prompts.json', 'datasets/greaterthan_generator.py'),
        'docstring': ('datasets/docstring_prompts.json', 'datasets/docstring_generator.py'),
    }
    if not tasks or len(set(tasks)) != len(tasks) or any(task not in resources for task in tasks):
        raise ValueError('Specify distinct supported evaluation tasks')
    if 'docstring' in tasks and len(tasks) != 1:
        raise ValueError('Docstring requires a separate artifact with its model tokenizer')
    tokenizer_name = 'NeelNanda/gpt-neox-tokenizer-digits' if 'docstring' in tasks else 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    encode = lambda text: tuple(tokenizer.encode(tokenizer.bos_token + text))
    manifest = {'schema': 1, 'purpose': 'temporary_historical_repair',
                'generation_seed': seed, 'examples_per_task': count,
                'identity_policy': 'unique evaluation clean token sequences absent from both input roles of the entire original pool; shared corrupt contexts permitted',
                'tokenizer': {'name': tokenizer_name, 'vocabulary_sha256': fingerprint(tokenizer.get_vocab()), 'prepend_bos': True},
                'tasks': {}}
    output.mkdir(parents=True, exist_ok=True)
    for short in tasks:
        source_path, generator_path = resources[short]
        original_data = json.loads((ROOT / source_path).read_text())
        original = original_data['prompts']
        generator = _generator(generator_path)
        if short == 'ioi':
            candidates = _ioi_candidates(original, generator, seed)
            recipe = 'Empirical original template/order/place/object geometry; fresh uniform ordered distinct names; released ABB/BAB to XYZ corruption with original answer spacing.'
        elif short == 'gt':
            candidates = iter(generator.get_year_data(10000, SimpleNamespace(tokenizer=tokenizer), seed=seed))
            recipe = 'Released GreaterThanConstants, noun/year sampling and century01 corruption; explicit independent seed.'
        else:
            candidates = iter(generator.get_docstring_data(10000, seed=seed))
            recipe = 'Historical rest-style Docstring generator: three matching arguments, two definition-prefix arguments, one definition-suffix argument, no doc-prefix arguments, description lengths three/two and random_random corruption; explicit independent seed.'
        prohibited = {encode(row[role]) for row in original for role in ('clean', 'corrupt')}
        accepted, identities, rejected = [], set(), Counter()
        for candidate in candidates:
            identity = encode(candidate['clean'])
            if identity in prohibited:
                rejected['historical_clean_input_match'] += 1
                continue
            if identity in identities:
                rejected['internal_clean_input_duplicate'] += 1
                continue
            for role in ('answers', 'wrong_answers'):
                if not candidate[role] or any(len(tokenizer.encode(answer)) != 1 for answer in candidate[role]):
                    raise ValueError('Candidate has invalid answer tokenization')
            identities.add(identity)
            accepted.append(candidate)
            if len(accepted) == count:
                break
        if len(accepted) != count:
            raise ValueError(f'Insufficient valid {short} candidates: {len(accepted)}, {dict(rejected)}')
        stats = validate_records(accepted, original, encode)
        path = output / f'{short}.json'
        metadata = {k: v for k, v in original_data.items() if k != 'prompts'}
        path.write_text(json.dumps({**metadata, 'prompts': accepted}, indent=2) + '\n')
        summary = {'clean_token_lengths': dict(Counter(len(encode(row['clean'])) for row in accepted))}
        if short == 'gt':
            summary['centuries'] = dict(Counter(row['clean'].split()[-1] for row in accepted))
        manifest['tasks'][short] = {'dataset': path.name, 'sha256': sha256(path),
            'source_dataset': source_path, 'source_sha256': sha256(ROOT / source_path),
            'generator': generator_path, 'generator_sha256': sha256(ROOT / generator_path),
            'recipe': recipe, 'rejected': dict(rejected), 'identity_checks': stats,
            'sampling_summary': summary,
            'record_sha256': [fingerprint(row) for row in accepted]}
        print(short, json.dumps(stats))
    manifest['protocol_id'] = protocol_id(manifest)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'datasets/evaluation/temporary_repair_20261006')
    parser.add_argument('--seed', type=int, default=20261006)
    parser.add_argument('--tasks', nargs='+', choices=('ioi', 'gt', 'docstring'), default=('ioi', 'gt'))
    args = parser.parse_args()
    build(args.output_dir, args.seed, tasks=args.tasks)
