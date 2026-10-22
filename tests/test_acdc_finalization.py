import json
import sys

import torch


def test_named_acdc_merge_is_idempotent(tmp_path, monkeypatch):
    from experiments import merge_acdc_partials as module
    partials = tmp_path / 'partials'
    partials.mkdir()
    for base, values in ((1, [1., 4.]), (3, [3., 2.])):
        name = f'cell_ioi_acdc_zero_seed42__base{base}-test'
        torch.save({'task': {'algo': {'m': torch.tensor(values)}}}, partials / (name + '.pkl'))
        (partials / (name + '_manifest.json')).write_text(json.dumps({
            'cell': {'split_seed': None}, 'cache_file': name + '.pkl', 'data_budget': None}))
    output = tmp_path / 'scores'
    monkeypatch.setattr(sys, 'argv', ['merge', '--task', 'ioi', '--ablation', 'zero',
        '--legacy-split', '--partials-dir', str(partials), '--output-dir', str(output),
        '--expect', '2', '--output-name', 'complete'])
    module.main()
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    module.main()
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before
    payload = torch.load(output / 'complete.pkl', weights_only=True)
    assert torch.equal(payload['task']['algo']['m'], torch.tensor([1., 2.]))
