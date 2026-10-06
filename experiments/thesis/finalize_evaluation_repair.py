"""Merge completed repair groups and render previews outside historical results."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from .evaluation_protocol import ROOT


def main(output: Path):
    output = output.resolve()
    exhibits = output / 'exhibits'
    commands = [
        ['temporary_evaluation_repair', '--merge', '--output-dir', str(output)],
        ['results_chapter_size_curves', '--source-dir', str(output), '--output-dir', str(exhibits), '--tasks', 'ioi', 'gt'],
        ['results_chapter_controlled_comparison', '--output-root', str(exhibits)],
        ['results_chapter_stability', '--source-dir', str(output), '--output-root', str(exhibits)],
        ['results_chapter_evaluation_ablation', '--source-dir', str(output), '--output-root', str(exhibits)],
        ['results_chapter_completeness', '--source-dir', str(output), '--output-root', str(exhibits)],
        ['evaluation_repair_report', '--output-dir', str(output)],
    ]
    status = output / 'finalization.json'
    state = {'complete': False, 'completed_steps': []}
    status.write_text(json.dumps(state, indent=2))
    for command in commands:
        subprocess.run([sys.executable, '-m', 'experiments.thesis.' + command[0], *command[1:]],
                       cwd=ROOT, check=True)
        state['completed_steps'].append(command[0])
        status.write_text(json.dumps(state, indent=2))
    state['complete'] = True
    status.write_text(json.dumps(state, indent=2))
    print('Temporary repair and preview exports complete:', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/results_chapter/temporary_repair_20261006')
    main(parser.parse_args().output_dir)
