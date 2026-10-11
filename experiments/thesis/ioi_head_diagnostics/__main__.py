"""SLURM-friendly entry point; every measurement uses the frozen inventory."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=['inventory','smoke','screen','causal','size-sensitivity','summarize'])
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--cell-index', type=int)
    args = parser.parse_args()
    if args.phase == 'inventory':
        from .protocol import inventory
        inventory(args.output_dir)
    elif args.phase == 'smoke':
        from .evaluation import smoke
        smoke(args.output_dir)
    elif args.phase == 'summarize':
        from .analysis import summarize
        summarize(args.output_dir)
    else:
        if args.cell_index is None:
            parser.error('--cell-index is required for worker phases')
        from .evaluation import worker
        worker(args.output_dir, args.phase, args.cell_index)


if __name__ == '__main__':
    main()
