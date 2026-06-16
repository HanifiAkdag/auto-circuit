"""Re-render the interactive overlay HTMLs from a cached overlay.pkl.

The faithfulness analysis pass dumps the raw curve data to
``<experiment>/interactive/overlay.pkl``. This script reloads it and regenerates the
HTMLs — so styling tweaks in `interactive_curves.py` can be applied in seconds
without re-measuring any metric (no GPU, no model load).

    python experiments/replot_interactive.py results/faithfulness_comparison/<experiment>
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from faithfulness_comparison import _write_interactive


def main():
    parser = argparse.ArgumentParser(description="Re-render interactive overlay HTMLs from overlay.pkl.")
    parser.add_argument("experiment", help="Experiment folder (or its interactive/ subdir) holding overlay.pkl")
    args = parser.parse_args()

    folder = Path(args.experiment)
    pkl = folder / "interactive" / "overlay.pkl"
    if not pkl.exists():  # allow passing the interactive/ dir directly
        pkl = folder / "overlay.pkl"
        folder = folder.parent
    if not pkl.exists():
        sys.exit(f"No overlay.pkl found under {args.experiment}")

    with open(pkl, "rb") as f:
        overlay = pickle.load(f)
    print(f"Loaded {len(overlay)} (task, measure, metric) overlays from {pkl}")
    _write_interactive(folder, overlay)


if __name__ == "__main__":
    main()
