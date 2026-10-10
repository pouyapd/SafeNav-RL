"""
evaluate.py — evaluate a checkpoint or a baseline (wrapper around `python -m safenav evaluate`).

Usage:
    python scripts/evaluate.py --checkpoint runs/default/checkpoints/best.pt --episodes 100
    python scripts/evaluate.py --baseline potential_field --safety-filter off --out results/apf
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safenav.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["evaluate", *sys.argv[1:]]))
