"""
train.py — SafeNav-RL training entry point (wrapper around `python -m safenav train`).

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/default_config.yaml --timesteps 500000 --seed 1
    python scripts/train.py --no-safety-filter --run-dir runs/no_filter
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safenav.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["train", *sys.argv[1:]]))
