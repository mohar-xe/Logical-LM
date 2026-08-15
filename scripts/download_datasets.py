#!/usr/bin/env python3
"""Download + normalize all five Logic-LM datasets into ``data/raw/``.

Usage:
    uv run scripts/download_datasets.py [--data-dir data]
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logiclm.datasets import download_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()
    counts = download_all(data_dir=args.data_dir)
    print("Downloaded datasets:")
    for name, count in counts.items():
        print(f"  {name}: {count} examples")


if __name__ == "__main__":
    main()
