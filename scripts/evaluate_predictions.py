"""Evaluate saved, aligned model predictions without fitting or recalibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dexa_nutri_stroke.evaluate import compare_at_thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--clin-threshold", type=float, required=True)
    parser.add_argument("--full-threshold", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    columns = ["outcome", "clin_probability", "clin_dexa_nutri_probability"]
    frame = pd.read_csv(args.predictions, usecols=columns)
    result = compare_at_thresholds(
        frame["outcome"].to_numpy(),
        frame["clin_probability"].to_numpy(),
        frame["clin_dexa_nutri_probability"].to_numpy(),
        threshold_ref=args.clin_threshold,
        threshold_new=args.full_threshold,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
