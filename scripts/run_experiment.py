from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dexa_nutri_stroke import load_config, run_experiment
from dexa_nutri_stroke.evaluate import subgroup_table
from dexa_nutri_stroke.schema import default_schema


def build_subgroups(valid_df) -> dict[str, np.ndarray]:
    d = valid_df
    groups = {
        "Hypertension: no": d["hypertension"] == 0,
        "Hypertension: yes": d["hypertension"] == 1,
        "Diabetes: no": d["diabetes"] == 0,
        "Diabetes: yes": d["diabetes"] == 1,
        "Atrial fibrillation: no": d["atrial_fibrillation"] == 0,
        "Atrial fibrillation: yes": d["atrial_fibrillation"] == 1,
        "Age < 65": d["age"] < 65,
        "Age >= 65": d["age"] >= 65,
        "Women": d["male"] == 0,
        "Men": d["male"] == 1,
        "NIHSS <= 4": d["initial_nihss"] <= 4,
        "NIHSS > 4": d["initial_nihss"] > 4,
    }
    if "onset_to_admission_hours" in d:
        groups["Onset to admission < 24 h"] = d["onset_to_admission_hours"] < 24
        groups["Onset to admission >= 24 h"] = d["onset_to_admission_hours"] >= 24
    if "dexa_delay_hours" in d:
        groups["DEXA < 48 h"] = d["dexa_delay_hours"] < 48
        groups["DEXA >= 48 h"] = d["dexa_delay_hours"] >= 48
    return {k: v.to_numpy() for k, v in groups.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    parser.add_argument("--outcome", default="unfavourable_outcome_3m")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.output is not None:
        cfg.output_dir = str(args.output)

    schema = default_schema(outcome=args.outcome)
    result = run_experiment(args.data, schema, cfg)

    y = result.dataset.valid[schema.outcome].to_numpy(dtype=int)
    table = subgroup_table(
        y,
        result.models["Clin"].valid_probabilities,
        result.models["Clin-DEXA-Nutri"].valid_probabilities,
        build_subgroups(result.dataset.valid),
        seed=cfg.seed,
    )
    if not table.empty and cfg.output_dir:
        out = Path(cfg.output_dir) / "subgroup_performance.csv"
        table.to_csv(out, index=False)
        print(f"[io] subgroup table written to {out}")

    print()
    print(result.summary().to_string(index=False))
    if cfg.output_dir:
        cfg.save(Path(cfg.output_dir) / "config_used.json")


if __name__ == "__main__":
    main()
