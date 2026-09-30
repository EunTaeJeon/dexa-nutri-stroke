from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(n: int = 900, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(68, 12, n), 30, 95)
    male = rng.binomial(1, 0.59, n)
    bmi = np.clip(rng.normal(23.8, 3.2, n), 14, 40)

    asm = np.clip(
        np.where(male == 1, 7.4, 5.9) + 0.13 * (bmi - 23.8) - 0.012 * (age - 68)
        + rng.normal(0, 0.75, n),
        3.0, 11.0,
    )
    ffmi = np.clip(0.75 * asm + 11.5 + rng.normal(0, 0.9, n), 11, 24)
    fmi = np.clip(bmi - ffmi + rng.normal(0, 0.8, n), 1.0, 20.0)

    albumin = np.clip(rng.normal(4.1, 0.45, n) - 0.004 * (age - 68), 2.2, 5.3)
    lymphocyte = np.clip(rng.lognormal(np.log(1.8), 0.4, n), 0.2, 6.0)
    pni = 10 * albumin + 0.005 * (lymphocyte * 1000)
    fu_pni = pni - np.clip(rng.normal(1.2, 2.5, n), -6, 10)

    nihss = np.clip(rng.poisson(np.clip(2.2 + 0.03 * (age - 68), 0.5, 12), n), 0, 30)
    pre_mrs = rng.choice([0, 1, 2, 3, 4], n, p=[0.80, 0.09, 0.05, 0.04, 0.02])

    htn = rng.binomial(1, 0.67, n)
    dm = rng.binomial(1, 0.37, n)
    af = rng.binomial(1, 0.18, n)
    smoking = rng.binomial(1, 0.26, n)
    cad = rng.binomial(1, 0.08, n)
    chf = rng.binomial(1, 0.05, n)
    cancer = rng.binomial(1, 0.07, n)
    dialysis = rng.binomial(1, 0.02, n)
    infection = rng.binomial(1, 0.09, n)
    oha = rng.binomial(1, 0.28, n)

    glucose = np.clip(rng.lognormal(np.log(120), 0.28, n) + 25 * dm, 60, 450)
    hba1c = np.clip(rng.normal(5.9, 0.9, n) + 1.1 * dm, 4.2, 13.0)
    crp = np.clip(rng.lognormal(np.log(0.3), 1.2, n), 0.01, 30)
    bun = np.clip(rng.normal(16, 6, n), 4, 70)
    egfr = np.clip(rng.normal(85, 22, n) - 0.4 * (age - 68), 8, 150)
    d_dimer = np.clip(rng.lognormal(np.log(0.6), 1.0, n), 0.05, 40)
    inr = np.clip(rng.normal(1.02, 0.12, n), 0.8, 3.5)
    bilirubin = np.clip(rng.lognormal(np.log(0.65), 0.4, n), 0.1, 5)
    alp = np.clip(rng.normal(78, 28, n), 25, 400)
    ffa = np.clip(rng.lognormal(np.log(600), 0.5, n), 60, 3000)
    homocysteine = np.clip(rng.normal(12, 5, n), 3, 60)
    toast = rng.choice([1, 2, 3, 4, 5], n, p=[0.22, 0.17, 0.31, 0.06, 0.24])
    onset_to_admission = np.clip(rng.lognormal(np.log(9), 1.2, n), 0.2, 200)
    dexa_delay = np.clip(rng.lognormal(np.log(30), 0.8, n), 1, 300)

    logit = (
        -0.95
        + 0.30 * nihss
        + 0.55 * pre_mrs
        + 0.030 * (age - 68)
        - 0.34 * (asm - 6.8)
        - 0.045 * (fu_pni - 48)
        + 0.55 * infection
        + 0.22 * dm
        + 0.004 * (glucose - 120)
        - 0.006 * (egfr - 85)
    )
    outcome = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit)))

    df = pd.DataFrame(
        {
            "age": np.round(age, 0),
            "male": male,
            "bmi": np.round(bmi, 1),
            "hypertension": htn,
            "diabetes": dm,
            "atrial_fibrillation": af,
            "coronary_artery_disease": cad,
            "congestive_heart_failure": chf,
            "cancer": cancer,
            "dialysis": dialysis,
            "smoking": smoking,
            "pre_admission_mrs": pre_mrs,
            "initial_nihss": nihss,
            "toast_subtype": toast,
            "onset_to_admission_hours": np.round(onset_to_admission, 1),
            "dexa_delay_hours": np.round(dexa_delay, 1),
            "in_hospital_infection": infection,
            "pre_admission_oha": oha,
            "glucose": np.round(glucose, 0),
            "hba1c": np.round(hba1c, 1),
            "crp": np.round(crp, 2),
            "bun": np.round(bun, 1),
            "egfr": np.round(egfr, 1),
            "d_dimer": np.round(d_dimer, 2),
            "inr": np.round(inr, 2),
            "total_bilirubin": np.round(bilirubin, 2),
            "alkaline_phosphatase": np.round(alp, 0),
            "free_fatty_acid": np.round(ffa, 0),
            "homocysteine": np.round(homocysteine, 1),
            "asm_index": np.round(asm, 2),
            "fat_free_mass_index": np.round(ffmi, 2),
            "fat_mass_index": np.round(fmi, 2),
            "pni_initial": np.round(pni, 1),
            "pni_followup": np.round(fu_pni, 1),
            "unfavourable_outcome_3m": outcome,
        }
    )

    for col, rate in [
        ("homocysteine", 0.18),
        ("free_fatty_acid", 0.15),
        ("d_dimer", 0.12),
        ("hba1c", 0.10),
        ("pni_followup", 0.34),
        ("crp", 0.06),
    ]:
        df.loc[rng.random(n) < rate, col] = np.nan

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=900)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("data/synthetic_cohort.csv"))
    args = parser.parse_args()

    df = generate(args.n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(
        f"wrote {len(df)} simulated records to {args.out} "
        f"(event rate {df['unfavourable_outcome_3m'].mean():.1%})"
    )


if __name__ == "__main__":
    main()
