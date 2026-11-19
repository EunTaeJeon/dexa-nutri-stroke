# Body composition and nutritional status in stroke outcome prediction

Code accompanying **Machine Learning-Based Prediction of Functional Outcomes in Acute Ischemic Stroke Using Body Composition and Nutritional Status**, by Eun-Tae Jeon, Sang-hun Lee, Seo-In Han, Ye-Sol Sim, and Jin-Man Jung.

The manuscript is under review at *Clinical Nutrition* (YCLNU-D-26-00948).

This repository provides a reference training workflow and the R2 evaluation of
saved predictions. The original research workflow also used study-specific
cohort linkage, saved development/validation splits, model artifacts, and
supplementary analyses. Running the reference training workflow does not
recreate those artifacts or all of the manuscript tables and figures.

## Installation and use

Use Python 3.10–3.12. From the repository directory:

```bash
python -m venv .venv
```

Activate the environment, then install the package:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/run_experiment.py --data path/to/data.csv
```

## Dependencies

The package dependencies are declared in `pyproject.toml`. The full dependency list, including Matplotlib, is in `requirements.txt`.

| Package | Version |
| --- | --- |
| NumPy | 1.26.4 |
| pandas | 2.2.3 |
| scikit-learn | 1.3.2 |
| SciPy | 1.13.1 |
| statsmodels | 0.14.4 |
| LightGBM | 4.5.0 |
| SHAP | 0.46.0 |
| Numba | 0.60.0 |
| Optuna | >=3.4 |
| PyYAML | >=6.0 |
| Matplotlib | >=3.7 |

The build backend requires setuptools >=68. To run the existing tests, install the development dependencies with `python -m pip install -e ".[dev]"`, then run `python -m pytest`.

## Input data

The input columns are defined in `src/dexa_nutri_stroke/schema.py`. Settings are in `config/default.yaml`. To select the 1-year outcome, pass `--outcome unfavourable_outcome_1y`.

Use one row per patient with predictors available at the intended time of
prediction. Code outcomes as 1 for modified Rankin Scale ≥2 and 0 for <2.
Leave unobserved outcomes missing. The 1-year model excludes follow-up PNI.
For subgroup analyses, use the original study's TOAST coding defined in
`TOAST_SUBTYPES` in the schema. Previous synthetic files used codes 1–5. For
those files, regenerate the synthetic cohort or add 1 to the TOAST codes.
Supply `dexa_delay_hours` for the scan-timing subgroups. Leave missing
subgroup values missing.

For a demonstration with synthetic data:

```bash
python scripts/make_synthetic_data.py --out synthetic_data.csv
python scripts/run_experiment.py --data synthetic_data.csv
```

Synthetic data are for checking execution and do not reproduce the manuscript results.

## Threshold-based evaluation of saved predictions

Prepare a CSV with one row per validation patient and the columns `outcome`,
`clin_probability`, and `clin_dexa_nutri_probability`. Verify patient identity
and outcome when joining saved predictions. Use the corresponding
development-derived cutoffs with their full precision. For example:

```bash
python scripts/evaluate_predictions.py --predictions path/to/paired_predictions.csv --clin-threshold 0.5625 --full-threshold 0.5416666666666666 --seed 42
```

For the 1-year manuscript analysis use cutoffs `0.4166666666666667` and
`0.4642857142857143`, and seed `43`. Both models classify probabilities at or
above their cutoff as positive. The output includes the observed estimates
and paired percentile bootstrap intervals (10,000 resamples) for the
threshold-based metrics and two-category NRI. Two-sided exact McNemar tests
compare sensitivity among patients with the outcome and specificity among
patients without it. These P values are unadjusted. Undefined estimates are
reported as JSON `null`. When a resampled ratio is undefined, the interval is
conditional on the resamples for which it is defined. The output reports
the number of such resamples.

The reference training command uses calibrated, out-of-fold development
probabilities to choose each model's Youden cutoff. It applies the frozen
cutoffs to the held-out calibrated probabilities and writes the same paired
evaluation under `threshold_comparison` in `comparison.json`. The
cross-validated AUROC describes development after feature and hyperparameter
selection. Held-out AUROC is reported separately.

## Data availability

Individual-level patient data cannot be shared because of privacy restrictions and the conditions of institutional review board approval, but are available from the corresponding author upon reasonable request.
