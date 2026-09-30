# Body composition and nutritional status in stroke outcome prediction

Code accompanying **Machine Learning-Based Prediction of Functional Outcomes in Acute Ischemic Stroke Using Body Composition and Nutritional Status**, by Eun-Tae Jeon, Sang-hun Lee, and Jin-Man Jung.

The manuscript is under review at *Clinical Nutrition* (YCLNU-D-26-00948).

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

For a demonstration with synthetic data:

```bash
python scripts/make_synthetic_data.py --out synthetic_data.csv
python scripts/run_experiment.py --data synthetic_data.csv
```

Synthetic data are for checking execution and do not reproduce the manuscript results.

## Data availability

Individual-level patient data cannot be shared because of privacy restrictions and the conditions of institutional review board approval, but are available from the corresponding author upon reasonable request.
