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
python -m pip install -e .
python scripts/run_experiment.py --data path/to/data.csv
```

The input columns are defined in `src/dexa_nutri_stroke/schema.py`. Settings are in `config/default.yaml`. To select the 1-year outcome, pass `--outcome unfavourable_outcome_1y`.

For a demonstration with synthetic data:

```bash
python scripts/make_synthetic_data.py --out synthetic_data.csv
python scripts/run_experiment.py --data synthetic_data.csv
```

Synthetic data are for checking execution and do not reproduce the manuscript results. Individual-level study data and fitted models are not included. Data access requests should be addressed to the corresponding author, Jin-Man Jung, at dr.jinmanjung@gmail.com.

Code contact: Eun-Tae Jeon, gksmfskdls@gmail.com.
