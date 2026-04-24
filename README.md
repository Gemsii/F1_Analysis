# F1 Analysis

Short exploratory analysis and a baseline predictive workflow for Formula 1
race dynamics and champion prediction (seasons 2000–2024).

## Contents

- `SIAP_EDA_BaslineModel.ipynb` — annotated EDA and baseline models (notebook).
- `eda_graphs.py` — plotting helpers used by the notebook.
- `eda_data_processing.py`, `prediction_data_processing.py` — data-prep functions.
- `data/` — raw CSV inputs from the F1 dataset (circuits, races, results, drivers, etc.).

## Quickstart

1. Create (optional) and activate a venv, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
%pip install pandas scikit-learn matplotlib
```

2. Open and run the notebook `SIAP_EDA_BaslineModel.ipynb` in Jupyter or VS Code.

3. Or import and call plotting functions from `eda_graphs.py` in a Python session.

## Notes

- Comments and user-facing strings were translated to English for readability.
- The notebook contains captured outputs; clear them if you want a clean commit.

Enjoy exploring the data — ask if you want tests, packaging, or CI setup.
