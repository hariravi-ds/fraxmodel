# FRAXplus Surrogate Models (MOF + Hip)

## What this is
This project trains a surrogate ML model that predicts:
- `mof_risk` (10-year major osteoporotic fracture risk, %)
- `hip_risk` (10-year hip fracture risk, %)

Outputs are intended to closely match FRAX/FRAXplus-style calculator outputs for the same input schema.

---

## Files
- `fraxplus_models.pkl`  
  Saved model bundle (MOF regressor + Hip classifier + Hip low/high regressors + calibration + feature schema)

- `predict_fraxplus.py`  
  Python helper that loads `fraxplus_models.pkl` and predicts risks for new inputs.

---

## Setup
```bash
pip install numpy pandas scikit-learn joblib
