import numpy as np
import pandas as pd
import joblib


def _prep_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    X = X.drop(
        columns=[c for c in ["continent", "bmi_units", "scanner",
                             "mof_risk", "hip_risk"] if c in X.columns],
        errors="ignore"
    )
    if "us_group" in X.columns:
        X["us_group"] = X["us_group"].astype(str).str.strip()
        X = X.drop(columns=[c for c in X.columns if c.startswith(
            "us_group_")], errors="ignore")
        X = pd.get_dummies(X, columns=["us_group"], drop_first=False)
    X = X.apply(pd.to_numeric, errors="coerce")
    return X


def predict_fraxplus(input_df: pd.DataFrame, model_path: str = "fraxplus_models.pkl", clamp_0_100: bool = True):
    bundle = joblib.load(model_path)

    mof_model = bundle["mof_model"]
    hip_clf = bundle["hip_clf"]
    hip_reg_low = bundle["hip_reg_low"]
    hip_reg_high = bundle["hip_reg_high"]
    feature_columns = bundle["feature_columns"]

    a = bundle.get("hip_calib_a", 0.0)
    b = bundle.get("hip_calib_b", 1.0)
    clip_max = float(bundle.get("hip_calib_log_clip_max", 100.0))
    log_clip_max = np.log1p(clip_max)

    X = _prep_features(input_df)
    X = X.reindex(columns=feature_columns, fill_value=0.0)

    # MOF
    mof_pred = np.expm1(mof_model.predict(X))

    # HIP (log space mixture)
    p_high = hip_clf.predict_proba(X)[:, 1]
    low_t = hip_reg_low.predict(X)
    high_t = hip_reg_high.predict(X)
    hip_pred_t = (1.0 - p_high) * low_t + p_high * high_t  # log1p space

    # Calibrate in log space + clip to avoid explosions
    hip_pred_t_cal = a + b * hip_pred_t
    hip_pred_t_cal = np.clip(hip_pred_t_cal, 0.0, log_clip_max)
    hip_pred = np.expm1(hip_pred_t_cal)

    if clamp_0_100:
        mof_pred = np.clip(mof_pred, 0.0, 100.0)
        hip_pred = np.clip(hip_pred, 0.0, 100.0)

    return pd.DataFrame({"mof_risk_pred": mof_pred, "hip_risk_pred": hip_pred})


input = pd.read_csv("sampleInput.csv")
pred_out = predict_fraxplus(input)
print(pred_out)
