import numpy as np
import pandas as pd
import joblib


def predict_fraxplus(input_df, model_path="fraxplus_models.pkl", clamp_0_100=True):
    bundle = joblib.load(model_path)

    mof_model = bundle["mof_model"]
    hip_clf = bundle["hip_clf"]
    hip_reg_low = bundle["hip_reg_low"]
    hip_reg_high = bundle["hip_reg_high"]
    feature_columns = bundle["feature_columns"]
    hip_high_threshold = bundle["hip_high_threshold"]

    # Align columns
    X = input_df.copy()
    if "us_group" in X.columns:
        X = pd.get_dummies(X, columns=["us_group"], drop_first=False)

    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0.0

    X = X[feature_columns]

    # MOF prediction
    mof_pred = np.expm1(mof_model.predict(X))

    # HIP prediction
    p_high = hip_clf.predict_proba(X)[:, 1]
    hip_low = hip_reg_low.predict(X)
    hip_high = hip_reg_high.predict(X)

    hip_pred = np.expm1((1 - p_high) * hip_low + p_high * hip_high)

    if clamp_0_100:
        mof_pred = np.clip(mof_pred, 0, 100)
        hip_pred = np.clip(hip_pred, 0, 100)

    return pd.DataFrame({
        "mof_risk_pred": mof_pred,
        "hip_risk_pred": hip_pred
    })
