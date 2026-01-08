from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Literal
import pandas as pd

from frax_predictor import predict_fraxplus

app = FastAPI()

# If Vite runs on http://localhost:5173
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RiskFactors(BaseModel):
    previousFracture: bool = False
    parentFracturedHip: bool = False
    smoking: bool = False
    glucocorticoids: bool = False
    rheumatoidArthritis: bool = False
    secondaryOsteoporosis: bool = False
    alcohol: bool = False


class FemoralNeck(BaseModel):
    type: Literal["bmd_g_cm2", "t_score"]
    value: float


class FraxRequest(BaseModel):
    continent: Optional[str] = None
    country: Optional[str] = None
    us_group: Optional[str] = None
    age: int
    sex: Literal["female", "male"]
    weight_kg: float
    height_cm: float
    risk_factors: RiskFactors
    t_score: Optional[FemoralNeck] = None


def bmi_from(weight_kg: float, height_cm: float) -> float:
    h = height_cm / 100.0
    cal = weight_kg / (h * h) if h > 0 else 0.0
    return round(cal, 2)


def map_country_to_us_group(country: Optional[str]) -> Optional[str]:
    return "US (Caucasian)"


@app.post("/api/frax/calculate")
def calculate(req: FraxRequest):
    row = {
        # core numerics
        "age": req.age,
        # adjust if your training used opposite encoding
        "sex_female": 1 if req.sex == "female" else 0,
        # adjust column names to match training!
        "weight_kg": req.weight_kg,
        "height_cm": req.height_cm,
        "bmi": bmi_from(req.weight_kg, req.height_cm),

        # toggles as 0/1
        "previousFracture": int(req.risk_factors.previousFracture),
        "parentFracturedHip": int(req.risk_factors.parentFracturedHip),
        "smoking": int(req.risk_factors.smoking),
        "glucocorticoids": int(req.risk_factors.glucocorticoids),
        "rheumatoidArthritis": int(req.risk_factors.rheumatoidArthritis),
        "secondaryOsteoporosis": int(req.risk_factors.secondaryOsteoporosis),
        "alcohol": int(req.risk_factors.alcohol),
    }

    # us_group used by your _prep_features() -> get_dummies
    row["us_group"] = (
        req.us_group or map_country_to_us_group(req.country) or "")
    row["bmd"] = 0.527
    # # femoral neck field — set the correct column name your model expects
    # if req.t_score:
    #     if req.t_score.type == "bmd_g_cm2":
    #         row["femoral_neck_bmd"] = req.femoral_neck.value  # rename if needed
    #     else:
    #         row["t_score"] = req.femoral_neck.value           # rename if needed

    df = pd.DataFrame([row])
    print(row)
    out = predict_fraxplus(df)

    return {
        "mof": float(out.loc[0, "mof_risk_pred"]),
        "hip": float(out.loc[0, "hip_risk_pred"]),
    }
