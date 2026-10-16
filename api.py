"""
GlucoGuard: Multi-Factor Diabetes Risk & Early Prevention Intelligence Engine
Production REST API Microservice (api.py)
IBM SkillsBuild / Bharat Cares AICTE Machine Learning Final Submission
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Resolve local paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import (
    prepare_single_patient_vector,
    prepare_batch_dataframe,
    RAW_FEATURE_COLUMNS
)
from src.model_trainer import (
    load_trained_engine,
    METADATA_PATH,
    GlucoGuardInferenceEngine
)

# ---------------------------------------------------------
# FastAPI App Initialization & CORS Setup
# ---------------------------------------------------------
app = FastAPI(
    title="GlucoGuard Clinical AI Diagnostic API",
    description=(
        "Production-grade clinical inference microservice for multi-factor diabetes risk stratification. "
        "Provides single-patient risk analysis, batch screening, model validation telemetry, and health monitoring."
    ),
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for cross-origin frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache inference engine and metadata in process
engine: Optional[GlucoGuardInferenceEngine] = None
metadata: Optional[Dict[str, Any]] = None


@app.on_event("startup")
def startup_event():
    """Initializes models and metadata on server startup."""
    global engine, metadata
    try:
        engine, metadata = load_trained_engine()
        print(f"[✓] GlucoGuard API initialized with champion model: {metadata.get('best_model_architecture')}")
    except Exception as exc:
        print(f"[X] Startup error initializing models: {exc}")
        raise exc


# ---------------------------------------------------------
# Pydantic Schemas for Validation
# ---------------------------------------------------------
class PatientRecord(BaseModel):
    Pregnancies: int = Field(default=0, ge=0, le=25, description="Number of times pregnant")
    Glucose: float = Field(..., ge=20.0, le=400.0, description="2-Hour oral glucose tolerance test (mg/dL)")
    BloodPressure: float = Field(..., ge=20.0, le=240.0, description="Diastolic blood pressure (mmHg)")
    SkinThickness: float = Field(default=20.0, ge=0.0, le=120.0, description="Triceps skinfold thickness (mm)")
    Insulin: float = Field(default=80.0, ge=0.0, le=900.0, description="2-Hour serum insulin (μIU/mL)")
    BMI: float = Field(..., ge=10.0, le=80.0, description="Body Mass Index (weight in kg / height in m²)")
    DiabetesPedigreeFunction: float = Field(default=0.35, ge=0.01, le=3.5, description="Genetic pedigree risk score")
    Age: int = Field(..., ge=1, le=120, description="Chronological biological age in years")

    class Config:
        schema_extra = {
            "example": {
                "Pregnancies": 2,
                "Glucose": 138.0,
                "BloodPressure": 78.0,
                "SkinThickness": 28.0,
                "Insulin": 110.0,
                "BMI": 31.4,
                "DiabetesPedigreeFunction": 0.45,
                "Age": 42
            }
        }


class SinglePredictionResponse(BaseModel):
    risk_probability: float = Field(..., description="Estimated probability of diabetes onset (0.0 to 1.0)")
    risk_score_pct: float = Field(..., description="Risk score expressed as a percentage")
    risk_tier: str = Field(..., description="Categorized clinical risk level ('Low Risk', 'Borderline Pre-Diabetic', 'High Risk')")
    high_risk_flag: bool = Field(..., description="True if patient crosses high risk diagnostic threshold (>= 65%)")
    model_used: str = Field(..., description="Name of the champion model architecture deployed")
    timestamp: str = Field(..., description="UTC ISO timestamp of the prediction")


class BatchPredictionResponse(BaseModel):
    total_screened: int
    high_risk_flagged: int
    average_cohort_risk_pct: float
    predictions: List[SinglePredictionResponse]


# ---------------------------------------------------------
# Microservice Endpoints
# ---------------------------------------------------------
@app.get("/", tags=["System & Health"])
def health_check():
    """Health check route returning service operational status, model architecture, and version."""
    model_arch = metadata.get("best_model_architecture", "Ensemble Classifier") if metadata else "Initializing"
    return {
        "status": "healthy",
        "service": "GlucoGuard Clinical Decision Support API",
        "version": "1.1.0",
        "model_architecture": model_arch,
        "sdg_alignment": "UN SDG 3: Good Health and Well-Being",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z"
    }


@app.get("/metrics", tags=["Model Telemetry"])
def get_model_metrics():
    """Returns cross-validation and test holdout benchmark telemetry from models/model_metadata.json."""
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata is still initializing.")

    return {
        "best_model_architecture": metadata.get("best_model_architecture"),
        "cv_benchmarks": metadata.get("cv_benchmarks"),
        "test_benchmarks": metadata.get("test_benchmarks"),
        "features": metadata.get("features"),
        "trained_at_utc": metadata.get("trained_at_utc")
    }


@app.post("/predict", response_model=SinglePredictionResponse, tags=["Inference"])
def predict_single_patient(patient: PatientRecord):
    """
    Accepts clinical patient biomarkers, applies automated imputation and feature engineering,
    and returns calibrated diabetes risk probability, clinical tier, and high-risk flags.
    """
    if engine is None or metadata is None:
        raise HTTPException(status_code=503, detail="Inference engine is not initialized.")

    patient_dict = patient.dict()
    vec = prepare_single_patient_vector(patient_dict)
    prob = float(engine.predict_risk_score(vec)[0])
    prob_rounded = round(prob, 4)
    pct = round(prob * 100, 2)

    if pct < 30.0:
        tier = "Low Risk"
    elif pct < 65.0:
        tier = "Borderline Pre-Diabetic"
    else:
        tier = "High Risk"

    return SinglePredictionResponse(
        risk_probability=prob_rounded,
        risk_score_pct=pct,
        risk_tier=tier,
        high_risk_flag=(pct >= 65.0),
        model_used=metadata.get("best_model_architecture", "Voting Ensemble (Champion)"),
        timestamp=datetime.utcnow().isoformat() + "Z"
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch_patients(patients: List[PatientRecord]):
    """
    Accepts a list of patient biomarker records, executes vectorized batch inference,
    and returns patient-level risk predictions and aggregate cohort health indices.
    """
    if engine is None or metadata is None:
        raise HTTPException(status_code=503, detail="Inference engine is not initialized.")

    if not patients:
        raise HTTPException(status_code=400, detail="Patient list cannot be empty.")

    records = [p.dict() for p in patients]
    import pandas as pd
    df_raw = pd.DataFrame(records)

    df_features = prepare_batch_dataframe(df_raw)
    probs = engine.predict_risk_score(df_features)

    model_name = metadata.get("best_model_architecture", "Voting Ensemble (Champion)")
    ts = datetime.utcnow().isoformat() + "Z"

    results = []
    high_risk_count = 0

    for prob in probs:
        p_round = round(float(prob), 4)
        pct = round(float(prob) * 100, 2)
        if pct < 30.0:
            tier = "Low Risk"
        elif pct < 65.0:
            tier = "Borderline Pre-Diabetic"
        else:
            tier = "High Risk"

        is_high = (pct >= 65.0)
        if is_high:
            high_risk_count += 1

        results.append(SinglePredictionResponse(
            risk_probability=p_round,
            risk_score_pct=pct,
            risk_tier=tier,
            high_risk_flag=is_high,
            model_used=model_name,
            timestamp=ts
        ))

    avg_risk = round(float(probs.mean() * 100), 2)

    return BatchPredictionResponse(
        total_screened=len(results),
        high_risk_flagged=high_risk_count,
        average_cohort_risk_pct=avg_risk,
        predictions=results
    )


if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
