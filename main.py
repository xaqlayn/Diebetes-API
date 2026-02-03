from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional
import pickle
import logging
from starlette.responses import HTMLResponse, JSONResponse
from starlette.middleware.cors import CORSMiddleware
import os

MODEL_PATH = os.environ.get("MODEL_PATH", "diabetes_model.sav")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1.0.0")

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("diabetes-api")

app = FastAPI(
    title="Diabetes Prediction API",
    version=MODEL_VERSION,
    description="Predict diabetes likelihood using medical features. New API is under /api/v1/*"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------
# Pydantic models
# ---------------------
class PredictInput(BaseModel):
    pregnancies: int = Field(..., ge=0)
    glucose: int = Field(..., ge=0, le=300)
    blood_pressure: int = Field(..., ge=0, le=200)
    skin_thickness: int = Field(..., ge=0, le=99)
    insulin: int = Field(..., ge=0, le=900)
    bmi: float = Field(..., ge=5, le=80)
    diabetes_pedigree_function: float = Field(..., ge=0, le=5)
    age: int = Field(..., ge=1, le=120)

class PredictResponse(BaseModel):
    model_version: str
    prediction: str
    probability: Optional[float] = None
    threshold: Optional[float] = None
    raw_prediction: Optional[int] = None

# Backwards-compatible model (PascalCase) used by original endpoint
class LegacyInput(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# ---------------------
# Load model
# ---------------------
model = None
model_loaded = False

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
        model_loaded = True
        logger.info("Loaded ML model from %s", MODEL_PATH)
except Exception as e:
    logger.error("Failed to load model from %s: %s", MODEL_PATH, e)
    # Keep app up for health checks; endpoints using model will return errors

# ---------------------
# Health & metadata
# ---------------------
@app.get("/api/v1/health")
def health():
    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_version": MODEL_VERSION,
    }

# ---------------------
# Modern prediction endpoint (recommended)
# ---------------------
@app.post("/api/v1/predict", response_model=PredictResponse, status_code=200)
async def predict_v1(input: PredictInput):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = [
        input.pregnancies,
        input.glucose,
        input.blood_pressure,
        input.skin_thickness,
        input.insulin,
        input.bmi,
        input.diabetes_pedigree_function,
        input.age,
    ]
    try:
        raw_pred = int(model.predict([features])[0])
        label = "diabetic" if raw_pred == 1 else "not_diabetic"
        response = {"model_version": MODEL_VERSION, "prediction": label, "raw_prediction": raw_pred}

        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba([features])[0][1])
            # you can store a threshold in config/environment if desired
            response["probability"] = round(prob, 4)
            response["threshold"] = 0.5
        return response
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail="Prediction failed")

# ---------------------
# Legacy endpoint (kept for compatibility)
# ---------------------
@app.post("/diabetes-predict")
async def legacy_predict(data: LegacyInput):
    # Accepts PascalCase keys for backwards compatibility.
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    features = [
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age,
    ]
    try:
        raw_pred = int(model.predict([features])[0])
        label = "Diabetic" if raw_pred == 1 else "Not Diabetic"
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba([features])[0][1])
            return {"prediction": label, "probability": round(prob, 2)}
        return {"prediction": label}
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction failed.")

# ---------------------
# Web demo (serves a modern UI)
# ---------------------
@app.get("/", response_class=HTMLResponse)
@app.get("/web", response_class=HTMLResponse)
def web():
    # This serves the modern UI file contents; you can host a static file instead.
    with open("static/ui.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)