# app_fastapi.py
# FastAPI service loading a LightGBM pipeline + Option-A threshold.
# Endpoints:
#   GET  /health
#   GET  /threshold
#   POST /predict   (JSON: {"instances": [ {...row...}, {...} ]})

import json
import joblib
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

ART_DIR = Path("artifacts")
PIPELINE_PATH = ART_DIR / "lgbm_pipeline.joblib"
THRESHOLD_JSON = ART_DIR / "threshold.json"
META_JSON = ART_DIR / "model_meta.json"

# ---------- FastAPI app ----------
app = FastAPI(title="Cakradana Risk Scoring API", version="1.0")

# (optional) CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Load artifacts ----------
if not PIPELINE_PATH.exists() or not THRESHOLD_JSON.exists():
    raise RuntimeError("Artifacts missing. Run train_and_export.py first.")

pipe = joblib.load(PIPELINE_PATH)

with open(THRESHOLD_JSON, "r") as f:
    THR_INFO = json.load(f)
THRESHOLD = float(THR_INFO["threshold"])
MIN_RECALL_NOT_RISKY = float(THR_INFO.get("constraint_recall_not_risky", 0.70))

META: Dict[str, Any] = {}
if META_JSON.exists():
    with open(META_JSON, "r") as f:
        META = json.load(f)

CAT_COLS: List[str] = META.get("cat_cols", [])
NUM_COLS: List[str] = META.get("num_cols", [])
EXPECTED_COLS: List[str] = CAT_COLS + NUM_COLS

# ---------- Schemas ----------
class PredictRequest(BaseModel):
    # Free-form rows; we'll align schema server-side
    instances: List[Dict[str, Any]]

class PredictItem(BaseModel):
    risk_score: float
    risk_label: int
    threshold_used: float

class PredictResponse(BaseModel):
    predictions: List[PredictItem]


# ---------- Helpers ----------
DROP_RAW_COLS = {"sender", "receiver", "date", "risk_type", "risk"}

def ensure_dataframe(instances: List[Dict[str, Any]]) -> pd.DataFrame:
    if not isinstance(instances, list) or not all(isinstance(x, dict) for x in instances):
        raise ValueError("`instances` must be a list of JSON objects.")
    df = pd.DataFrame(instances)
    # Drop raw ID/label columns if clients send them
    for c in list(DROP_RAW_COLS.intersection(df.columns)):
        df = df.drop(columns=[c])
    return df

def align_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Align incoming payload to the training schema using model_meta:
      - Ensure all expected categorical & numeric columns exist
      - Fill missing with safe defaults
      - Cast dtypes
      - Keep only expected columns (order matters for transformers)
    """
    # Add missing categorical with "unknown"
    for c in CAT_COLS:
        if c not in df.columns:
            df[c] = "unknown"
        df[c] = df[c].astype("object").fillna("unknown")

    # Add missing numeric with 0.0
    for c in NUM_COLS:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Remove unexpected columns, order columns
    if EXPECTED_COLS:
        df = df.reindex(columns=EXPECTED_COLS)
    return df


# ---------- Routes ----------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "pipeline_loaded": PIPELINE_PATH.exists(),
        "threshold": THRESHOLD,
        "constraint_recall_not_risky": MIN_RECALL_NOT_RISKY,
        "meta": {
            "cat_cols": CAT_COLS,
            "num_cols": NUM_COLS,
            "pos_weight_used": META.get("pos_weight_used", None),
            "ap_val": META.get("ap_val", None),
        },
    }

@app.get("/threshold")
def get_threshold():
    return {
        "threshold": THRESHOLD,
        "constraint_recall_not_risky": MIN_RECALL_NOT_RISKY,
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        df = ensure_dataframe(req.instances)
        df = align_schema(df)

        # Predict probabilities and apply Option-A threshold picked at training time
        proba = pipe.predict_proba(df)[:, 1]
        label = (proba >= THRESHOLD).astype(int)

        preds = [
            PredictItem(risk_score=float(p), risk_label=int(l), threshold_used=THRESHOLD)
            for p, l in zip(proba, label)
        ]
        return PredictResponse(predictions=preds)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))