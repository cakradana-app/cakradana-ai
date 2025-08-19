# app.py
# Flask API that loads the pipeline + Option-A threshold
# Endpoints:
#   GET  /health
#   GET  /threshold
#   POST /predict   (JSON: {"instances": [ {...row...}, {...} ]})

import json
import joblib
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify

ART_DIR = Path("artifacts")
PIPELINE_PATH = ART_DIR / "lgbm_pipeline.joblib"
THRESHOLD_JSON = ART_DIR / "threshold.json"
META_JSON = ART_DIR / "model_meta.json"

app = Flask(__name__)

# Load artifacts at startup
pipe = joblib.load(PIPELINE_PATH)
with open(THRESHOLD_JSON, "r") as f:
    THR_INFO = json.load(f)
THRESHOLD = float(THR_INFO["threshold"])
MIN_RECALL_NOT_RISKY = float(THR_INFO["constraint_recall_not_risky"])

META = {}
if META_JSON.exists():
    with open(META_JSON, "r") as f:
        META = json.load(f)

# ---------- Helpers ----------
def ensure_dataframe(instances):
    """
    instances: list[dict] or dict -> returns DataFrame
    """
    if isinstance(instances, dict):
        instances = [instances]
    if not isinstance(instances, list) or not all(isinstance(x, dict) for x in instances):
        raise ValueError("Payload must contain 'instances': list of JSON objects.")
    df = pd.DataFrame(instances)
    # Optional: drop raw id columns if clients send them
    for col_drop in ["sender", "receiver", "date", "risk_type", "risk"]:
        if col_drop in df.columns:
            df = df.drop(columns=[col_drop])
    return df

# ---------- Routes ----------
@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": PIPELINE_PATH.exists(),
        "threshold": THRESHOLD,
        "constraint_recall_not_risky": MIN_RECALL_NOT_RISKY,
        "meta": META
    })

@app.get("/threshold")
def get_threshold():
    return jsonify({
        "threshold": THRESHOLD,
        "constraint_recall_not_risky": MIN_RECALL_NOT_RISKY
    })

@app.post("/predict")
def predict():
    """
    Request JSON:
    {
      "instances": [
        {
          "sender_type": "individual",
          "receiver_type": "political-party",
          "amount": 2000000,
          ...
          # include ALL engineered feature columns used during training,
          # except raw IDs and the target columns.
        },
        ...
      ]
    }
    """
    try:
        payload = request.get_json(silent=True) or {}
        instances = payload.get("instances")
        if instances is None:
            return jsonify({"error": "Missing 'instances' in JSON."}), 400

        df = ensure_dataframe(instances)

        # Predict probabilities
        proba = pipe.predict_proba(df)[:, 1]
        label = (proba >= THRESHOLD).astype(int)

        # Build response
        out = []
        for i in range(len(df)):
            out.append({
                "risk_score": float(proba[i]),
                "risk_label": int(label[i]),
                "threshold_used": THRESHOLD
            })

        return jsonify({"predictions": out})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # FLASK_RUN_PORT or default 8000
    import os
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)