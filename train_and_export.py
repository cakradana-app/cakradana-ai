# train_and_export.py
# Train LightGBM, pick Option-A threshold, export pipeline + threshold.

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, average_precision_score
import lightgbm as lgb

# ======= CONFIG =======
DATA_CSV = "featured_synthetic_donations.csv"   # <-- path to your FEATURE-ENGINEERED dataset
TARGET_COL = "risk"                   # boolean or {0,1}
MIN_RECALL_NOT_RISKY = 0.70           # constraint: recall(not_risky) >= 0.70
MODEL_DIR = Path("artifacts")         # where to save pipeline + threshold
MODEL_DIR.mkdir(exist_ok=True)
PIPELINE_PATH = MODEL_DIR / "lgbm_pipeline.joblib"
THRESHOLD_JSON = MODEL_DIR / "threshold.json"
META_JSON = MODEL_DIR / "model_meta.json"

print(f"Loading dataset: {DATA_CSV}")
df = pd.read_csv(DATA_CSV)

# Ensure target is 0/1 integer
y = df[TARGET_COL].astype(int)
X = df.drop(columns=[TARGET_COL])

# OPTIONAL: drop high-cardinality IDs if present
for col_drop in ["sender", "receiver", "date", "risk_type"]:
    if col_drop in X.columns:
        X = X.drop(columns=[col_drop])

# Identify dtypes
cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]
num_cols = [c for c in X.columns if c not in cat_cols]

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y
)

# Class imbalance handling
pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

# Preprocessor
pre = ColumnTransformer(
    transformers=[
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ("num", "passthrough", num_cols),
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

# Model
lgbm = lgb.LGBMClassifier(
    random_state=42,
    n_estimators=800,
    learning_rate=0.05,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=2.0,
    scale_pos_weight=pos_weight
)

pipe = Pipeline(steps=[("pre", pre), ("clf", lgbm)])

print("Training...")
pipe.fit(X_train, y_train)

# Predict proba on validation
y_val_proba = pipe.predict_proba(X_val)[:, 1]

# ---------- OPTION A: pick threshold ----------
# Constraint recall(not_risky) >= MIN_RECALL_NOT_RISKY  <=>  FPR <= 1 - MIN_RECALL_NOT_RISKY
fpr, tpr, thr = roc_curve(y_val, y_val_proba)
recall_not_risky = 1 - fpr
mask = recall_not_risky >= MIN_RECALL_NOT_RISKY

if np.any(mask):
    idx = np.argmax(tpr[mask])             # maximize TPR among feasible thresholds
    thr_use = float(thr[mask][idx])
else:
    # no exact feasible point -> pick closest specificity
    idx = np.argmin(np.abs(recall_not_risky - MIN_RECALL_NOT_RISKY))
    thr_use = float(thr[idx])
    print("[warn] No threshold reaches the constraint exactly; using closest point.")

print(f"Selected threshold (Option A): {thr_use:.4f}")

# Save artifacts
joblib.dump(pipe, PIPELINE_PATH)
with open(THRESHOLD_JSON, "w") as f:
    json.dump({"threshold": thr_use, "constraint_recall_not_risky": MIN_RECALL_NOT_RISKY}, f, indent=2)

# Some metadata
ap = average_precision_score(y_val, y_val_proba)
with open(META_JSON, "w") as f:
    json.dump({
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "pos_weight_used": float(pos_weight),
        "ap_val": float(ap),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val))
    }, f, indent=2)

print(f"Artifacts saved to: {MODEL_DIR.resolve()}")