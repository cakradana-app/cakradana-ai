# 🏛️ Cakradana AI

<div align="center">

![Cakradana Logo](assets/logo.png)

**AI System for Transparency in Indonesian Election Financing**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-green.svg)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3.0-orange.svg)](https://lightgbm.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.1-red.svg)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-2.2.2-purple.svg)](https://pandas.pydata.org/)

</div>

---

> **Cakradana** is an open-source analytics & AI toolkit to **detect risky political donations** in Indonesia. It combines feature engineering (temporal, network, and behavioral), rule-based checks (legal limits), and a machine learning model (LightGBM) with **threshold tuning** that prioritizes **recall of the *risky* class** under the constraint **recall of *not_risky* ≥ 70%**.

* 🎯 **Goal**: help **KPU, PPATK, parties, and candidates** identify patterns like smurfing, proxy accounts, and fake self-funding.
* 🧠 **Model**: LightGBM + rich feature engineering.
* ⚖️ **Compliance**: checks contribution limits per Election/Party Law and detects prohibited funding sources.
* 🚀 **Serving**: FastAPI for real‑time risk scoring.

---

## 🌐 `cakradana-ai` Repository

GitHub repo: [https://github.com/cakradana-app/cakradana-ai](https://github.com/cakradana-app/cakradana-ai)

This repository hosts the **AI/ML source code** for Cakradana, including a synthetic data generator, feature engineering, model training, evaluation, and an API service. Actual repo structure:

```
├── app_fastapi.py
├── app_flask.py
├── artifacts
│   ├── lgbm_pipeline.joblib
│   ├── model_meta.json
│   └── threshold.json
├── Dockerfile
├── featured_synthetic_donations.csv
├── generate_synthetic_data.py
├── README.md
├── requirements.txt
├── synthetic_donations.csv
├── train_and_export.py
└── train.ipynb
```

> **Note**: the **primary** service uses `app_fastapi.py`. The `app_flask.py` file is included as an alternative/prototype. The model and threshold are stored in `artifacts/`.

---

## 🧩 Architecture & Workflow

1. **Data Ingestion**
   Donation CSV with minimum columns: `sender, sender_type, receiver, receiver_type, date, amount, risk, risk_type` (the last two are used as labels during training).

2. **Feature Engineering**

   * **Temporal**: totals & frequency per period, last 30‑day transactions, velocity (average inter-donation interval), time components (month/day).
   * **Network (Graph)**: number of unique receivers per sender (out-degree), number of unique senders per receiver (in-degree), simple degree centrality, fan‑in/fan‑out patterns.
   * **Donor Behavior**: total/mean/std of donations per sender, receiver diversity, **largest-donation proportion per sender**, flag **exceeding legal limits** based on `sender_type`×`receiver_type`.

3. **Model Training**

   * **Estimator**: LightGBM (GBDT) with **class weighting** (`scale_pos_weight`) to handle imbalance.
   * **Validation**: stratified split; primary metrics ROC‑AUC, PR‑AUC, F1; feature importance analysis.

4. **Threshold Tuning (Option‑A)**
   From the ROC curve on the validation set: **choose a threshold** that **maximizes recall of the *risky* class** **subject to** *recall(*not_risky*) ≥ 0.70* (≡ **FPR ≤ 0.30**). This threshold is saved to `artifacts/threshold.json` and used in the service.

5. **Serving (FastAPI + Docker)**
   `app_fastapi.py` loads the **trained pipeline** + **threshold** and provides a `/predict` endpoint for risk scores and labels. The **Dockerfile** is configured to **automatically run `app_fastapi.py`** (via Uvicorn) when the container starts.

---

## 🛠️ Tech Stack

* **Language**: Python 3.10+ / 3.11+
* **Data**: pandas, NumPy
* **ML**: LightGBM, scikit‑learn (pipeline, metrics; class weighting & thresholding)
* **Serving**: FastAPI (primary), Uvicorn, Pydantic; Flask (optional)
* **Packaging/Runtime**: **Docker** (production container; auto-runs `app_fastapi.py`)
* **Artifacts**: joblib (`artifacts/lgbm_pipeline.joblib`, `threshold.json`, `model_meta.json`)

---

## 📈 Model Statistics (Validation)

> Configuration: LightGBM with **class weights** and **Option‑A** thresholding. Target `risk` (binary: 1=risky, 0=not_risky), stratified split.

**Key Summary**

* **ROC‑AUC**: **0.8424**
* **Accuracy**: **0.7377**
* **F1 (binary)**: **0.7472**

**Classification Report**

| Class         | Precision | Recall |   F1‑score |  Support |
| ------------- | --------- | ------ | ---------: | -------: |
| not_risky(0)  | 0.7569    | 0.7000 |     0.7273 |     1650 |
| risky(1)      | 0.7211    | 0.7753 | **0.7472** |     1651 |
| **Accuracy**  |           |        | **0.7377** | **3301** |
| **Macro avg** | 0.7390    | 0.7376 |     0.7373 |     3301 |
| **Weighted**  | 0.7390    | 0.7377 |     0.7373 |     3301 |

**Confusion Matrix**

```
[[TN, FP],
 [FN, TP]] = [[1155, 495],
              [ 371,1280]]
```

**Brief interpretation**

* The model shows **good** discrimination (ROC‑AUC ≈ 0.84).
* With **Option‑A**, recall of the *risky* class is **boosted** (≈ 0.78) while maintaining *not_risky* recall **≥ 0.70**.
* *Trade‑off*: higher *risky* recall increases FP on *not_risky*—aligned with a risk‑averse compliance strategy.

> **Operational Suggestion**: if the goal is triage, use **score ranking** + *top‑K review* (e.g., top 10–20%) to balance investigation workload and case capture.

---

## 🚀 Quickstart

### 1) Setup

```bash
git clone https://github.com/cakradana-app/cakradana-ai.git
cd cakradana-ai
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\\Scripts\\activate)
pip install -r requirements.txt
```

### 2) Generate synthetic data (if needed)

```bash
python generate_synthetic_data.py
```

### 3) Train the model & export artifacts

```bash
python train_and_export.py
```

### 4) Run the API (FastAPI)

```bash
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000
# open http://localhost:8000/docs
```

### 4b) Run via Docker (recommended for production)

```bash
# build image
docker build -t cakradana-ai .

# run container (Dockerfile auto-runs app_fastapi.py via Uvicorn)
docker run -p 8000:8000 --name cakradana cakradana-ai
# open http://localhost:8000/docs
```

### 5) Prediction example

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
        "sender_type": "individual",
        "receiver_type": "political-party",
        "amount": 2000000,
        "total_donasi_sender": 5000000,
        "jumlah_transaksi_sender": 3,
        "rata_rata_donasi_sender": 1666666.7,
        "std_donasi_sender": 1000000,
        "jumlah_donasi_30hari_sender": 2,
        "selang_waktu_rata2_sender": 7.5,
        "receiver_unik_per_sender": 2,
        "sender_unik_per_receiver": 15,
        "degree_centrality_sender": 2,
        "degree_centrality_receiver": 15,
        "proporsi_donasi_terbesar_per_sender": 0.6,
        "flag_donasi_melebihi_batas": 0
      }
    ]
  }'
```

---

## 🧪 Data & Schema

Minimum columns for inference:

* **Raw features**: `sender_type`, `receiver_type`, `amount`, `date` (optional for real‑time temporal features).
* **Derived features (examples)**:
  `total_donasi_sender`, `jumlah_transaksi_sender`, `rata_rata_donasi_sender`, `std_donasi_sender`, `jumlah_donasi_30hari_sender`, `selang_waktu_rata2_sender`, `receiver_unik_per_sender`, `sender_unik_per_receiver`, `degree_centrality_sender`, `degree_centrality_receiver`, `proporsi_donasi_terbesar_per_sender`, `flag_donasi_melebihi_batas`.

**Sample datasets in repo**:

* `synthetic_donations.csv` → basic synthetic transaction data (mockup).
* `featured_synthetic_donations.csv` → feature‑engineered training‑ready output.

> **Production**: ensure feature engineering is consistent between training & inference (use the same pipeline or a dedicated feature service).

---

## ⚙️ Important Configuration

* **Class weight**: `scale_pos_weight` to handle imbalance.
* **Thresholding (Option‑A)**: save the selected threshold to `artifacts/threshold.json` and **do not hard‑code** it in the service.
* **Calibration** (optional): `CalibratedClassifierCV` (isotonic) if probability scores need better reliability.
* **Validation**: consider *time‑based split* (train on earlier period → test on later) and *group split by sender* to reduce leakage.

---

## 🧭 Short Roadmap

* 🔍 **Advanced graph features** (betweenness, community) and/or GNN for collusive patterns (not used yet).
* 📊 **Monitoring & drift** (data drift, performance drift) + alerting.
* 🧾 **Explainability** (e.g., more advanced feature importance) for audit (no SHAP in repo currently).
* 🧱 **Hard rules** (KYC/AML) configurable by regulators.

---

## 🔐 Ethics & Compliance

Cakradana is intended as an **analytical decision‑support tool**. Model results are **not** evidence of violations; always conduct **human review** and refer to applicable regulations. Maintain privacy, data security, and audit trails according to standards.