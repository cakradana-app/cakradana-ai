# 🏛️ Cakradana AI

<div align="center">

![Cakradana Logo](assets/logo.png)

**Sistem AI untuk Transparansi Pembiayaan Pemilu Indonesia**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Node.js](https://img.shields.io/badge/Node.js-18+-green.svg)](https://nodejs.org/)
[![Express.js](https://img.shields.io/badge/Express.js-4.21+-blue.svg)](https://expressjs.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-8.9+-green.svg)](https://www.mongodb.com/)

</div>

---

> **Cakradana** adalah rangkaian alat analitik & AI open-source untuk **mendeteksi donasi politik berisiko** di Indonesia. Sistem ini memadukan feature engineering (temporal, jaringan, dan perilaku), rule-based checks (batas UU), serta model machine learning (LightGBM) dengan **penyetelan ambang (threshold)** yang mengutamakan **recall kelas *risky*** di bawah kendala **recall kelas *not\_risky* ≥ 70%**.

* 🎯 **Tujuan**: membantu **KPU, PPATK, partai, dan kandidat** mengidentifikasi pola *smurfing*, *proxy account*, dan *self-funded palsu*.
* 🧠 **Model**: LightGBM + rekayasa fitur yang kaya.
* ⚖️ **Kepatuhan**: pemeriksaan batas sumbangan sesuai UU Pemilu & Parpol, serta deteksi sumber dana terlarang.
* 🚀 **Serving**: FastAPI untuk penyajian skor risiko secara real‑time.

---

## 🌐 Repositori `cakradana-ai`

Repo GitHub: [https://github.com/cakradana-app/cakradana-ai](https://github.com/cakradana-app/cakradana-ai)

Repositori ini menampung **kode sumber AI/ML** untuk Cakradana, termasuk generator data sintetik, feature engineering, pelatihan model, evaluasi, dan layanan API. Struktur repo aktual:

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

> **Catatan**: layanan **utama** menggunakan `app_fastapi.py`. Berkas `app_flask.py` disertakan sebagai alternatif/prototipe. Model & ambang disimpan di `artifacts/`.

---

## 🧩 Arsitektur & Alur Kerja

1. **Ingest Data**
   CSV donasi dengan kolom minimal: `sender, sender_type, receiver, receiver_type, date, amount, risk, risk_type` (dua terakhir dipakai sebagai label saat training).

2. **Feature Engineering**

   * **Temporal**: total & frekuensi per periode, transaksi 30‑hari terakhir, *velocity* (rata‑rata selang waktu antar donasi), komponen waktu (bulan/hari).
   * **Jaringan (Graph)**: jumlah penerima unik per pengirim (out‑degree), jumlah pengirim unik per penerima (in‑degree), *degree centrality* sederhana, pola fan‑in/fan‑out.
   * **Perilaku Donatur**: total/mean/std donasi per pengirim, variasi penerima, **proporsi donasi terbesar per penerima**, flag **melebihi batas UU** berdasarkan `sender_type`×`receiver_type`.

3. **Pelatihan Model**

   * **Estimator**: LightGBM (GBDT) dengan **class weighting** (`scale_pos_weight`) untuk menyeimbangkan kelas.
   * **Validasi**: split terstratifikasi; metrik utama ROC‑AUC, PR‑AUC, F1; analisis *feature importance*.

4. **Penyetelan Ambang (Option‑A)**
   Dari kurva ROC pada set validasi: **pilih ambang** yang **memaksimalkan recall kelas *risky*** **dengan kendala** *recall(*not\_risky*) ≥ 0.70* (≡ **FPR ≤ 0.30**). Ambang ini disimpan ke `artifacts/threshold.json` dan dipakai di layanan.

5. **Serving (FastAPI + Docker)**
   `app_fastapi.py` memuat **pipeline terlatih** + **threshold** lalu menyediakan endpoint `/predict` untuk skor & label risiko. **Dockerfile** dikonfigurasi untuk **otomatis menjalankan `app_fastapi.py`** (via Uvicorn) saat kontainer start.

---

## 🛠️ Tech Stack

* **Bahasa**: Python 3.10+ / 3.11+
* **Data**: pandas, NumPy
* **ML**: LightGBM, scikit‑learn (pipeline, metrics; class weighting & thresholding)
* **Serving**: FastAPI (utama), Uvicorn, Pydantic; Flask (opsional)
* **Packaging/Runtime**: **Docker** (kontainer produksi; otomatis menjalankan `app_fastapi.py`)
* **Artifacts**: joblib (`artifacts/lgbm_pipeline.joblib`, `threshold.json`, `model_meta.json`)

---

## 📈 Statistik Model (Validasi)

> Konfigurasi: LightGBM dengan **class weight** dan **Option‑A** thresholding. Target `risk` (biner: 1=risky, 0=not\_risky), *split* terstratifikasi.

**Ringkasan Utama**

* **ROC‑AUC**: **0.8424**
* **Akurasi**: **0.7377**
* **F1 (binary)**: **0.7472**

**Classification Report**

| Kelas         | Precision | Recall |   F1‑score |  Support |
| ------------- | --------- | ------ | ---------: | -------: |
| not\_risky(0) | 0.7569    | 0.7000 |     0.7273 |     1650 |
| risky(1)      | 0.7211    | 0.7753 | **0.7472** |     1651 |
| **Akurasi**   |           |        | **0.7377** | **3301** |
| **Macro avg** | 0.7390    | 0.7376 |     0.7373 |     3301 |
| **Weighted**  | 0.7390    | 0.7377 |     0.7373 |     3301 |

**Confusion Matrix**

```
[[TN, FP],
 [FN, TP]] = [[1155, 495],
              [ 371,1280]]
```

**Interpretasi singkat**

* Model **baik** dalam diskriminasi (ROC‑AUC ≈ 0.84).
* Dengan **Option‑A**, recall kelas *risky* **ditingkatkan** (≈ 0.78) sambil menjaga recall *not\_risky* **≥ 0.70**.
* *Trade‑off*: sebagian kenaikan recall *risky* dibayar dengan FP pada *not\_risky*—sesuai strategi kepatuhan yang *risk‑averse*.

> **Saran Operasional**: jika tujuan adalah triase, gunakan **ranking skor** + *top‑K review* (mis. 10–20% tertinggi) untuk menyeimbangkan beban investigasi dan tangkapan kasus.

---

## 🚀 Quickstart

### 1) Persiapan

```bash
git clone https://github.com/cakradana-app/cakradana-ai.git
cd cakradana-ai
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\\Scripts\\activate)
pip install -r requirements.txt
```

### 2) Generate data sintetik (bila perlu)

````bash
python generate_synthetic_data.py
````

### 3) Latih Model & Simpan Artifacts

````bash
python train_and_export.py
````

### 4) Jalankan API (FastAPI)

```bash
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000
# buka http://localhost:8000/docs
```

### 4b) Jalankan via Docker (disarankan untuk produksi)

````bash
# build image
docker build -t cakradana-ai .

# run container (Dockerfile otomatis menjalankan app_fastapi.py via Uvicorn)
docker run -p 8000:8000 --name cakradana cakradana-ai
# buka http://localhost:8000/docs
```bash
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000
# buka http://localhost:8000/docs
````

### 5) Contoh Prediksi

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

## 🧪 Data & Skema

Minimal kolom untuk inference:

* **Fitur asli**: `sender_type`, `receiver_type`, `amount`, `date` (opsional untuk fitur temporal real‑time).
* **Fitur turunan (contoh)**:
  `total_donasi_sender`, `jumlah_transaksi_sender`, `rata_rata_donasi_sender`, `std_donasi_sender`, `jumlah_donasi_30hari_sender`, `selang_waktu_rata2_sender`, `receiver_unik_per_sender`, `sender_unik_per_receiver`, `degree_centrality_sender`, `degree_centrality_receiver`, `proporsi_donasi_terbesar_per_sender`, `flag_donasi_melebihi_batas`.

**Dataset contoh di repo**:

* `synthetic_donations.csv` → data transaksi sintetik dasar (mockup).
* `featured_synthetic_donations.csv` → hasil feature engineering siap latih.

> **Produksi**: pastikan proses feature engineering konsisten antara training & inference (gunakan pipeline yang sama atau service khusus fitur).

---

## ⚙️ Konfigurasi Penting

* **Class weight**: `scale_pos_weight` untuk menangani *imbalance*.
* **Thresholding (Option‑A)**: simpan ambang terpilih ke `artifacts/threshold.json` dan **jangan hard‑code** di service.
* **Kalibrasi** (opsional): `CalibratedClassifierCV` (isotonic) bila skor probabilitas perlu lebih reliabel.
* **Validasi**: pertimbangkan *time‑based split* (train lebih awal → uji periode berikutnya) dan *group split by sender* untuk mengurangi *leakage*.

---

## 🧭 Roadmap Singkat

* 🔍 **Graph features lanjutan** (betweenness, community) dan/atau GNN untuk pola kolusif (belum digunakan saat ini).
* 📊 **Monitoring & drift** (data drift, performance drift) + alerting.
* 🧾 **Explainability** (mis. feature importance tingkat lanjut) untuk audit (tanpa SHAP di repo saat ini).
* 🧱 **Hard rules** (KYC/AML) yang dapat dikonfigurasi oleh regulator.

---

## 🔐 Etika & Kepatuhan

Cakradana ditujukan sebagai **alat bantu analitik**. Hasil model **bukan** bukti pelanggaran; selalu lakukan **review manusia** dan rujuk pada regulasi yang berlaku. Jaga privasi, keamanan data, dan audit trail sesuai standar.