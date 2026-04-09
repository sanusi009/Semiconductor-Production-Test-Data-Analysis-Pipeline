# 🔬 Semiconductor Production Test — Data Analysis Pipeline

> End-to-end Python pipeline for electrical sensor data preprocessing,  
> anomaly detection, and visualisation — aligned with IC manufacturing workflows.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4+-orange?style=flat-square&logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

---

## 📌 Overview

This project simulates a real-world **IC production test data pipeline** — the kind used in semiconductor companies to identify defective chips before they leave the factory floor.

The pipeline processes 1,200 chip test records across 7 electrical parameters, handles missing sensor values, detects anomalies using a **hybrid ML + statistical approach**, and produces a full visual report.

---

## 🧪 Dataset — IC Test Parameters

| Feature | Description | Unit |
|---|---|---|
| `VDD_mV` | Supply voltage | mV |
| `IDD_uA` | Quiescent current | µA |
| `TEMP_C` | Die temperature | °C |
| `FREQ_MHz` | Oscillator frequency | MHz |
| `LEAKAGE_nA` | Gate leakage current | nA |
| `VOUT_mV` | LDO output voltage | mV |
| `RISE_TIME_ns` | Signal rise time | ns |

- **1,200 chips** with **~5% injected defect rate**  
- **~3% missing values** simulating real sensor drop-outs

---

## ⚙️ Pipeline Stages
Raw Sensor Data
│
▼

Preprocessing
├── Median imputation  (missing values)
├── Winsorisation      [1% – 99%]
└── StandardScaler     (zero mean, unit variance)
│
▼
Anomaly Detection
├── Isolation Forest   (unsupervised ML, contamination=5%)
├── Z-Score filter     (|z| > 3 per feature)
└── Union flag         (chip flagged if either triggers)
│
▼
Visualisation
├── Feature distributions (normal vs anomalous)
├── Correlation heatmap
├── Anomaly score histogram
├── PCA 2D projection map
└── Feature sensitivity bar chart
---

## 📊 Results

| Metric | Value |
|---|---|
| Chips tested | 1,200 |
| Anomalies detected | 60 (5.0%) |
| Most sensitive feature | `IDD_uA` |
| PCA variance explained (2 PCs) | ~55% |

---

## 🚀 Run It

**Option 1 — Google Colab (recommended)**  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)  
Upload `Semiconductor Production Test Data Analysis Pipeline.ipynb` and click **Runtime → Run all**

**Option 2 — Local**
```bash
git clone https://github.com/sanusi009/semiconductor-analysis
cd semiconductor-analysis
pip install -r requirements.txt
python semiconductor_sensor_analysis.py
```

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `SciPy` · `Matplotlib`

---

## 👤 Author

**Sanusi Isiaka Olatunji**  
M.Sc. Data Science — University of Leoben  
[LinkedIn](https://linkedin.com/in/sanusi-olatunji-43990198) · [GitHub](https://github.com/sanusi009)
