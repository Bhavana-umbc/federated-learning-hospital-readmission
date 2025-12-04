# 🚑 Federated Learning for Hospital Readmission Prediction  
### *Privacy-Preserving AI for 30-Day Readmission Risk in Diabetic Patients*

This project implements a **fully working Federated Learning system** designed for healthcare settings — improving prediction performance **without sharing sensitive patient data**.

Developed as part of **DATA 606 – Capstone in Data Science** at UMBC.

---

## 🎯 Project Objectives

- Predict 30-day hospital readmission for diabetic patients  
- Enable collaboration between hospitals with **strict privacy**
- Protect data using **Differential Privacy**
- Demonstrate **real MLOps features**: versioning, rollback, live UI
- Support **continuous model improvement** using new patient data

---

## 🏥 System Highlights

This project goes beyond normal ML — it’s an **end-to-end production simulation**.

### 🔐 Federated Learning + Differential Privacy
- Local training at each simulated hospital
- Only **gradient updates** (with noise) are shared
- No raw patient data leaves the hospital

### 🏛️ Model Registry & Versioning
- Models stored as `G.major.minor.patch`
- Promotion only if metrics improve (AUC + ECE)
- **Rollback** supported — hospitals can restore earlier versions

### 🏥 Hospital Autonomy
- Each hospital can:
  - Follow **global stable** model
  - **Pin** to their preferred version
  - **Rollback** if performance declines

### 🖥️ Gradio UI – Real Deployment Simulation
- Username/password login per hospital
- Prediction interface showing risk + probability + version used
- Admin controls: switch version, rollback, maintenance

---

## 🧠 Technical Overview

| Component | Status | Description |
|----------|:-----:|-------------|
| Baseline Model | ✅ | Logistic Regression with shared preprocessing |
| FL Setup | ✅ | 4 hospitals (A large, B/C/D small) |
| Privacy | ✅ | Gradient clipping + Gaussian noise |
| Model Registry | ✅ | Semantic versioning + metrics |
| Continuous Learning | ✅ | Monthly updates using Future Pool |
| UI | ✅ | Login, Prediction & Admin panels |

---

## 📂 Repository Contents

| File | Description |
|------|-------------|
| `Final_code.ipynb` | Main notebook with full FL pipeline + UI |
| `diabetic_data.csv` | Input dataset (if included) |
| `README.md` | Documentation (this file) |

---

## ▶️ How to Run

1️⃣ Open **`Final_code.ipynb` in Google Colab**  
2️⃣ Install dependencies (automatically handled inside notebook)  
3️⃣ Upload the dataset when prompted  
   📌 Download link:  
   https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008  
4️⃣ Run **all cells** to:
- Perform cleaning & EDA
- Train baseline + federated models
- Launch the Gradio UI

5️⃣ Click the UI link → Login → Make predictions → Try rollback/version switching

---

## 🧪 Demonstrated Features

| Feature | Supported |
|--------|:--------:|
| Local-only model training | ✔ |
| FL with DP-protected updates | ✔ |
| Model versioning & registry | ✔ |
| Candidate promotion gating | ✔ |
| Rollback / Unpin / Switch version | ✔ |
| Monthly federated refresh | ✔ |
| Local prediction logging | ✔ |

---

## 📊 Why This Matters

> Small hospitals struggle because their data is limited — Federated Learning lets them benefit from collective intelligence **without ever exposing patient data**.

This project demonstrates:
- Improved fairness in healthcare AI  
- Compliance with privacy laws  
- Realistic deployment behavior in hospitals  
- Production-style lifecycle for medical models

---

## 🔮 Future Enhancements

- Replace logistic regression with XGBoost or Neural Networks  
- Add secure aggregation (no server visibility of updates)  
- Deploy across multiple machines instead of simulation  
- Include model monitoring dashboards  

---

## 👩‍💻 Author

**Bhavana Vemam Reddy**  
Graduate Student — UMBC  
*DATA 606 Capstone in Data Science*
