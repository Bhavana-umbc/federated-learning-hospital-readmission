# Federated Learning for Hospital Readmission Prediction

This repository contains the code for my SENG 691 project:
**Privacy-Preserving Federated Learning for 30-day hospital readmission risk prediction.**

## Contents

- `Final_code.ipynb`  
  Main Google Colab notebook implementing:
  - Data loading and EDA
  - Baseline logistic regression model
  - Simulation of 4 virtual hospitals (A, B, C, D)
  - Federated learning pipeline (SGD + FedAvg + DP-style noise)
  - Model registry and versioning (G.x.y.z)
  - Gradio UI with hospital login, prediction, and admin controls
  - Local-only prediction logging per hospital

## How to Run

1. Open `Final_code.ipynb` in Google Colab.
2. Install dependencies (Gradio, scikit-learn, joblib, etc. as in the notebook).
3. Upload the UCI Diabetes dataset (`diabetic_data.csv`) or modify the path.
4. Run all cells to:
   - Generate EDA outputs and baseline models
   - Train federated models
   - Launch the Gradio UI.

## Dataset

UCI Diabetes readmission dataset:  
https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008
