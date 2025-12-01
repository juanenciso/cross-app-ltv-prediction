# 🎯 Cross-App Lifetime Value (LTV) Prediction Using Multimodal Behavioral Data

A production-grade machine learning workflow for predicting **user Lifetime Value (LTV)** across multiple apps using **multimodal behavioral data**.

This project simulates a real AdTech environment where users interact with several apps, generating:
- 🧩 **Sequential events** (modeled with Transformers)
- 📊 **Aggregated tabular features** (engagement, revenue, retention)

The goal is to build a **multimodal ML pipeline** that significantly outperforms tabular-only baselines.

---

## 📌 1. Why This Matters (Problem Overview)

Accurately predicting user LTV is essential for:

- 📈 Acquisition bidding optimization (CPI / CPA)
- 💸 ROAS forecasting and budget allocation
- 🔍 Early identification of high-value segments
- 🔁 Cross-app engagement modeling
- 🧠 Portfolio-wide user understanding

This repository shows how to combine **sequence modeling + tabular modeling** for improved predictive accuracy.

---

## 🧠 2. Technical Approach

### **2.1 Data Modalities**
The project uses *two* feature types:

#### 🔹 Sequential Input  
Time-ordered user events (per app), modeled with **Transformers**:
- session length  
- view count  
- app launch sequence  
- completion ratios  
- engagement streaks  

#### 🔹 Tabular Input
Aggregated behavioral metrics:
- total revenue  
- average retention  
- total sessions  
- ARPU  
- churn probability proxies  

---

## 🧱 3. Model Architecture

### 🔸 **Multimodal Fusion Model**
- Transformer encoder → event embeddings  
- Tabular MLP → dense features  
- Concatenation → fusion layer  
- Regression head → predicted LTV  

Includes:
- 🧪 PyTorch Lightning training loop  
- 🧮 XGBoost/LinearRegression baselines  
- 🎛 Automatic validation metrics  

---

## 🔧 4. Pipeline Steps

1. Generate synthetic multimodal dataset  
2. Prepare event sequences + tabular matrices  
3. Train multimodal Transformer fusion model  
4. Evaluate on hold-out test set  
5. Train and compare baseline tabular model  
6. Print metrics (R², MAE)  

---

## 🧪 5. Results

From your run:

| Model | R² | MAE | Notes |
|-------|------|-------|--------|
| **Transformer + Tabular** | **0.9860** | **2.40** | ✔ Best performance |
| **Linear Regression (tabular-only)** | 0.9860 | 2.68 | Worse MAE |

📌 **~10% MAE improvement** → sequence modeling adds meaningful predictive power.

---

## 📁 6. Repository Structure

cross-app-ltv-prediction/
│── data/
│ ├── multimodal_events.csv
│ ├── tabular_features.csv
│
│── src/
│ ├── generate_synthetic_data.py
│ ├── dataset.py
│ ├── model.py
│ ├── train.py
│ ├── evaluate.py
│
├── requirements.txt
├── README.md
└── .gitignore


---

## 📦 7. Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 8. Training

Run full training pipeline:

```
python src/train.py
```

---

## 🧪 9. Evaluation

```
python src/evaluate.py
```

Outputs include:

R² score

MAE

Baseline vs multimodal comparison

---

Optional model checkpoints

## ⭐ 10. Key Features

✔ Synthetic user–action dataset generator

✔ Transformer-based sequential encoder

✔ Tabular + sequential fusion

✔ PyTorch Lightning training

✔ XGBoost/Linear regression baselines

✔ Metrics for direct comparison

✔ Fully reproducible project

---

## 🧩 Future Improvements

Add LSTM or CNN sequence encoders

Add GBDT fusion (CatBoost/XGBoost)

Add Databricks/mlflow integration

Cross-validation on temporal splits

---

## 🙋‍♂️ Author

Juan Sebastián Enciso García
Data Scientist | Machine Learning | Reinforcement Learning



