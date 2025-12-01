Cross-App Lifetime Value (LTV) Prediction Using Multimodal Behavioral Data

This project implements a scalable machine learning workflow for predicting user Lifetime Value (LTV) across multiple mobile apps using multimodal features combining:

Sequential behavioral data (event sequences modeled with Transformers)

Tabular aggregated features (engagement, revenue, retention signals)

It is designed following the standards used in large AdTech, gaming, and mobile analytics companies.

1. Problem Overview

Accurately predicting user LTV is fundamental for:

Acquisition bidding and CPI optimization

Budget allocation and ROAS forecasting

Identifying high-value user segments early

Portfolio-wide cross-app engagement modeling

This project simulates a realistic scenario where users interact with several apps in a portfolio, producing sequences of events and aggregated behavioral metrics.
A multimodal model is trained to improve predictive accuracy compared to standard tabular baselines.

2. Technical Approach
2.1 Data Components

The pipeline uses two feature modalities:

Sequential Input

Time-ordered user events per app

Each event encoded as a vector

Processed through a Transformer encoder

Tabular Input

Numerical engagement and retention aggregates

Processed through a small feed-forward network

Outputs of both branches are fused to predict final LTV.

3. Model Architecture
3.1 Multimodal Model (Transformer + MLP)
Sequential events → Transformer Encoder → Event Embedding
Tabular features → MLP Block        → Tabular Embedding
                                  
[Fusion: concatenation]

Combined embedding → Regression Head → Predicted LTV


The multimodal model captures both long-term temporal structure (via self-attention) and global behavioral signals (via tabular features).

4. Baselines

To evaluate modeling impact, a baseline is included:

Linear Regression (tabular only)
Traditional approach used in mobile analytics and BI pipelines.

The comparison quantifies the value of sequential modeling.

5. Results
5.1 Multimodal Model (Transformer + Tabular)

R²: 0.9860

MAE: 2.4063

5.2 Baseline Linear Regression

R²: 0.9860

MAE: 2.6855

5.3 Interpretation

The multimodal model achieves a ~10% reduction in MAE while keeping the same R².
This demonstrates that sequential modeling adds meaningful predictive signal beyond aggregated tabular metrics.

6. Training Pipeline
Steps:

Generate synthetic multimodal dataset

Prepare sequential tensors and tabular matrices

Train Transformer + Tabular fusion model using PyTorch Lightning

Evaluate performance on test set

Train baseline linear model

Compare metrics and visualize results

7. Repository Structure
.
├── data/
│   ├── multimodal_events.csv
│   ├── tabular_features.csv
├── src/
│   ├── generate_synthetic_data.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
├── requirements.txt
├── README.md
└── .gitignore

8. Requirements

Install dependencies:

pip install -r requirements.txt

9. Training

Run the full training pipeline:

python src/train.py

10. Evaluation
python src/evaluate.py


Outputs include:

R²

MAE

Baseline comparison

Optionally: saved model checkpoints

11. Key Advantages of This Approach

Handles multimodal data (sequences + tabular)

Captures long-term behavioral patterns

More robust to delayed labels and censored revenue

Architecture scales to 100M+ events

Production-ready structure using Lightning

12. Future Extensions

Replace Transformer with a causal attention architecture

Add CatBoost model for improved tabular fusion

Incorporate survival models for censored LTV

Build MLflow tracking pipeline

Export model to batch/real-time inference service


