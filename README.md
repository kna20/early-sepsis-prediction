# Will It Travel? Early Sepsis Prediction Across Hospitals

**Course:** EECE 490/690 – Introduction to Machine Learning  
**University:** American University of Beirut  
**Team:** Kareem Ajami & Bahiya Jawhar

## Project Overview

This project investigates whether machine learning models trained to predict sepsis in one hospital can generalize to patients in another hospital.

We compare different algorithms, clinical feature groups, missingness indicators, and temporal features to understand how these choices affect early sepsis prediction.

## Dataset

We use the [PhysioNet/Computing in Cardiology Challenge 2019 dataset](https://physionet.org/content/challenge-2019/1.0.0/), containing hourly ICU measurements from 40,336 patients across two publicly available hospitals.

The dataset includes 40 original features:
- 8 vital signs
- 26 laboratory measurements
- 6 demographic and administrative variables

Sepsis labels are designed to support prediction approximately six hours before onset.

## Methodology

Hospital A is divided into 80% development patients and 20% held-out test patients.

We use five-fold patient-level cross-validation on the development set for model comparisons. Hospital B serves as an external test dataset to assess cross-hospital generalization.

Our preprocessing includes causal forward filling, training-based median imputation, standardization, and optional missingness indicators.

## Experiments

**Completed:**
- Logistic Regression baselines
- Class weighting
- Missingness indicators and time since last measurement
- Clinical feature-group comparisons
- ICULOS ablation experiments
- Three-hour temporal features, including rolling means, variability, and changes

**Next steps:**
- Evaluate Logistic Regression on held-out Hospital A patients and Hospital B
- Implement and evaluate Random Forest
- Implement and evaluate XGBoost
- Compare model performance across hospitals

## Evaluation Metrics

We evaluate models using AUROC, AUPRC, precision, recall, F1-score, false positive rate, and the PhysioNet clinical utility score.

## Repository Structure

- `sepsis.py` — Shared data-loading, preprocessing, and evaluation functions
- `02_holdout_70_15_15.ipynb` — Initial holdout experiments
- `03_baseline_cv.ipynb` — Logistic Regression baselines
- `04_missingness.ipynb` — Missingness experiments
- `05_feature_groups.ipynb` — Clinical feature groups
- `06_iculos_check.ipynb` — ICULOS investigation
- `07_temporal_features.ipynb` — Temporal feature engineering
- `archive/` — Earlier exploratory and superseded work
- `requirements.txt` — Python dependencies

## Tools

Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, and Jupyter/Google Colab.

## Dataset Access

The dataset is available publicly through PhysioNet. Raw patient files are not included in this repository.
