# early-sepsis-prediction
# Early Sepsis Prediction Using Machine Learning

**A Comparison of Algorithms and Clinical Feature Groups**

**Course:** EECE 490/690 – Introduction to Machine Learning  
**University:** American University of Beirut  
**Team:** Kareem Ajami & Bahiya Jawhar

## Project Overview

This project investigates early sepsis prediction using machine learning on ICU patient data. Our goal is to compare different algorithms and determine which groups of clinical features contribute most to predicting sepsis approximately six hours before onset.

## Dataset

We use the [PhysioNet/Computing in Cardiology Challenge 2019 dataset](https://physionet.org/content/challenge-2019/1.0.0/), containing 40,336 ICU patient records with hourly measurements of 40 features.

The dataset includes:
- 8 vital signs
- 26 laboratory measurements
- 6 demographic and administrative features

## Proposed Methodology

We plan to compare three algorithms:
1. Logistic Regression (baseline)
2. Random Forest
3. XGBoost

We will investigate different clinical feature groups individually and in combination.

Hospital A will be used for training and validation, with an 80/20 patient-level split. Hospital B will be reserved for external testing.

Missing measurements will be handled using forward filling and training-set-based median imputation.

## Evaluation

We plan to evaluate the models using:
- AUROC and AUPRC
- Precision, Recall, and F1-score
- PhysioNet clinical utility score

## Future Experiments

If time permits, we will investigate temporal feature engineering using rolling averages, variability, and measurement changes over 3-hour and 6-hour windows.

## Tools

Python, Pandas, NumPy, Scikit-learn, XGBoost, and Matplotlib.

## Project Status

Initial development — data exploration and baseline implementation.

## Dataset Access

The dataset is available from PhysioNet. Raw patient files are not included in this repository.
