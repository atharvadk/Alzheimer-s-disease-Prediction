🧠 Alzheimer’s Disease Feature Analysis & Machine Learning Models

This repository contains data, scripts, and model outputs for an end-to-end machine-learning analysis of Alzheimer’s disease. The project evaluates multiple feature-selection techniques, performs class balancing, and compares model performance across multiple train/test splits.

📁 Repository Structure
cp/
├── alzheimers_disease_data.csv
├── balanced_alzheimers_data.csv
├── 70_30_original.csv
├── 75_25_original.csv
├── 80_20_original.csv
├── smote.R
│
├── importance/            # Feature importance results
├── lasso/                 # LASSO regression outputs
├── non-medical/           # Experiments using non-medical predictors only
├── rf/                    # Random Forest models & comparisons
├── rfe/                   # Recursive Feature Elimination outputs
└── temp/                  # Intermediate/temporary files

🎯 Project Objectives

Identify the most predictive features for Alzheimer’s disease.

Compare multiple feature-selection methods:

Random Forest feature importance

LASSO regression

Recursive Feature Elimination (RFE)

Evaluate model stability across multiple data splits (70/30, 75/25, 80/20).

Assess predictive power of non-medical features.

Address class imbalance using SMOTE.

🔬 Workflow Overview
1. Data Preparation

Load the main Alzheimer’s dataset.

Create multiple train/test splits.

Apply SMOTE (optional) using smote.R.

2. Feature Selection

Feature-selection results are located in:

importance/

lasso/

rfe/

rf/ (RF-based ranking)

Each folder contains CSV outputs and analysis artifacts.

3. Model Training

Random Forest models are trained across:

Full feature set

Selected features from LASSO / RFE / RF

Non-medical-only subsets

Performance comparisons for each split are stored inside rf/.

4. Cross-Split Comparison

To evaluate robustness, the results across all splits (70/30, 75/25, 80/20) are compared, identifying:

Consistently selected features

Stable importance rankings

Features with strong predictive contribution

📊 Included Data Files
File	Description
alzheimers_disease_data.csv	Raw Alzheimer’s dataset
balanced_alzheimers_data.csv	SMOTE-balanced dataset
XX_YY_original.csv	Train/test split datasets (70/30, 75/25, 80/20)
🧪 Reproducing the Analysis
Run SMOTE balancing
source("smote.R")

Re-run feature-selection scripts

Each folder (e.g., lasso/, rfe/) contains the specific R code and outputs needed.

Review model results

Inside rf/ you will find:

Feature importance tables

Model performance metrics

Cross-split comparison CSVs

🚀 Future Enhancements

Add unified R scripts for full pipeline reproducibility

Add visualizations for cross-method feature importance

Expand models (XGBoost, logistic regression, SVM)

Convert findings into a research-ready PDF report

📄 License

This project is provided for research and educational purposes. Add your license information here if needed.
