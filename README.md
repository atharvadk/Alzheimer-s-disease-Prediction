# 🧠 Alzheimer’s Disease Feature Analysis & Machine Learning Models
An end-to-end machine learning project designed to analyze Alzheimer’s Disease datasets, perform feature selection, apply SMOTE for class balancing, and evaluate model performance across multiple train/test splits. The goal is to identify stable, clinically meaningful predictors of Alzheimer’s disease.

## 📑 Table of Contents
- [About](#about)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Tech Stack](#tech-stack)
- [Screenshots](#screenshots)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## 🔍 About
This project analyzes Alzheimer’s disease using multiple machine learning techniques in R. It evaluates different feature-selection methods (LASSO, RFE, Random Forest), applies SMOTE to handle class imbalance, and compares model performance across multiple train/test splits (70/30, 75/25, 80/20).  
The project is intended for data scientists, researchers, and clinicians exploring early-stage detection or interpretable predictive modeling.

## ✨ Features
- SMOTE-based class balancing  
- Feature selection using LASSO, RFE, and Random Forest  
- Random Forest modeling across multiple train/test splits  
- Non-medical variable performance analysis  
- Cross-split feature stability comparison  
- Clear CSV outputs for all experiments  

## 📁 Project Structure
```
cp/
├── alzheimers_disease_data.csv
├── balanced_alzheimers_data.csv
├── 70_30_original.csv
├── 75_25_original.csv
├── 80_20_original.csv
├── smote.R
│
├── importance/
├── lasso/
├── non-medical/
├── rf/
├── rfe/
└── temp/
```

## 🛠 Installation
```bash
git clone [https://github.com/your/repo.git](https://github.com/atharvadk/Alzheimer-s-disease-Prediction)
cd repo
```

Install required R packages:
```r
install.packages(c("tidyverse", "randomForest", "glmnet", "caret", "DMwR"))
```

## 🚀 Usage
Run SMOTE balancing:
```r
source("smote.R")
```

Run feature-selection and modeling scripts:
```r
source("rf/run_rf_model.R")
source("lasso/run_lasso.R")
source("rfe/run_rfe.R")
```

## ⚙️ Configuration
Required R packages:
```
tidyverse
randomForest
glmnet
caret
DMwR
```

## 🧱 Tech Stack
- **Language:** R  
- **Models:** Random Forest, LASSO, RFE  
- **Data Processing:** SMOTE, tidyverse  
- **Outputs:** CSV, R scripts  

## 🖼 Screenshots
_Add screenshots or visual outputs if needed._

## 🗺 Roadmap
- [ ] Add unified end-to-end pipeline  
- [ ] Add feature stability visualizations  
- [ ] Add alternative models (XGBoost, Logistic Regression)  
- [ ] Publish research report  

## 🤝 Contributing
1. Fork the repo  
2. Create a branch  
3. Commit changes  
4. Open a pull request  

## 🙌 Acknowledgements
Thanks to Alzheimer’s research communities, data contributors, and the R open-source community.
