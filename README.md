# Diabetes Prediction App (Machine Learning (Logistic-Regression) + Streamlit)

A complete end-to-end Machine Learning project that predicts whether a person is diabetic based on medical attributes.  
This project covers the full ML lifecycle: EDA → Preprocessing → Model Building → Evaluation → Deployment using Streamlit.

## 🚀 Live Demo
👉 https://your-streamlit-app-link.streamlit.app

---

## Project Overview

This project uses the **Pima Indians Diabetes Dataset** to build a binary classification model that predicts diabetes risk based on health metrics like glucose, BMI, insulin, and age.

The trained model is deployed using **Streamlit**, allowing users to interactively input data and get real-time predictions.

---

## Objectives

- Perform Exploratory Data Analysis (EDA)
- Handle missing and invalid values
- Train a Logistic Regression model
- Evaluate performance using classification metrics
- Deploy the model using Streamlit

---

## Dataset

- Records: 768 patients  
- Features: 8 medical attributes  
- Target: `Outcome` (0 = Non-diabetic, 1 = Diabetic)

### Features:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

---

## Exploratory Data Analysis (EDA)

- Distribution analysis using histograms
- Outlier detection using box plots
- Correlation analysis
- Identified invalid zero values in medical features

---

## Data Preprocessing

- Replaced invalid zero values with NaN
- Applied median imputation
- Feature scaling using StandardScaler

---

## Model Building

**Algorithm Used:** Logistic Regression  
- Train-test split: 80/20
- Feature scaling applied
- Binary classification model

---

## Model Evaluation

| Metric | Score |
|--------|------|
| Accuracy | ~71% |
| Precision | ~60% |
| Recall | ~50% |
| F1-score | ~0.55 |
| ROC-AUC | ~0.81 |

The ROC curve indicates good class separability.

---

## Interpretation

- Most influential features:
  - Glucose (strongest predictor)
  - BMI
  - Age
- Model aligns with real-world medical insights

---

## Deployment (Streamlit)

The model is deployed using **Streamlit** for interactive predictions.

### Features:
- User-friendly input interface
- Real-time prediction
- Probability output
- Custom UI with background styling


## Install Dependencies
pip install -r requirements.txt

## Run Streamlit App
streamlit run app.py

## Author

Deepashree
Aspiring Data Analyst / Data Scientist
Passionate about building real-world ML projects and deploying intelligent applications.
