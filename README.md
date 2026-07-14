# Diabetes Progression Prediction: Regularization Techniques

This repository features a Python-based machine learning project that predicts diabetes disease progression using baseline clinical features. The project implements and compares standard **Linear Regression** against regularized models, specifically **Ridge (L2)** and **Lasso (L1)** regressions, utilizing cross-validation to optimize hyperparameter tuning.

---

## 📌 Project Overview
When training standard linear regression models, overfitting can become an issue if features are highly correlated or numerous. This project demonstrates how to address overfitting and improve model generalizability using regularization techniques:
* **Linear Regression:** Baseline model without any regularization penalty.
* **Ridge Regression ($L_2$ Regularization):** Adds a penalty equal to the square of the magnitude of coefficients to prevent extreme weight values.
* **Lasso Regression ($L_1$ Regularization):** Adds a penalty equal to the absolute value of the magnitude of coefficients, which can drive irrelevant feature weights completely to zero (feature selection).

---

## 📊 Dataset
The project uses the **Diabetes Dataset** bundled within `scikit-learn`. 
* **Samples:** 442
* **Dimensionality:** 10 baseline predictor variables (Age, Sex, Body Mass Index, Average Blood Pressure, and six blood serum measurements).
* **Target:** A quantitative measure of disease progression one year after baseline.

---

## 🛠️ Requirements & Installation

To run this code locally, you need Python installed along with the following libraries:

```bash
pip install numpy pandas matplotlib scikit-learn
