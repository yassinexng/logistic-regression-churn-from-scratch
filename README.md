# 📊 Logistic Regression Analysis of Telco Customer Churn

This repository contains a **from-scratch Python implementation** of a Logistic Regression model, applied to the **Telco Customer Churn dataset**.

---

## 🧠 Project Description

This project is a from-scratch Python implementation of a Logistic Regression model to analyze and predict **Telco customer churn**.  
It covers the complete **data science pipeline**, from data cleaning and feature engineering to model training using **Gradient Ascent** and **Maximum Likelihood Estimation (MLE)**.

The focus is on **understanding the mathematical foundation** of logistic regression — including the sigmoid function, log-odds, and evaluation metrics such as **AUC** — all built **without relying on high-level ML libraries**.

---

## 📂 Repository Structure

1. **`Logistic_Regression_Analysis_of_Telco_Customer_Churn.py`** – The complete, runnable Python script containing the code for:
   - Model definition
   - Data preprocessing
   - Training and evaluation

2. **`Logistic_Regression_Analysis_of_Telco_Customer_Churn.pdf`** – A detailed **academic companion paper** explaining the underlying statistical and mathematical theory.

---

## 📘 Project Overview

> “The dataset used for this analysis contains core information about the Telco Customer Churn, specifically focusing on the customers who left within the last month.”  
> “The goal of this project is to use the data in the feature columns such as `tenure`, `Contract`, and `MonthlyCharges` to predict the value in the `Churn` column for a new, unseen customer.”

We began with the **core theoretical foundations** — sigmoid, log-odds, MLE, and gradient ascent — then implemented everything **entirely from scratch**, without using any pre-built ML functions.

---

## ⚙️ Features

This implementation was **built from the ground up**, including:

- **Core Model:**  
  `sigmoid` and `predict_probability` functions

- **Cost Function:**  
  `compute_log_likelihood` using Maximum Likelihood Estimation (MLE)

- **Optimization:**  
  `train_model` using **Gradient Ascent** to find optimal parameters  

- **Data Preparation:**  
  Full pipeline for data cleaning, one-hot encoding, and feature scaling

- **Evaluation:**  
  Custom-built functions for:
  - `confusion_matrix`
  - `precision`
  - `recall`
  - `accuracy`
  - `f1_score`
  - `compute_roc_auc`

---

## 🚀 How to Run

This project is **self-contained** and generates its own sample data for demonstration purposes.

### 🧩 Prerequisites

You only need the following libraries:

```bash
pip install numpy pandas
```

### ▶️ Running the Analysis

```bash
git clone https://github.com/your-username/logistic-regression-churn-from-scratch.git
cd logistic-regression-churn-from-scratch
python Logistic_Regression_Analysis_of_Telco_Customer_Churn.py
```

---

## 📈 Example Results

### 🔁 Training Convergence

The Log-Likelihood (score) increases with each iteration, showing that the model is learning properly.

```
Training model...
Step 0:   Score = -0.6912
Step 100: Score = -0.5234
Step 200: Score = -0.4987
Step 300: Score = -0.4876
Step 400: Score = -0.4823
Step 499: Score = -0.4798
```

### 📊 Model Performance Metrics

```
The Metrics:
  Accuracy:  0.825
  Precision: 0.700
  Recall:    0.712
  F1-Score:  0.706
  AUC:       0.847
```

### 💡 Feature Insights

```
Contract (Month-to-month): OR = 2.34
→ Month-to-month customers are 2.34× more likely to churn.

Tenure: OR = 0.68
→ Each extra month decreases churn odds by 32%. Loyalty compounds.

Monthly Charges: OR = 1.42
→ Higher bills increase churn by 42%.
```

---

## 📖 Theoretical Deep Dive

For a detailed explanation of the **mathematical and statistical principles**, see the companion PDF report.  
It covers the following sections:

- **The Theory:** The Logistic Model  
- **The Learning Process:** Maximum Likelihood  
- **The Optimization:** Gradient Ascent  
- **Data Preparation:** Telco Churn Dataset  
- **Model Evaluation:** Beyond Training

---

## 👨‍💻 Author

**Yassine Mouadi**  
📅 *October 2025*

---

> *“This project bridges the gap between pure theory and real implementation — showing how logistic regression truly learns from data.”*
