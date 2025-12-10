# XGBoost Probabilistic Calibration Web App

### A Streamlit Application for Probability Calibration and Model Evaluation

---

## 📌 Overview

This project implements a probabilistic calibration workflow for an XGBoost classification model using the Bank Marketing Dataset. The goal is to improve the reliability of predicted probabilities using **Isotonic Regression** and compare it against the **Baseline XGBoost model** and **Platt Scaling**.

The final result is deployed as a **Streamlit web application**, allowing users to:

* Input customer features
* Generate predicted probabilities (baseline vs calibrated)
* Visualize calibration curves
* Compare model performance metrics
* Inspect confusion matrices

This application is designed for academic demonstration, particularly for a **thesis defense**, but the structure and workflow follow industry best practices.

---

## 🚀 Live Demo

(Add your Streamlit Cloud URL after deployment)

```
https://share.streamlit.io/<your-username>/<your-repo>
```

---

## 📂 Project Structure

```
bank_marketing_calibration_app/
│
├── app.py                          # Main Streamlit application
│
├── models/                         # Saved ML models & preprocessing
│   ├── baseline_model.pkl
│   ├── isotonic_model.pkl
│   ├── encoder.pkl
│
├── artifacts/                      # Evaluation-related artifacts
│   ├── metrics.pkl
│   ├── reliability.pkl
│   ├── cm_baseline.png
│   ├── cm_isotonic.png
│
├── requirements.txt                # Dependencies for Streamlit Cloud
│
└── README.md                       # Project documentation
```

---

## 🧠 Modeling Workflow

### 1. Data Preparation

* Handle missing values
* Rare category grouping
* Categorical encoding (one-hot)
* Numerical transformations (log-transform, special-value handling)
* Removal of leakage features (e.g., Duration)

### 2. Model Training

Models trained:

1. Baseline XGBoost
2. Platt Scaling Calibration
3. Isotonic Regression Calibration

Only the **Baseline** and **Isotonic** models are deployed in the app, as Isotonic provides the best probabilistic calibration performance.

---

## 🧪 Evaluation Metrics Included

The application includes a comparison of:

* **Accuracy**
* **ROC-AUC**
* **Brier Score**
* **Expected Calibration Error (ECE)**

Visual artifacts:

* Reliability (Calibration) Curve
* Confusion Matrix (Baseline vs Isotonic)

---

## 🖥️ Streamlit App Features

### 🔹 Input Form

Interactive form for users to manually input customer attributes.

### 🔹 Probability Comparison

Displays:

* Baseline predicted probability
* Isotonic-calibrated probability
* Absolute difference between the two

### 🔹 Model Evaluation Dashboard

Includes:

* Metrics table
* Reliability curve visualization
* Confusion matrices

---

## 📦 Installation

### 1. Clone the Repository

```
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Run Locally

```
streamlit run app.py
```

---

## 🌐 Deployment (Streamlit Cloud)

1. Push the project to GitHub
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Log in with GitHub
4. Select your repository and choose `app.py`
5. Deploy

---

## 📜 License

This project is intended for academic use. Modify and adapt freely for research or educational purposes.

---

## 🙌 Acknowledgements

This project is based on research exploring probability calibration techniques for machine learning models using XGBoost and Isotonic Regression.
