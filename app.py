import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load model dan encoder
baseline_model = pickle.load(open("baseline_model.pkl", "rb"))
isotonic_model = pickle.load(open("isotonic_model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
metrics_df = pickle.load(open("metrics.pkl", "rb"))  # berisi tabel metrik evaluasi
reliability_data = pickle.load(open("reliability.pkl", "rb"))  # data curve

st.title("XGBoost Probabilistic Calibration Demo")
st.write("Perbandingan Baseline vs Isotonic Calibration")

st.header("Input Form Prediksi")

age = st.number_input("Age", 18, 95, 30)
balance = st.number_input("Balance", -10000, 100000, 0)
day = st.number_input("Day", 1, 31, 5)
campaign = st.number_input("Campaign", 1, 50, 1)
pdays = st.number_input("Pdays", -1, 900, -1)
previous = st.number_input("Previous", 0, 300, 0)

job = st.selectbox("Job", encoder["job_classes"])
marital = st.selectbox("Marital", encoder["marital_classes"])
education = st.selectbox("Education", encoder["education_classes"])
contact = st.selectbox("Contact", encoder["contact_classes"])
month = st.selectbox("Month", encoder["month_classes"])
poutcome = st.selectbox("Poutcome", encoder["poutcome_classes"])
housing = st.selectbox("Housing Loan", ["yes", "no"])
loan = st.selectbox("Personal Loan", ["yes", "no"])
default = st.selectbox("Has Default?", ["yes", "no"])

# Build dataframe
input_dict = {
    "Age": age,
    "Balance": balance,
    "Day": day,
    "Campaign": campaign,
    "Pdays": pdays,
    "Previous": previous,
    "Job": job,
    "Marital": marital,
    "Education": education,
    "Default": default,
    "Housing": housing,
    "Loan": loan,
    "contact": contact,
    "Month": month,
    "poutcome": poutcome
}

input_df = pd.DataFrame([input_dict])

processed = encoder["transform"](input_df)

baseline_prob = baseline_model.predict_proba(processed)[:, 1][0]
iso_prob = isotonic_model.predict_proba(processed)[:, 1][0]

st.subheader("Hasil Prediksi Probabilitas")
st.write(f"**Baseline Probability:** {baseline_prob:.4f}")
st.write(f"**Isotonic-Calibrated Probability:** {iso_prob:.4f}")
st.write(f"**Perbedaan:** {abs(baseline_prob - iso_prob):.4f}")

st.header("Evaluasi Model")

st.subheader("Perbandingan Metrik Baseline vs Isotonic")
st.dataframe(metrics_df)

st.subheader("Reliability Curve")
fig, ax = plt.subplots()
ax.plot(reliability_data["baseline_x"], reliability_data["baseline_y"], label="Baseline")
ax.plot(reliability_data["iso_x"], reliability_data["iso_y"], label="Isotonic")
ax.plot([0,1], [0,1], linestyle="--", color="gray")
ax.set_xlabel("Predicted Probability")
ax.set_ylabel("Actual Frequency")
ax.legend()
st.pyplot(fig)

st.subheader("Confusion Matrix Baseline")
st.image("cm_baseline.png")

st.subheader("Confusion Matrix Isotonic")
st.image("cm_isotonic.png")
