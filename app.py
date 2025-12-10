import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load encoder and models
encoder = pickle.load(open("models/encoder.pkl", "rb"))
baseline_model = pickle.load(open("models/baseline_model.pkl", "rb"))
isotonic_model = pickle.load(open("models/isotonic_model.pkl", "rb"))
metrics_df = pickle.load(open("artifacts/metrics.pkl", "rb"))
reliability = pickle.load(open("artifacts/reliability.pkl", "rb"))

# Transform input function
def transform_input(input_df, encoder):
    df = input_df.copy()
    df["Pdays_zero"] = (df["Pdays"] == -1).astype(int)
    df["Pdays"] = df["Pdays"].replace(-1, 0)
    df["Campaign_log"] = np.log1p(df["Campaign"])
    df["Previous_log"] = np.log1p(df["Previous"])
    df["Job"] = df["Job"].replace(["housemaid", "unknown"], "rare")
    df["Education"] = df["Education"].replace("unknown", "rare")
    df["Month"] = df["Month"].replace(["dec", "mar"], "rare")
    df = pd.get_dummies(df)
    final_columns = encoder["final_columns"]
    for col in final_columns:
        if col not in df:
            df[col] = 0
    df = df[final_columns]
    return df

st.title("XGBoost Probabilistic Calibration Web App")
st.write("Perbandingan probabilitas baseline dan isotonic calibration.")

st.header("Input Form Prediksi")

age = st.number_input("Age", min_value=18, max_value=95, value=30)
balance = st.number_input("Balance", min_value=-10000, max_value=100000, value=0)
day = st.number_input("Day", min_value=1, max_value=31, value=1)
campaign = st.number_input("Campaign", min_value=1, max_value=50, value=1)
pdays = st.number_input("Pdays", min_value=-1, max_value=900, value=-1)
previous = st.number_input("Previous", min_value=0, max_value=300, value=0)

job = st.selectbox("Job", encoder["job_classes"])
marital = st.selectbox("Marital", encoder["marital_classes"])
education = st.selectbox("Education", encoder["education_classes"])
contact = st.selectbox("Contact Type", encoder["contact_classes"])
month = st.selectbox("Month Contacted", encoder["month_classes"])
poutcome = st.selectbox("Previous Outcome", encoder["poutcome_classes"])
housing = st.selectbox("Housing Loan", ["yes", "no"])
loan = st.selectbox("Personal Loan", ["yes", "no"])
default = st.selectbox("Default", ["yes", "no"])

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
processed = transform_input(input_df, encoder)

baseline_prob = baseline_model.predict_proba(processed)[:, 1][0]
isotonic_prob = isotonic_model.predict_proba(processed)[:, 1][0]

st.subheader("Probabilitas Hasil Prediksi")
st.write(f"Baseline Probability: **{baseline_prob:.4f}**")
st.write(f"Isotonic-Calibrated Probability: **{isotonic_prob:.4f}**")
st.write(f"Perbedaan: **{abs(baseline_prob - isotonic_prob):.4f}**")

st.header("Perbandingan Evaluasi Model")
st.dataframe(metrics_df)

st.subheader("Reliability Curve")
fig, ax = plt.subplots()
ax.plot(reliability["baseline_x"], reliability["baseline_y"], label="Baseline")
ax.plot(reliability["iso_x"], reliability["iso_y"], label="Isotonic")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax.set_xlabel("Predicted Probability")
ax.set_ylabel("Actual Frequency")
ax.legend()
st.pyplot(fig)

st.subheader("Confusion Matrix Baseline")
st.image("artifacts/cm_baseline.png")

st.subheader("Confusion Matrix Isotonic")
st.image("artifacts/cm_isotonic.png")