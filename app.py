from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "churn_ann_model.keras"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "feature_columns.json"


st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="centered",
)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #334155 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.2);
    }
    .hero h1 {
        margin: 0;
        font-size: 2.1rem;
    }
    .hero p {
        margin: 0.5rem 0 0;
        opacity: 0.9;
    }
    .result-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="hero">
        <h1>Customer Churn Predictor</h1>
        <p>Enter customer details to estimate the probability of churn using the trained ANN model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model file: {MODEL_PATH}. Run model.ipynb and save the artifacts first."
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Missing scaler file: {SCALER_PATH}. Run model.ipynb and save the artifacts first."
        )

    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(
            f"Missing feature columns file: {FEATURE_COLUMNS_PATH}. Run model.ipynb and save the artifacts first."
        )

    model = load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as file_handle:
        scaler = pickle.load(file_handle)
    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as file_handle:
        feature_columns = json.load(file_handle)

    return model, scaler, feature_columns


try:
    model, scaler, feature_columns = load_artifacts()
except Exception as error:
    st.error(str(error))
    st.stop()


with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600, step=1)
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.number_input("Age", min_value=18, max_value=100, value=40, step=1)
        tenure = st.number_input("Tenure", min_value=0, max_value=10, value=3, step=1)
        balance = st.number_input("Balance", min_value=0.0, max_value=250000.0, value=60000.0, step=1000.0)

    with col2:
        num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2, step=1)
        has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
        is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])
        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            max_value=500000.0,
            value=80000.0,
            step=1000.0,
        )
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

    submitted = st.form_submit_button("Predict Churn")


if submitted:
    raw_row = pd.DataFrame(
        [
            {
                "CreditScore": credit_score,
                "Geography": geography,
                "Gender": gender,
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": num_products,
                "HasCrCard": 1 if has_cr_card == "Yes" else 0,
                "IsActiveMember": 1 if is_active_member == "Yes" else 0,
                "EstimatedSalary": estimated_salary,
            }
        ]
    )

    processed_row = raw_row.copy()
    processed_row["Gender"] = processed_row["Gender"].map({"Female": 0, "Male": 1})
    processed_row = pd.get_dummies(processed_row, columns=["Geography"], drop_first=True)
    processed_row = processed_row.reindex(columns=feature_columns, fill_value=0)

    scaled_row = scaler.transform(processed_row)
    churn_probability = float(model.predict(scaled_row, verbose=0)[0][0])
    stay_probability = 1.0 - churn_probability
    churn_label = "WILL CHURN" if churn_probability >= 0.5 else "WILL STAY"

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.subheader("Prediction")
    st.metric("Churn probability", f"{churn_probability * 100:.2f}%")
    st.metric("Stay probability", f"{stay_probability * 100:.2f}%")
    st.write(f"**Decision:** {churn_label}")
    st.progress(stay_probability if churn_label == "WILL STAY" else churn_probability)
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption("The model uses the same preprocessing pipeline as the notebook: gender encoding, geography one-hot encoding, and standard scaling.")
