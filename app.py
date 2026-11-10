import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configure Page Layout
st.set_page_config(
    page_title="Store Sales Prediction App", page_icon="📊", layout="wide"
)

# Load Trained Pipeline
@st.cache_resource
def load_pipeline():
    pipeline_path = "model_pipeline.joblib"
    if not os.path.exists(pipeline_path):
        return None
    return joblib.load(pipeline_path)

pipeline = load_pipeline()

# Title and Description
st.title("📊 Store Sales Predictive Modeling Application")
st.markdown(
    """
This web application provides real-time sales revenue prediction and high-sales probability classification
based on historical store transactions, economic indicators, and promotional data.
"""
)

if pipeline is None:
    st.warning(
        "⚠️ Model pipeline not found. Please execute `python train_models.py` in your local directory to generate `model_pipeline.joblib`."
    )
    st.stop()

# Extract attributes from pipeline
reg_model = pipeline["reg_model"]
clf_model = pipeline["clf_model"]
feature_names = pipeline["feature_names"]
store_levels = pipeline["store_levels"]
family_levels = pipeline["family_levels"]
metrics = pipeline["metrics"]

# Sidebar Navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Choose Mode:",
    ["Single Scenario Predictor", "Batch Dataset Prediction", "Model Analytics & Metrics"],
)

if page == "Single Scenario Predictor":
    st.header("🎛️ Single Scenario Predictor")
    st.markdown("Specify store parameters to evaluate expected daily sales.")

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_store = st.selectbox(
            "Store Number (store_nbr)", options=store_levels, index=0
        )
        selected_family = st.selectbox(
            "Product Family (family)", options=family_levels, index=3
        )
        onpromotion = st.number_input(
            "On-Promotion Count", min_value=0, max_value=1000, value=15, step=1
        )

    with col2:
        oil_price = st.number_input(
            "Daily Oil Price ($)", min_value=0.0, max_value=200.0, value=53.75, step=0.5
        )
        month = st.slider("Month", min_value=1, max_value=12, value=5)
        day_of_week = st.selectbox(
            "Day of Week (1=Mon, 7=Sun)", options=[1, 2, 3, 4, 5, 6, 7], index=4
        )

    with col3:
        day_of_month = st.slider("Day of Month", min_value=1, max_value=31, value=15)
        is_payday = st.radio("Is Payday?", options=[0, 1], index=0, horizontal=True)
        promo_flag = st.radio(
            "Promotion Active Flag?", options=[0, 1], index=1, horizontal=True
        )

    # Construct single observation dataframe
    input_dict = {
        "store_nbr": [str(selected_store)],
        "family": [str(selected_family)],
        "onpromotion": [onpromotion],
        "oil_price": [oil_price],
        "month": [month],
        "day_of_week": [day_of_week],
        "day_of_month": [day_of_month],
        "is_payday": [is_payday],
        "promo_flag": [promo_flag],
    }

    input_df = pd.DataFrame(input_dict)

    # Encode inputs according to training features
    encoded_df = pd.get_dummies(input_df, columns=["store_nbr", "family"])
    for col in feature_names:
        if col not in encoded_df.columns:
            encoded_df[col] = 0
    encoded_df = encoded_df[feature_names]

    if st.button("🚀 Predict Sales"):
        predicted_sales = reg_model.predict(encoded_df)[0]
        predicted_sales = max(0.0, float(predicted_sales))  # Handle non-negative bounds

        clf_prob = clf_model.predict_proba(encoded_df)[0][1]
        is_high_sales = "YES 🟢" if clf_prob >= 0.5 else "NO 🔴"

        st.markdown("---")
        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.metric(
                label="Predicted Sales Revenue",
                value=f"${predicted_sales:,.2f}",
            )

        with res_col2:
            st.metric(
                label="High Sales Probability",
                value=f"{clf_prob * 100:.1f}%",
            )

        with res_col3:
            st.metric(
                label="High Sales Target (> Median)",
                value=is_high_sales,
            )

elif page == "Batch Dataset Prediction":
    st.header("📁 Batch Prediction via CSV Upload")
    st.write(
        "Upload a `.csv` dataset containing the required features (`store_nbr`, `family`, `onpromotion`, `oil_price`, `month`, `day_of_week`, `day_of_month`, `is_payday`, `promo_flag`)."
    )

    uploaded_file = st.file_uploader("Upload Input CSV", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded File:")
        st.dataframe(batch_df.head(5))

        req_cols = [
            "store_nbr",
            "family",
            "onpromotion",
            "oil_price",
            "month",
            "day_of_week",
            "day_of_month",
            "is_payday",
            "promo_flag",
        ]
        missing_cols = [c for c in req_cols if c not in batch_df.columns]

        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            batch_raw = batch_df[req_cols].copy()
            batch_raw["store_nbr"] = batch_raw["store_nbr"].astype(str)
            batch_raw["family"] = batch_raw["family"].astype(str)

            encoded_batch = pd.get_dummies(
                batch_raw, columns=["store_nbr", "family"]
            )
            for col in feature_names:
                if col not in encoded_batch.columns:
                    encoded_batch[col] = 0
            encoded_batch = encoded_batch[feature_names]

            batch_df["predicted_sales"] = np.maximum(
                0.0, reg_model.predict(encoded_batch)
            )
            batch_df["high_sales_prob"] = clf_model.predict_proba(encoded_batch)[
                :, 1
            ]
            batch_df["predicted_high_sales"] = (
                batch_df["high_sales_prob"] >= 0.5
            ).astype(int)

            st.success("Predictions complete!")
            st.write("Results Output:")
            st.dataframe(batch_df.head(10))

            csv_data = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv_data,
                file_name="store_sales_predictions.csv",
                mime="text/csv",
            )

elif page == "Model Analytics & Metrics":
    st.header("📈 SAS & Python Benchmark Performance Analytics")
    st.markdown(
        "Model statistics based on Validation ASE and ROC metrics:"
    )

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.metric("Regression R² Score", f"{metrics['reg_r2']:.4f}")
    with m_col2:
        st.metric("Validation ASE", f"{metrics['reg_ase']:,.2f}")
    with m_col3:
        st.metric("Classification Accuracy", f"{metrics['clf_acc']*100:.2f}%")
    with m_col4:
        st.metric("ROC AUC Score", f"{metrics['clf_auc']:.4f}")

    st.markdown("---")
    st.subheader("Model Selection Summary Reference")
    summary_data = {
        "Model / Algorithm": [
            "Stepwise Regression (Selected Model)",
            "Forward Selection Regression",
            "Backward Elimination Regression",
            "Adaptive LASSO Regression",
            "Random Forest Classifier (Champion)",
            "Gradient Boosting Classifier",
        ],
        "Task": [
            "Sales Revenue",
            "Sales Revenue",
            "Sales Revenue",
            "Sales Revenue",
            "High Sales (Binary)",
            "High Sales (Binary)",
        ],
        "R² / Accuracy": [
            "0.6641",
            "0.6680",
            "0.6675",
            "0.6574",
            "70.79%",
            "70.09%",
        ],
        "Validation Error / ROC AUC": [
            "626,521.78 ASE",
            "627,992.14 ASE",
            "653,809.26 ASE",
            "625,409.55 ASE",
            "0.7649 AUC",
            "0.7541 AUC",
        ],
    }
    st.table(pd.DataFrame(summary_data))
