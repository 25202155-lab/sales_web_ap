import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Sales Prediction Engine",
    page_icon="📈",
    layout="wide"
)

# ---------------------------------------------------------
# SAS MODEL PARAMETERS (Stepwise Model at Step 9)
# Source: SAS Cross-Validation & Parameter Estimates Output
# ---------------------------------------------------------
INTERCEPT = -268.601934

# Continuous Covariates
COEFF_ONPROMOTION = 12.871090
COEFF_OIL_PRICE = 3.587544

# Store Number Dummy Coefficients (Reference: store_nbr 54 = 0.0)
STORE_COEFFS = {
    1: 73.357222, 2: 213.042648, 3: 891.330605, 4: 154.492476, 5: 44.400551,
    6: 302.740196, 7: 372.480672, 8: 437.650828, 9: 293.028019, 10: -88.446240,
    11: 331.286430, 12: -37.273803, 13: -61.896079, 14: -27.248492, 15: -38.960842,
    16: -93.073557, 17: 171.619022, 18: 42.829299, 19: 7.581982, 20: 184.388160,
    21: 117.564533, 22: -84.619525, 23: 5.887199, 24: 282.741582, 25: 44.277343,
    26: -69.591556, 27: 208.010969, 28: 171.231447, 29: 82.443328, 30: -100.174761,
    31: 120.035085, 32: -107.321775, 33: -9.371910, 34: 59.505470, 35: -109.782995,
    36: 72.629399, 37: 164.107879, 38: 144.260689, 39: 210.583886, 40: 203.278699,
    41: 141.755350, 42: 148.651545, 43: 159.932760, 44: 1170.799586, 45: 1052.021836,
    46: 711.107797, 47: 947.690873, 48: 554.807220, 49: 836.457442, 50: 397.606806,
    51: 453.880522, 52: 112.090565, 53: 100.453775, 54: 0.0
}

# Product Family Dummy Coefficients (Reference: SEAFOOD = 0.0)
FAMILY_COEFFS = {
    "AUTOMOTIVE": -27.201648, "BABY CARE": 46.158884, "BEAUTY": -12.581279,
    "BEVERAGES": 3296.533147, "BOOKS": -195.224661, "BREAD/BAKERY": 452.633487,
    "CELEBRATION": -20.562827, "CLEANING": 999.681490, "DAIRY": 613.824745,
    "DELI": 168.057817, "EGGS": 154.463648, "FROZEN FOODS": 114.827496,
    "GROCERY I": 4035.775062, "GROCERY II": -2.373846, "HARDWARE": -37.532479,
    "HOME AND KITCHEN I": 14.151550, "HOME AND KITCHEN II": 17.756280,
    "HOME APPLIANCES": -45.426781, "HOME CARE": 215.393033, "LADIESWEAR": -64.321346,
    "LAWN AND GARDEN": -103.383029, "LINGERIE": -27.941069, "LIQUOR, WINE, BEER": 78.848012,
    "MAGAZINES": -28.170132, "MEATS": 277.029774, "PERSONAL CARE": 240.021656,
    "PET SUPPLIES": -23.046066, "PLAYERS AND ELECTRONICS": -24.960516,
    "POULTRY": 317.802070, "PREPARED FOODS": 78.523791, "PRODUCE": 2079.378466,
    "SCHOOL AND OFFICE SUPPLIES": -23.350673, "SEAFOOD": 0.0
}

# Month Dummy Coefficients (Reference: month 8 = 0.0)
MONTH_COEFFS = {
    1: 11.860346, 2: -0.224802, 3: 40.437655, 4: 7.629505,
    5: 26.811286, 6: 32.738595, 7: 21.471854, 8: 0.0
}

# Day of Week Dummy Coefficients (Reference: day_of_week 7 = 0.0)
DAY_OF_WEEK_COEFFS = {
    1: -121.564014, 2: -180.633605, 3: -211.956363, 4: -218.475178,
    5: -185.984887, 6: -4.542544, 7: 0.0
}

# Day of Month Dummy Coefficients (Reference: day_of_month 31 = 0.0)
DAY_OF_MONTH_COEFFS = {
    1: 73.717530, 2: 141.698482, 3: 84.035684, 4: 78.456031, 5: 69.559706,
    6: 9.217216, 7: 24.271951, 8: 9.164725, 9: 22.494875, 10: 3.768479,
    11: 13.170775, 12: -8.882117, 13: -4.937352, 14: -27.566329, 15: -44.574212,
    16: 13.931295, 17: 2.786591, 18: 1.892655, 19: 7.559669, 20: 8.831345,
    21: 4.744837, 22: 2.540212, 23: 16.254839, 24: -2.763054, 25: -6.035407,
    26: -11.412916, 27: -27.161887, 28: 11.808146, 29: -6.405078, 30: -21.979452,
    31: 0.0
}

# Indicator Flag Dummy Coefficients (Reference Level: 1 = 0.0)
IS_PAYDAY_COEFFS = {0: -52.733539, 1: 0.0}
PROMO_FLAG_COEFFS = {0: 64.156475, 1: 0.0}


def calculate_prediction(inputs: dict) -> float:
    """Calculates regression raw output and truncates at zero."""
    raw_pred = (
        INTERCEPT
        + (inputs["onpromotion"] * COEFF_ONPROMOTION)
        + (inputs["oil_price"] * COEFF_OIL_PRICE)
        + STORE_COEFFS[inputs["store_nbr"]]
        + FAMILY_COEFFS[inputs["family"]]
        + MONTH_COEFFS[inputs["month"]]
        + DAY_OF_WEEK_COEFFS[inputs["day_of_week"]]
        + DAY_OF_MONTH_COEFFS[inputs["day_of_month"]]
        + IS_PAYDAY_COEFFS[inputs["is_payday"]]
        + PROMO_FLAG_COEFFS[inputs["promo_flag"]]
    )
    return max(0.0, raw_pred)


# ---------------------------------------------------------
# APPLICATION INTERFACE
# ---------------------------------------------------------
st.title("📈 Retail Sales Prediction Engine")
st.markdown(
    """
    This deployment application estimates sales based on store attributes, product metadata, 
    and macroeconomic temporal factors using the validated **Stepwise Regression Model** 
    ($R^2 = 0.6655$, Validation ASE = $617,048.48$).
    """
)

tab1, tab2 = st.tabs(["🎯 Single Prediction", "📂 Batch File Prediction (CSV)"])

# ---------------------------------------------------------
# TAB 1: SINGLE PREDICTION INTERFACE
# ---------------------------------------------------------
with tab1:
    st.sidebar.header("Input Parameters")

    # Group 1: Entity Identifiers
    st.sidebar.subheader("Store & Product")
    family_input = st.sidebar.selectbox("Product Family", sorted(list(FAMILY_COEFFS.keys())))
    store_input = st.sidebar.selectbox("Store Number", sorted(list(STORE_COEFFS.keys())))

    # Group 2: Operational & Economic Predictors
    st.sidebar.subheader("Operations & Macroeconomics")
    onpromotion_input = st.sidebar.number_input("Items on Promotion", min_value=0, value=15, step=1)
    oil_price_input = st.sidebar.number_input("Oil Price ($)", min_value=0.0, value=48.50, step=0.50)

    # Group 3: Temporal Factors
    st.sidebar.subheader("Calendar Factors")
    month_input = st.sidebar.slider("Month", min_value=1, max_value=8, value=3)
    day_of_week_input = st.sidebar.selectbox(
        "Day of Week", 
        options=[1, 2, 3, 4, 5, 6, 7],
        format_func=lambda x: {1: "Monday (1)", 2: "Tuesday (2)", 3: "Wednesday (3)", 
                               4: "Thursday (4)", 5: "Friday (5)", 6: "Saturday (6)", 
                               7: "Sunday (7)"}[x]
    )
    day_of_month_input = st.sidebar.slider("Day of Month", min_value=1, max_value=31, value=15)

    # Group 4: Binary Flags
    st.sidebar.subheader("Event Flags")
    is_payday_input = st.sidebar.radio("Is Payday Event?", options=[0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
    promo_flag_input = st.sidebar.radio("Promotion Flag Active?", options=[0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")

    # Execute Prediction
    input_data = {
        "family": family_input,
        "store_nbr": store_input,
        "onpromotion": onpromotion_input,
        "oil_price": oil_price_input,
        "month": month_input,
        "day_of_week": day_of_week_input,
        "day_of_month": day_of_month_input,
        "is_payday": is_payday_input,
        "promo_flag": promo_flag_input
    }

    predicted_sales = calculate_prediction(input_data)

    # Display Metrics
    col_metric1, col_metric2, col_metric3 = st.columns(3)
    with col_metric1:
        st.metric(label="Predicted Sales Unit Output", value=f"{predicted_sales:,.2f}")
    with col_metric2:
        st.metric(label="Selected Model R-Square", value="0.6655")
    with col_metric3:
        st.metric(label="Validation ASE", value="617,048.48")

    st.divider()

    # Visual Factor Breakdown
    st.subheader("Model Coefficient Factor Breakdown")
    
    breakdown_data = [
        {"Factor": "Baseline Intercept", "Impact": INTERCEPT},
        {"Factor": f"Product Family ({family_input})", "Impact": FAMILY_COEFFS[family_input]},
        {"Factor": f"Store ({store_input})", "Impact": STORE_COEFFS[store_input]},
        {"Factor": f"Promotions ({onpromotion_input} units)", "Impact": onpromotion_input * COEFF_ONPROMOTION},
        {"Factor": f"Oil Price (${oil_price_input:.2f})", "Impact": oil_price_input * COEFF_OIL_PRICE},
        {"Factor": f"Month ({month_input})", "Impact": MONTH_COEFFS[month_input]},
        {"Factor": f"Day of Week ({day_of_week_input})", "Impact": DAY_OF_WEEK_COEFFS[day_of_week_input]},
        {"Factor": f"Day of Month ({day_of_month_input})", "Impact": DAY_OF_MONTH_COEFFS[day_of_month_input]},
        {"Factor": f"Payday Flag ({is_payday_input})", "Impact": IS_PAYDAY_COEFFS[is_payday_input]},
        {"Factor": f"Promo Flag ({promo_flag_input})", "Impact": PROMO_FLAG_COEFFS[promo_flag_input]},
    ]
    
    df_breakdown = pd.DataFrame(breakdown_data)
    
    col_chart, col_table = st.columns([3, 2])
    with col_chart:
        fig = px.bar(
            df_breakdown, 
            x="Impact", 
            y="Factor", 
            orientation="h",
            title="Individual Feature Contributions to Predicted Value",
            color="Impact",
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig, use_container_width=True)
    with col_table:
        st.dataframe(df_breakdown.style.format({"Impact": "{:,.2f}"}), use_container_width=True)


# ---------------------------------------------------------
# TAB 2: BATCH PREDICTION INTERFACE
# ---------------------------------------------------------
with tab2:
    st.subheader("Upload Batch Dataset")
    st.write(
        "Upload a `.csv` file containing the 9 target features: "
        "`store_nbr`, `family`, `onpromotion`, `oil_price`, `month`, `day_of_week`, `day_of_month`, `is_payday`, `promo_flag`."
    )
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            required_cols = {"store_nbr", "family", "onpromotion", "oil_price", "month", "day_of_week", "day_of_month", "is_payday", "promo_flag"}

            if not required_cols.issubset(df_upload.columns):
                st.error(f"Missing required columns in CSV! Ensure the file contains: {required_cols}")
            else:
                preds = []
                for _, row in df_upload.iterrows():
                    row_dict = {
                        "family": str(row["family"]).strip(),
                        "store_nbr": int(row["store_nbr"]),
                        "onpromotion": float(row["onpromotion"]),
                        "oil_price": float(row["oil_price"]),
                        "month": int(row["month"]),
                        "day_of_week": int(row["day_of_week"]),
                        "day_of_month": int(row["day_of_month"]),
                        "is_payday": int(row["is_payday"]),
                        "promo_flag": int(row["promo_flag"]),
                    }
                    preds.append(calculate_prediction(row_dict))

                df_upload["Predicted_Sales"] = preds
                st.success("Batch predictions calculated successfully!")
                st.dataframe(df_upload.head(100), use_container_width=True)

                # Download Output Button
                csv_bytes = df_upload.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Predicted Sales CSV",
                    data=csv_bytes,
                    file_name="sales_predictions_output.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"An error occurred while parsing the file: {e}")
