import streamlit as st
import pandas as pd
import joblib

# ==============================
# Load model
# ==============================

model_data = joblib.load("house_price_model.pkl")

model = model_data["model"]
features = model_data["features"]

# ==============================
# Page configuration
# ==============================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction")
st.write("Enter the house details below to predict its price.")

st.divider()

# ==============================
# User inputs
# ==============================

posted_by = st.selectbox(
    "Posted By",
    ["Owner", "Dealer", "Builder"]
)

under_construction = st.selectbox(
    "Under Construction",
    [0, 1]
)

rera = st.selectbox(
    "RERA Approved",
    [0, 1]
)

bhk = st.number_input(
    "Number of BHK",
    min_value=1,
    max_value=10,
    value=2
)

bhk_or_rk = st.selectbox(
    "BHK or RK",
    ["BHK", "RK"]
)

square_ft = st.number_input(
    "Square Feet",
    min_value=100.0,
    max_value=100000.0,
    value=1000.0
)

ready_to_move = st.selectbox(
    "Ready to Move",
    [0, 1]
)

resale = st.selectbox(
    "Resale",
    [0, 1]
)

longitude = st.number_input(
    "Longitude",
    value=77.5946
)

latitude = st.number_input(
    "Latitude",
    value=12.9716
)

# ==============================
# Create input dataframe
# ==============================

input_data = pd.DataFrame({
    "UNDER_CONSTRUCTION": [under_construction],
    "RERA": [rera],
    "BHK_NO.": [bhk],
    "SQUARE_FT": [square_ft],
    "READY_TO_MOVE": [ready_to_move],
    "RESALE": [resale],
    "LONGITUDE": [longitude],
    "LATITUDE": [latitude],
})

# Add categorical columns
input_data["POSTED_BY_Dealer"] = 1 if posted_by == "Dealer" else 0
input_data["POSTED_BY_Owner"] = 1 if posted_by == "Owner" else 0
input_data["BHK_OR_RK_RK"] = 1 if bhk_or_rk == "RK" else 0

# ==============================
# Match training features
# ==============================

input_data = input_data.reindex(
    columns=features,
    fill_value=0
)

# ==============================
# Prediction
# ==============================

if st.button("Predict House Price", type="primary"):

    prediction = model.predict(input_data)[0]

    st.success(
        f"🏠 Estimated House Price: ₹ {prediction:,.2f} Lakhs"
    )