import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "house_price_model.pkl"
)

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found!")
    st.write("Expected location:")
    st.code(MODEL_PATH)
    st.stop()

model = joblib.load(MODEL_PATH)

# -----------------------------
# App Title
# -----------------------------
st.title("🏠 House Price Prediction")
st.write("Enter the house details below to predict its price.")

st.divider()

# -----------------------------
# User Inputs
# -----------------------------

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

bhk_no = st.number_input(
    "Number of BHK",
    min_value=1,
    max_value=20,
    value=2,
    step=1
)

bhk_or_rk = st.selectbox(
    "BHK or RK",
    ["BHK", "RK"]
)

square_ft = st.number_input(
    "Area (Square Feet)",
    min_value=100.0,
    max_value=100000.0,
    value=1000.0,
    step=50.0
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
    value=77.0,
    format="%.6f"
)

latitude = st.number_input(
    "Latitude",
    value=28.0,
    format="%.6f"
)

# -----------------------------
# Prediction
# -----------------------------

if st.button("🔮 Predict House Price"):

    # Create input DataFrame
    input_data = pd.DataFrame({
        "UNDER_CONSTRUCTION": [under_construction],
        "RERA": [rera],
        "BHK_NO.": [bhk_no],
        "SQUARE_FT": [square_ft],
        "READY_TO_MOVE": [ready_to_move],
        "RESALE": [resale],
        "LONGITUDE": [longitude],
        "LATITUDE": [latitude]
    })

    # Prediction
    try:
        prediction = model.predict(input_data)[0]

        st.success(
            f"🏠 Predicted House Price: ₹ {prediction:.2f} Lakhs"
        )

    except Exception as e:
        st.error("❌ Prediction failed.")
        st.code(str(e))