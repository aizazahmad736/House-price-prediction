import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load model and feature names
# -----------------------------

model = joblib.load("house_price_model.pkl")
model_features = joblib.load("model_features.pkl")

# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction")
st.write("Enter the house details below to predict its price.")

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

bhk = st.number_input(
    "Number of BHK",
    min_value=1,
    max_value=20,
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
    value=77.5
)

latitude = st.number_input(
    "Latitude",
    value=12.9
)

city = st.text_input(
    "City",
    value="Bangalore"
)

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Price"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "POSTED_BY": [posted_by],
        "UNDER_CONSTRUCTION": [under_construction],
        "RERA": [rera],
        "BHK_NO.": [bhk],
        "BHK_OR_RK": [bhk_or_rk],
        "SQUARE_FT": [square_ft],
        "READY_TO_MOVE": [ready_to_move],
        "RESALE": [resale],
        "LONGITUDE": [longitude],
        "LATITUDE": [latitude],
        "CITY": [city]
    })

    # One-hot encode categorical columns
    input_data = pd.get_dummies(
        input_data,
        columns=["POSTED_BY", "BHK_OR_RK", "CITY"],
        drop_first=True
    )

    # Make sure input has exactly the same features as training data
    input_data = input_data.reindex(
        columns=model_features,
        fill_value=0
    )

    # Prediction
    prediction = model.predict(input_data)[0]

    st.success(
        f"🏠 Estimated House Price: {prediction:.2f} Lacs"
    )