# import streamlit as st
# import joblib

# # Load model
# model = joblib.load("house_price_model.pkl")

# st.title("🏠 House Price Prediction")

# st.write("Enter house details to predict the price")

# st.success("Model loaded successfully!")

import joblib

# Load trained model
model = joblib.load("house_price_model.pkl")


def predict_price(features):
    prediction = model.predict([features])
    return prediction[0]


print("Model loaded successfully")