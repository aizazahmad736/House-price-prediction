import streamlit as st
import joblib

# Load model
model = joblib.load("house_price_model.pkl")

st.title("🏠 House Price Prediction")

st.write("Enter house details to predict the price")

st.success("Model loaded successfully!")

