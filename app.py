import streamlit as st
import pandas as pd
import joblib

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.hero {
    padding: 35px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 30px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 8px;
}

.hero p {
    font-size: 18px;
    margin: 0;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 15px;
}

.prediction-box {
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    margin-top: 30px;
    background: linear-gradient(135deg, #11998e, #38ef7d);
}

.prediction-label {
    font-size: 18px;
    font-weight: 600;
}

.prediction-price {
    font-size: 42px;
    font-weight: 800;
    margin-top: 10px;
}

.info-card {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(128,128,128,0.3);
    text-align: center;
}

.footer {
    text-align: center;
    margin-top: 50px;
    padding: 20px;
    opacity: 0.7;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

try:
    model_data = joblib.load("house_price_model.pkl")

    model = model_data["model"]
    features = model_data["features"]

except Exception as e:
    st.error("❌ Could not load the model.")
    st.write(e)
    st.stop()

# =========================================================
# HERO SECTION
# =========================================================

st.markdown("""
<div class="hero">

<h1>🏠 House Price Predictor</h1>

<p>
AI-Powered Real Estate Price Prediction
</p>

</div>
""", unsafe_allow_html=True)

st.write(
    "Enter the property details below and our Machine Learning model "
    "will estimate the property's price."
)

# =========================================================
# PROPERTY INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">🏠 Property Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    posted_by = st.selectbox(
        "👤 Posted By",
        ["Owner", "Dealer", "Builder"]
    )

with col2:

    bhk = st.number_input(
        "🛏️ Number of BHK",
        min_value=1,
        max_value=10,
        value=2
    )

with col3:

    bhk_or_rk = st.selectbox(
        "🏢 Property Type",
        ["BHK", "RK"]
    )

col4, col5, col6 = st.columns(3)

with col4:

    square_ft = st.number_input(
        "📐 Area (Square Feet)",
        min_value=100.0,
        max_value=100000.0,
        value=1000.0,
        step=100.0
    )

with col5:

    under_construction = st.selectbox(
        "🏗️ Under Construction",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with col6:

    rera = st.selectbox(
        "📋 RERA Approved",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

# =========================================================
# PROPERTY STATUS
# =========================================================

st.markdown(
    '<div class="section-title">🏡 Property Status</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    ready_to_move = st.selectbox(
        "✅ Ready to Move",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with col2:

    resale = st.selectbox(
        "🔄 Resale Property",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

# =========================================================
# LOCATION
# =========================================================

st.markdown(
    '<div class="section-title">📍 Location</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    longitude = st.number_input(
        "🌐 Longitude",
        value=77.5946,
        format="%.6f"
    )

with col2:

    latitude = st.number_input(
        "🌐 Latitude",
        value=12.9716,
        format="%.6f"
    )

# =========================================================
# PREDICTION BUTTON
# =========================================================

st.write("")

predict_button = st.button(
    "🔮 Predict House Price",
    type="primary",
    use_container_width=True
)

# =========================================================
# PREDICTION
# =========================================================

if predict_button:

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

    # Add categorical features

    input_data["POSTED_BY_Dealer"] = (
        1 if posted_by == "Dealer" else 0
    )

    input_data["POSTED_BY_Owner"] = (
        1 if posted_by == "Owner" else 0
    )

    input_data["BHK_OR_RK_RK"] = (
        1 if bhk_or_rk == "RK" else 0
    )

    # Match training columns

    input_data = input_data.reindex(
        columns=features,
        fill_value=0
    )

    # Make prediction

    prediction = model.predict(input_data)[0]

    # =====================================================
    # RESULT
    # =====================================================

    st.markdown(
        f"""
        <div class="prediction-box">

        <div class="prediction-label">
        💰 Estimated Property Price
        </div>

        <div class="prediction-price">
        ₹ {prediction:,.2f} Lakhs
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.success(
        "✅ Prediction generated successfully using the Random Forest model."
    )

# =========================================================
# MODEL INFORMATION
# =========================================================

st.write("")

st.markdown(
    '<div class="section-title">🤖 Machine Learning Model</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        """
        <div class="info-card">

        <h3>🌲 Algorithm</h3>

        <p>Random Forest Regressor</p>

        </div>
        """,
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        """
        <div class="info-card">

        <h3>📊 Problem Type</h3>

        <p>Regression</p>

        </div>
        """,
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        """
        <div class="info-card">

        <h3>🎯 Target</h3>

        <p>House Price in Lakhs</p>

        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# ABOUT
# =========================================================

with st.expander("ℹ️ About this project"):

    st.write(
        """
        This House Price Prediction application uses a Machine Learning
        model to estimate property prices based on features such as:

        • Number of bedrooms  
        • Property area  
        • RERA approval  
        • Construction status  
        • Ready-to-move status  
        • Resale status  
        • Geographic location  

        The model was trained using a Random Forest Regressor.
        """
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">

    🏠 House Price Prediction Project  
    <br>
    Built with Python • Scikit-learn • Pandas • Streamlit

    </div>
    """,
    unsafe_allow_html=True
)