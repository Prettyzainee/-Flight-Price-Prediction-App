# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# -----------------------------
# Load Model & Preprocessor
# -----------------------------
@st.cache_resource
def load_resources():
    model = load_model("flight_price_model.h5", compile=False)  # Don't try to load old metrics
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_resources()

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    h1, h2, h3, h4 { color: #1f4e79; }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# App Title
# -----------------------------
st.title("✈️ Flight Price Prediction App")
st.write("Enter your flight details below and get an instant price prediction.")

# -----------------------------
# Input Form
# -----------------------------
with st.form("flight_form"):
    col1, col2 = st.columns(2)

    airline = col1.selectbox("Airline", [
        "SpiceJet", "AirAsia", "Vistara", "Indigo", "GoAir", "Air India", "missing"
    ])

    source_city = col1.selectbox("Source City", [
        "Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai", "missing"
    ])

    destination_city = col2.selectbox("Destination City", [
        "Mumbai", "Delhi", "Bangalore", "Kolkata", "Hyderabad", "Chennai", "missing"
    ])

    departure_time = col1.selectbox("Departure Time", [
        "Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night", "missing"
    ])

    arrival_time = col2.selectbox("Arrival Time", [
        "Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night", "missing"
    ])

    stops = col1.selectbox("Stops", ["zero", "one", "two_or_more", "missing"])

    travel_class = col2.selectbox("Class", ["Economy", "Business", "missing"])

    duration = st.slider("Duration (hours)", min_value=0.5, max_value=30.0, step=0.1, value=2.5)
    days_left = st.slider("Days Left until Departure", min_value=0, max_value=365, step=1, value=30)

    submitted = st.form_submit_button("🔍 Predict Price")

# -----------------------------
# Prediction Logic
# -----------------------------
if submitted:
    # Build input DataFrame
    input_df = pd.DataFrame([{
        "airline": airline,
        "source_city": source_city,
        "departure_time": departure_time,
        "stops": stops,
        "arrival_time": arrival_time,
        "destination_city": destination_city,
        "class": travel_class,
        "duration": duration,
        "days_left": days_left
    }])

    # Preprocess
    input_transformed = preprocessor.transform(input_df)

    # Predict
    prediction = model.predict(input_transformed)
    predicted_price = float(prediction[0][0])

    # Display result
    st.success(f"💰 Estimated Flight Price: **₦{predicted_price:,.2f}**")
    st.caption("Note: This is a machine learning prediction and may differ from actual ticket prices.")
