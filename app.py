import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load saved model and scaler
model = joblib.load("lightgbm_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Concrete Strength Predictor", layout="centered")
st.title("🔨 Concrete Compressive Strength Predictor")

# About Section
st.markdown("""
### 🧱 About This App
This appplication predicts the **compressive strength of concrete** (in MPa) using a trained LightGBM machine learning model.  
Input the mix proportions of your concrete design, and get a strength estimate based on historical data.
""")

st.markdown("Enter your concrete mix details:")

# Inputs
cement = st.number_input("Cement (kg/m³)", 0.0, 1000.0, 150.0)
slag = st.number_input("Blast_Furnace_Slag (kg/m³)", 0.0, 300.0, 0.0)
Fly_Ash = st.number_input("Fly_Ash (kg/m³)", 0.0, 300.0, 0.0)
water = st.number_input("Water (kg/m³)", 0.0, 300.0, 150.0)
superplasticizer = st.number_input("Superplasticizer (kg/m³)", 0.0, 50.0, 5.0)
coarse_agg = st.number_input("Coarse_Aggregate (kg/m³)", 0.0, 1200.0, 900.0)
fine_agg = st.number_input("Fine_Aggregate (kg/m³)", 0.0, 1000.0, 800.0)
age = st.number_input("Age (days)", 1, 365, 28)

if st.button("Predict Strength"):
    df = pd.DataFrame([[cement, slag, Fly_Ash, water, superplasticizer,
                    coarse_agg, fine_agg, age]],
                  columns=["Cement", "Blast_Furnace_Slag", "Fly_Ash", "Water",
                           "Superplasticizer", "Coarse_Aggregate", "Fine_Aggregate", "Age"])

    scaled = scaler.transform(df)
    prediction = model.predict(scaled)[0]
    st.success(f"Predicted Compressive Strength: {prediction:.2f} MPa")

# Footer
st.markdown("---")
st.caption("👷‍♂️ Developed by Kiran Subedi | Website: https://kiransubedi545.com.np/ | Email: Kiransubedi545@gmail.com")

