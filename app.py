import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load saved model and scaler
model = joblib.load("lightgbm_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Concrete Strength Predictor", layout="centered")
st.title("🔨 Concrete Compressive Strength Predictor")

# Inject custom tab styling
st.markdown("""
<style>
/* Highlight active tab */
[data-baseweb="tab"]:nth-child(1)[aria-selected="true"] {
    background-color: #e6f7ff;
    border-radius: 10px 10px 0 0;
}
[data-baseweb="tab"]:nth-child(2)[aria-selected="true"] {
    background-color: #fff0f5;
    border-radius: 10px 10px 0 0;
}
</style>
""", unsafe_allow_html=True)

# Create tabs
tab1, tab2 = st.tabs(["🔍 Compressive Strength Prediction", "🧮 Compressive Strength Calculation"])

# --- Tab 1: ML-Based Prediction ---
with tab1:
    st.markdown(""" 
### 🧱 About This App
This app predicts the **compressive strength of concrete** (in MPa) using a trained LightGBM machine learning model.  
Input the mix proportions of your concrete design, and get a strength estimate based on historical data.
""")

st.markdown("**Enter your concrete mix details:**")

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

# Footer in Tab 1
st.markdown("---")
st.caption("👷‍♂️ Developed by Kiran Subedi | Website: https://kiransubedi545.com.np/ | Email: Kiransubedi545@gmail.com")


# --- Tab 2: Manual Load-Based Calculation ---
with tab2:
    st.markdown("""
    ### 🧮 Compressive Strength by Load and Area
   This app Calculates the **compressive strength of concrete** (in MPa) using applied load and cube dimensions.  
Input the Applied load of on concrete during test, and get Compressive strength of concrete.
    """)

    load = st.number_input("Applied Load (kN)", min_value=0.0, value=100.0)
    side_length = st.number_input("Cube Side Length (mm)", min_value=50.0, value=150.0)

    if side_length > 0:
        area_mm2 = side_length ** 2
        area_m2 = area_mm2 / 1_000_000  # convert to m²
        strength_calc = (load * 1000) / area_mm2  # MPa = N/mm²

        st.write(f"📐 Cube Area: {area_mm2:.0f} mm² ({area_m2:.4f} m²)")
        st.success(f"Calculated Compressive Strength: {strength_calc:.2f} MPa")

# Footer in Tab 2
st.markdown("---")
st.caption("👷‍♂️ Developed by Kiran Subedi | Website: https://kiransubedi545.com.np/ | Email: Kiransubedi545@gmail.com")

