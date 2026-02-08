import streamlit as st
# Force reboot

import pandas as pd
import numpy as np
import pickle

# Set page config
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# Load the model and scaler
@st.cache_resource
def load_models():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_models()

# UI Design
st.title("❤️ Heart Disease Prediction App")
st.markdown("""
This app predicts the **10-year risk of coronary heart disease (CHD)** based on patient data from the Framingham Heart Study.
""")

if model is None:
    st.error("Model files not found! Please run `model.py` first to generate `model.pkl` and `scaler.pkl`.")
    st.info("To generate the files, run: `python3 model.py` (ensure you have scikit-learn installed)")
else:
    with st.sidebar:
        st.header("Patient Information")
        
        # Demographic
        st.subheader("Demographic")
        male = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        age = st.number_input("Age", min_value=1, max_value=120, value=40)
        
        # Behavioral
        st.subheader("Behavioral")
        current_smoker = st.selectbox("Is currently smoking?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        cigs_per_day = st.number_input("Cigarettes per day", min_value=0, max_value=100, value=0)
        
        # Medical History
        st.subheader("Medical History")
        bp_meds = st.selectbox("On BP Medication?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        prevalent_stroke = st.selectbox("Previously had a stroke?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        prevalent_hyp = st.selectbox("Currently Hypertensive?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        diabetes = st.selectbox("Has Diabetes?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Current Vitals")
        tot_chol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=700, value=200)
        sys_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=80, max_value=300, value=120)
        dia_bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=40, max_value=150, value=80)

    with col2:
        st.subheader("Physical Body Measures")
        bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        heart_rate = st.number_input("Heart Rate", min_value=40, max_value=200, value=75)
        glucose = st.number_input("Glucose level", min_value=40, max_value=400, value=80)

    # Prediction Logic
    if st.button("Predict Risk", use_container_width=True):
        # Prepare input data
        input_data = np.array([[male, age, current_smoker, cigs_per_day, bp_meds,
                                prevalent_stroke, prevalent_hyp, diabetes, tot_chol,
                                sys_bp, dia_bp, bmi, heart_rate, glucose]])
        
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0][1]

        st.divider()
        if prediction[0] == 1:
            st.error(f"⚠️ High Risk! The patient has a {probability:.1%} chance of developing CHD in the next 10 years.")
        else:
            st.success(f"✅ Low Risk. The patient has a {probability:.1%} chance of developing CHD in the next 10 years.")

st.markdown("---")
st.caption("Disclaimer: This is for educational purposes only and not for medical diagnosis.")
