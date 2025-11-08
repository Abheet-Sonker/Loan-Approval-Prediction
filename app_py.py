# -*- coding: utf-8 -*-
"""app.py"""

import streamlit as st
import pickle
import numpy as np
import joblib

# ===========================
# Load model and scaler
# ===========================
model = joblib.load(open('Loan_acceptance_model.pkl', 'rb'))
scaler = joblib.load(open('Loan_acceptance_model_scaler.pkl', 'rb'))

# ===========================
# Streamlit App UI
# ===========================
st.set_page_config(page_title="Loan Acceptance Prediction System", layout="centered")
st.title("💰 Loan Acceptance Prediction System (Created by Abheet)")
st.write("Enter your details below to check the likelihood of accepting a personal loan offer.")

# ===========================
# User Inputs
# ===========================
Income = st.number_input("Annual Income (in ₹1000s)", min_value=0, max_value=10**9, value=0)
Family = st.number_input("Family Members (Number)", min_value=1, max_value=20, value=1)
CCAvg = st.number_input("Average Monthly Credit Card Spend (in ₹1000s)", min_value=0.0, max_value=1000.0, value=0.0)
Education = st.selectbox("Education Level", ["1 - 12th", "2 - UG", "3 - PG"])
Mortgage = st.number_input("Mortgage Amount (in ₹1000s)", min_value=0, max_value=10**6, value=0)
CDAccount = st.selectbox("Have Certificate of Deposit (CD) Account?", ["0 - No", "1 - Yes"])

# Convert categorical inputs
Education = int(Education.split(" - ")[0])
CDAccount = int(CDAccount.split(" - ")[0])

# ===========================
# Prediction
# ===========================
if st.button("🔍 Predict Loan Acceptance"):
    # Arrange inputs in correct feature order
    input_features = np.array([[Income, Family, CCAvg, Education, Mortgage, CDAccount]])

    # Scale input
    scaled_features = scaler.transform(input_features)

    # Model prediction
    prediction = model.predict(scaled_features)[0]
    prediction_prob = model.predict_proba(scaled_features)[0]
    probability_of_acceptance = prediction_prob[1] * 100

    # Display result
    if prediction == 1:
        st.success(f"✅ **Yes**, you are likely to accept the loan offer.")
        st.write(f"📈 Probability of acceptance: **{probability_of_acceptance:.2f}%**")
    else:
        st.error(f"❌ **No**, you are unlikely to accept the loan offer.")
        st.write(f"📉 Probability of acceptance: **{probability_of_acceptance:.2f}%**")
