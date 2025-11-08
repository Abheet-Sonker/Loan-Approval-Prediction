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
st.set_page_config(page_title="Loan Prediction System", layout="centered")
st.title("💰 Loan Acceptance Prediction System (Created by Abheet)")
st.write("Enter your details below to check the likelihood of accepting a personal loan offer.")

# ===========================
# User Inputs
# ===========================
Income = st.number_input("Annual Income (in ₹1000s)", min_value=0, max_value=10**9, value=0)
Family = st.number_input("Family Size (in Number)", min_value=0, max_value=20, value=1)
Education = st.selectbox("Education Level", ["1 - 12th", "2 - UG", "3 - PG"])
Mortgage = st.number_input("Mortgage Amount (in ₹1000s)", min_value=0, max_value=10**6, value=0)
CDAccount = st.selectbox("Have CD Account?", ["0 - No", "1 - Yes"])

# Convert categorical input
Education = int(Education.split(" - ")[0])
CDAccount = int(CDAccount.split(" - ")[0])

# ===========================
# Prediction
# ===========================
if st.button("🔍 Predict Loan Acceptance"):
    # Prepare input array
    input_features = np.array([[Income, Family, Education, Mortgage, CDAccount]])

    # Scale input using the saved scaler
    scaled_features = scaler.transform(input_features)

    # Predict
    prediction = model.predict(scaled_features)[0]
    prediction_prob = model.predict_proba(scaled_features)[0]
    probability_of_acceptance = prediction_prob[1] * 100

    # Display results
    if prediction == 1:
        st.success(f"✅ **Yes**, you are likely to accept the loan offer.")
        st.write(f"📈 Probability of acceptance: **{probability_of_acceptance:.2f}%**")
    else:
        st.error(f"❌ **No**, you are unlikely to accept the loan offer.")
        st.write(f"📉 Probability of acceptance: **{probability_of_acceptance:.2f}%**")
