import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model
model = joblib.load("models/fraud_model.pkl")

st.set_page_config(page_title="Fraud Detection System", layout="centered")

st.title("💳 Credit Card Fraud Detection System")
st.write("Enter transaction details to check fraud risk")

st.divider()

# Input fields (V1 to V28)
features = []

for i in range(1, 29):
    features.append(st.number_input(f"V{i}", value=0.0))

time = st.number_input("Time", value=0.0)
amount = st.number_input("Amount", value=0.0)

input_data = [time, amount] + features
input_df = pd.DataFrame([input_data])

st.divider()

if st.button("Check Fraud"):
    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    st.subheader("Result")

    st.progress(int(prob * 100))

    if pred == 1:
        st.error(f"⚠ FRAUD DETECTED (Risk: {prob:.2f})")
    else:
        st.success(f"✅ LEGIT TRANSACTION (Risk: {prob:.2f})")

    st.write("Probability Score:", round(prob, 4))