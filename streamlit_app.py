import streamlit as st
import requests

st.set_page_config(page_title="Fraud Detection System", layout="wide")

# =========================
# HEADER
# =========================
st.title("💳 Credit Card Fraud Detection System")
st.markdown("### Real-time AI Fraud Detection Dashboard")

st.divider()

# =========================
# SIDEBAR INPUT (CLEAN UI)
# =========================
st.sidebar.header("Transaction Input")

def input_features():

    Time = st.sidebar.number_input("Time", value=0.0)
    Amount = st.sidebar.number_input("Amount", value=0.0)

    V = []
    for i in range(1, 29):
        V.append(st.sidebar.number_input(f"V{i}", value=0.0))

    return Time, V, Amount


Time, V, Amount = input_features()

# =========================
# MAIN PANEL
# =========================
col1, col2, col3 = st.columns(3)

col1.metric("Model", "Random Forest")
col2.metric("System", "Live API")
col3.metric("Status", "Active 🟢")

st.divider()

# =========================
# PREDICT BUTTON
# =========================
if st.button("🚀 Check Fraud"):

    url = "http://127.0.0.1:8000/predict"

    data = {
        "Time": Time,
        "Amount": Amount
    }

    for i in range(28):
        data[f"V{i+1}"] = V[i]

    response = requests.post(url, json=data)

    if response.status_code == 200:
        result = response.json()

        st.subheader("Prediction Result")

        if result["fraud_prediction"] == 1:
            st.error("🚨 FRAUD DETECTED!")
            st.markdown("### ❌ Transaction Blocked")
        else:
            st.success("✅ LEGITIMATE TRANSACTION")
            st.markdown("### ✔ Transaction Approved")

        st.metric("Fraud Probability", f"{result['fraud_probability']:.4f}")

    else:
        st.error("API not running")