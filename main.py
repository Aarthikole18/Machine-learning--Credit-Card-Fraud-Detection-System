import pandas as pd
import joblib

# =========================
# LOAD MODEL & SCALER
# =========================
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

print("Model Loaded ✅")


# =========================
# SAMPLE INPUT (SIMULATION)
# =========================
# One transaction (example from dataset format)
sample_transaction = [
    0,  # Time
    -1.3598071336738,  # V1
    -0.0727811733098497,  # V2
    2.53634673796914,  # V3
    1.37815522427443,  # V4
    -0.338320769942518,  # V5
    0.462387777762292,  # V6
    0.239598554061257,  # V7
    0.0986979012610507,  # V8
    0.363786969611213,  # V9
    0.0907941719789316,  # V10
    -0.551599533260813,  # V11
    -0.617800855762348,  # V12
    -0.991389847235408,  # V13
    -0.311169353699879,  # V14
    1.46817697209427,  # V15
    -0.470400525259478,  # V16
    0.207971241929242,  # V17
    0.0257905801985591,  # V18
    0.403992960255733,  # V19
    0.251412098239705,  # V20
    -0.018306777944153,  # V21
    0.277837575558899,  # V22
    -0.110473910188767,  # V23
    0.0669280749146731,  # V24
    0.128539358273528,  # V25
    -0.189114843888824,  # V26
    0.133558376740387,  # V27
    -0.0210530534538215,  # V28
    149.62  # Amount
]


# =========================
# CONVERT TO DATAFRAME
# =========================
columns = [f"V{i}" for i in range(1, 29)]
columns = ["Time"] + columns + ["Amount"]

df_input = pd.DataFrame([sample_transaction], columns=columns)


# =========================
# SCALE INPUT
# =========================
df_scaled = scaler.transform(df_input)


# =========================
# PREDICTION
# =========================
prediction = model.predict(df_scaled)[0]
probability = model.predict_proba(df_scaled)[0][1]


# =========================
# OUTPUT RESULT
# =========================
print("\n===== RESULT =====")

if prediction == 1:
    print("🚨 FRAUD DETECTED!")
else:
    print("✅ NORMAL TRANSACTION")

print("Fraud Probability:", round(probability, 4))