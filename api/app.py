from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("models/fraud_model.pkl")

app = FastAPI(title="Credit Card Fraud Detection API")

# -----------------------------
# INPUT SCHEMA
# -----------------------------
class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

# -----------------------------
# HOME
# -----------------------------
@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}

# -----------------------------
# PREDICT ENDPOINT
# -----------------------------
@app.post("/predict")
def predict(tx: Transaction):

    data = np.array([[
        tx.V1, tx.V2, tx.V3, tx.V4, tx.V5, tx.V6, tx.V7,
        tx.V8, tx.V9, tx.V10, tx.V11, tx.V12, tx.V13, tx.V14,
        tx.V15, tx.V16, tx.V17, tx.V18, tx.V19, tx.V20,
        tx.V21, tx.V22, tx.V23, tx.V24, tx.V25, tx.V26,
        tx.V27, tx.V28, tx.Amount
    ]])

    prob = model.predict_proba(data)[0][1]
    prediction = "FRAUD" if prob > 0.5 else "LEGIT"

    return {
        "fraud_probability": float(prob),
        "prediction": prediction
    }