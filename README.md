💳 Credit Card Fraud Detection System
🚀 Overview

An end-to-end Machine Learning system that detects fraudulent credit card transactions using a trained classification model.
It includes a FastAPI backend for serving predictions and a Streamlit dashboard for real-time interaction.

🎯 Problem Statement

Fraudulent transactions cause huge financial losses in banking systems.
This project focuses on building a model that accurately detects fraud while minimizing false positives.

🧠 Solution Approach
Trained ML model on credit card transaction dataset
Handled class imbalance using SMOTE
Built REST API using FastAPI
Created interactive dashboard using Streamlit
Enabled real-time prediction pipeline
🏗️ System Architecture

Streamlit UI → FastAPI Backend → Trained ML Model → Prediction Output

⚙️ Tech Stack
Python
Pandas, NumPy
Scikit-learn
Imbalanced-learn (SMOTE)
FastAPI
Streamlit
Joblib / Pickle

📊 Features
Real-time fraud detection
Probability score output
Clean interactive UI
REST API for predictions
Saved ML model for reuse

🚀 How to Run the Project
1️⃣ Install dependencies
pip install -r requirements.txt
2️⃣ Train the model
python src/train.py
3️⃣ Run FastAPI backend
uvicorn api.app:app --reload

👉 Open API docs:

http://127.0.0.1:8000/docs

4️⃣ Run Streamlit UI
streamlit run streamlit_app.py

📸 Screenshots
📊 Dashboard UI

✅ Normal Transaction

🚨 Fraud Detection Result

📡 FastAPI Docs

📂 Project Structure
Credit Card Fraud Detection System/
│
├── api/
│   └── app.py
├── data/
│   └── creditcard.csv
├── images/
├── models/
├── src/
│   └── train.py
├── streamlit_app.py
├── requirements.txt
└── README.md

👨‍💻 Author

Aarthi Kole