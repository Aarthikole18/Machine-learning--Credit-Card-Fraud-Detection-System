import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE


# =========================
# 1. LOAD DATASET
# =========================
df = pd.read_csv("data/creditcard.csv")

print("Dataset Loaded ✅")
print("Shape:", df.shape)


# =========================
# 2. SPLIT FEATURES & TARGET
# =========================
X = df.drop("Class", axis=1)
y = df["Class"]

print("Fraud ratio:", y.mean())


# =========================
# 3. TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train-Test Split Done ✅")


# =========================
# 4. HANDLE IMBALANCE (SMOTE)
# =========================
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("SMOTE Applied ✅")


# =========================
# 5. FEATURE SCALING
# =========================
scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test = scaler.transform(X_test)

print("Scaling Done ✅")


# =========================
# 6. TRAIN MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_res, y_train_res)

print("Model Training Done ✅")


# =========================
# 7. PREDICTIONS
# =========================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# =========================
# 8. EVALUATION
# =========================
print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)


# Precision-Recall AUC
precision, recall, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)

print("\nPR AUC Score:", pr_auc)


# =========================
# 9. SAVE MODEL
# =========================
joblib.dump(model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nModel Saved ✅ (models/model.pkl)")


# =========================
# DONE
# =========================
print("\n🚀 TRAINING COMPLETED SUCCESSFULLY")