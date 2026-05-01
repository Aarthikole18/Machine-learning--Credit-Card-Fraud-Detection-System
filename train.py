import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Features & target
X = df.drop("Class", axis=1)
y = df["Class"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

# Train
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print("\n--- Classification Report ---\n")
print(classification_report(y_test, pred))

print("\nROC-AUC:", roc_auc_score(y_test, prob))

# Save model
joblib.dump(model, "models/fraud_model.pkl")

print("\nModel saved successfully!")