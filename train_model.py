import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load new dataset
df = pd.read_csv("data/credit_risk_dataset.csv")

# Clean column names
df.columns = df.columns.str.strip()

print("Columns Found:")
print(df.columns.tolist())

# -------- Encode categorical columns --------
label_encoders = {}

categorical_cols = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file"
]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# -------- Target --------
# loan_status already 0/1 in this dataset usually
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# Remove nulls
data = pd.concat([X, y], axis=1).dropna()
X = data.drop("loan_status", axis=1)
y = data["loan_status"]

# -------- Split --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------- Model --------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\n========================")
print("Model Accuracy:", round(accuracy * 100, 2), "%")
print("========================")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Feature Importance
print("\nFeature Importance:")

importance = model.feature_importances_

for feature, score in sorted(
    zip(X.columns, importance),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{feature}: {score:.4f}")

# Save model
os.makedirs("model", exist_ok=True)

with open("model/loan_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save encoders also (important for Flask later)
with open("model/encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("\nModel + Encoders Saved Successfully!")