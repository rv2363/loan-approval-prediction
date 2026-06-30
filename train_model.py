import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# cleaned dataset
df = pd.read_csv("../data/credit_risk_dataset_cleaned.csv")

# target and features
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# smaller model
model = RandomForestClassifier(
    n_estimators=20,
    max_depth=6,
    random_state=42
)

# train
model.fit(X_train, y_train)

# save compressed
joblib.dump(model, "../model/loan_model.pkl", compress=9)

print("Model saved")