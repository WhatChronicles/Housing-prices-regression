import os
import pandas as pd
import joblib
from typing import Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def train_models() -> Dict:
    # ✅ Load Titanic data
    df = pd.read_csv("data/train.csv")

    # ✅ Select features and target
    categorical_cols = ["Sex", "Embarked", "Pclass"]
    numeric_cols = ["Age", "Fare", "SibSp", "Parch"]
    target_col = "Survived"

    # ✅ Drop rows with missing target or key features
    df = df.dropna(subset=[target_col] + numeric_cols)

    # ✅ Encode categorical features
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # ✅ Prepare training data
    X = df[categorical_cols + numeric_cols]
    y = df[target_col]

    # ✅ Split for validation (optional)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # ✅ Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # ✅ Ensure models/ folder exists
    os.makedirs("models", exist_ok=True)

    # ✅ Save model
    joblib.dump(model, "models/best_model.pkl")

    # ✅ Save encoders
    joblib.dump(encoders, "models/label_encoders.pkl")

    print("✅ Model and encoders saved to 'models/'")

    return {
        "model": model,
        "encoders": encoders
    }

# Optional: run directly
if __name__ == "__main__":
    train_models()