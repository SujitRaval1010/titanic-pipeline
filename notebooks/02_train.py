# notebooks/02_train.py

import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import importlib.util


# --- Load preprocessing function dynamically (since file starts with "01_") ---
def load_preprocessor():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prep_path = os.path.join(script_dir, "01_data_prep.py")

    spec = importlib.util.spec_from_file_location("data_prep", prep_path)
    data_prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_prep)
    return data_prep.preprocess_data


# --- Training pipeline ---
def train_model():
    # 1. Get path to raw data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(script_dir, "..", "data", "titanic.csv")
    input_csv = os.path.normpath(input_csv)

    # 2. Preprocess data
    preprocess_data = load_preprocessor()
    df = preprocess_data(input_csv)

    # 3. Define features & target
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    # 4. Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42, stratify=y
    )

    # 5. Train model with MLflow tracking
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log metrics and model to MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        print(f"✅ Training complete. Accuracy: {acc:.4f}")


if __name__ == "__main__":
    train_model()
