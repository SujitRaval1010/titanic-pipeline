# notebooks/02_train.py

import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# --- Configurable experiment name (consistent everywhere) ---
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/titanic")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "databricks")


def train_model():
    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 1. Get paths to prepared train/test data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_csv = os.path.join(script_dir, "..", "data", "titanic_train.csv")
    test_csv = os.path.join(script_dir, "..", "data", "titanic_test.csv")

    # 2. Load train/test datasets
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # 3. Define features & target
    X_train = train_df.drop("Survived", axis=1)
    y_train = train_df["Survived"]
    X_test = test_df.drop("Survived", axis=1)
    y_test = test_df["Survived"]

    # 4. Train model with MLflow tracking
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log metrics and model
        mlflow.log_metric("accuracy", float(acc))
        input_example = X_train.head(1)
        mlflow.sklearn.log_model(model, "model", input_example=input_example)

        print(f"✅ Training complete. Accuracy: {acc:.4f}")


if __name__ == "__main__":
    train_model()
