# notebooks/02_train.py

import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import pandas as pd

# --- Configurable experiment name & tracking ---
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/titanic")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "databricks")


def train_model():
    # --- Configure MLflow ---
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_registry_uri("databricks")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # --- Get paths to prepared train/test data ---
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    train_csv = os.path.join(script_dir, "..", "data", "titanic_train.csv")
    test_csv = os.path.join(script_dir, "..", "data", "titanic_test.csv")

    # --- Load train/test datasets ---
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # --- Define features & target ---
    X_train = train_df.drop("Survived", axis=1)
    y_train = train_df["Survived"]
    X_test = test_df.drop("Survived", axis=1)
    y_test = test_df["Survived"]

    # --- Drop columns that are completely empty ---
    X_train = X_train.dropna(axis=1, how="all")
    X_test = X_test[X_train.columns]  # ensure same columns as X_train

    # --- Handle missing values for numeric columns ---
    numeric_cols = X_train.select_dtypes(include="number").columns
    imputer = SimpleImputer(strategy="median")
    X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = imputer.transform(X_test[numeric_cols])

    # --- Train model & log with MLflow ---
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log params and metrics
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", float(acc))

        # Log model with schema & input example
        input_example = X_train.head(1)
        mlflow.sklearn.log_model(model, "model", input_example=input_example)

        print(f"✅ Training complete. Accuracy: {acc:.4f}")
        print(f"📂 Logged to experiment: {EXPERIMENT_NAME}")


if __name__ == "__main__":
    train_model()
