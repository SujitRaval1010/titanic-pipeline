# notebooks/03_evaluate.py

import os
import mlflow
from mlflow.tracking import MlflowClient

# --- Configuration ---
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/titanic")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "databricks")
ACCURACY_THRESHOLD = float(os.getenv("ACCURACY_THRESHOLD", 0.8))


def evaluate_latest_run():
    # Configure MLflow client
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # Fetch experiment
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"❌ Experiment '{EXPERIMENT_NAME}' not found!")

    # Fetch latest run
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise RuntimeError("❌ No runs found in MLflow experiment!")

    latest_run = runs[0]
    run_id = latest_run.info.run_id
    metrics = latest_run.data.metrics

    print(f"🔎 Latest Run ID: {run_id}")
    print(f"📊 Metrics: {metrics}")

    # Extract accuracy safely
    accuracy = metrics.get("accuracy", None)
    if accuracy is None:
        raise RuntimeError("❌ Accuracy metric not found in the latest run!")

    # Compare against threshold
    if accuracy >= ACCURACY_THRESHOLD:
        print(f"✅ Model passed! Accuracy = {accuracy:.3f} (>= {ACCURACY_THRESHOLD})")
        return True
    else:
        print(f"❌ Model failed. Accuracy = {accuracy:.3f} (< {ACCURACY_THRESHOLD})")
        return False


if __name__ == "__main__":
    success = evaluate_latest_run()
    # Exit with proper code for CI/CD pipelines
    if not success:
        exit(1)
