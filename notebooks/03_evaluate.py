import os
import mlflow
from mlflow.tracking import MlflowClient

# --- Configuration ---
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/titanic")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "databricks")
ACCURACY_THRESHOLD = float(os.getenv("ACCURACY_THRESHOLD", 0.8))

def evaluate_latest_run():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # Get experiment
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found!")

    # Get latest run
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1
    )

    if not runs:
        raise ValueError("No runs found in MLflow experiment!")

    latest_run = runs[0]
    run_id = latest_run.info.run_id
    metrics = latest_run.data.metrics

    print(f"🔎 Latest Run ID: {run_id}")
    print(f"📊 Metrics: {metrics}")

    # Extract accuracy
    accuracy = metrics.get("accuracy")
    if accuracy is None:
        raise ValueError("Accuracy metric not found in the latest run!")

    # Compare against threshold
    if accuracy >= ACCURACY_THRESHOLD:
        print(f"✅ Model passed! Accuracy = {accuracy:.3f} (>= {ACCURACY_THRESHOLD})")
        return True
    else:
        print(f"❌ Model failed. Accuracy = {accuracy:.3f} (< {ACCURACY_THRESHOLD})")
        return False

if __name__ == "__main__":
    evaluate_latest_run()
