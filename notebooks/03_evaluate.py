import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# --- Configuration ---
EXPERIMENT_NAME = "Default"
ACCURACY_THRESHOLD = 0.8

def evaluate_latest_run():
    client = MlflowClient()

    # Get experiment
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found!")

    # Get latest run
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
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

