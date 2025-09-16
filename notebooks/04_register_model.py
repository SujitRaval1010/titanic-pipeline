import os
import time
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# --- Config ---
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/titanic")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "databricks")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "titanic_model")
STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Staging")
ACCURACY_THRESHOLD = float(os.getenv("ACCURACY_THRESHOLD", 0.8))
POLL_SECONDS = 2
MAX_POLL_ATTEMPTS = 15

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# Get experiment (create if not exists)
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    experiment_id = client.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id

# Find best run by accuracy
runs = client.search_runs(
    experiment_ids=[experiment_id],
    order_by=["metrics.accuracy DESC"],
    max_results=1
)

if not runs:
    raise ValueError("No runs found in experiment!")

best_run = runs[0]
accuracy = best_run.data.metrics.get("accuracy", 0.0)
run_id = best_run.info.run_id

if accuracy < ACCURACY_THRESHOLD:
    print(f"❌ Model accuracy {accuracy} < {ACCURACY_THRESHOLD}, skipping registration")
    exit(0)

model_uri = f"runs:/{run_id}/model"
print(f"✅ Best run {run_id} with accuracy {accuracy:.4f}. Registering model from {model_uri} -> {MODEL_NAME}")

# Register model using mlflow.register_model (recommended)
try:
    mv = mlflow.register_model(model_uri, MODEL_NAME)
    version = mv.version
    print(f"Registered model version: {version}")
except MlflowException as e:
    # If model already exists (or other error), try to create model version via client
    print(f"mlflow.register_model failed with: {e}. Trying client.create_model_version fallback.")
    try:
        client.create_registered_model(MODEL_NAME)
    except Exception:
        pass
    mv_info = client.create_model_version(name=MODEL_NAME, source=mv.source if 'mv' in locals() else model_uri, run_id=run_id)
    version = mv_info.version
    print(f"Created model version (fallback): {version}")

# Promote the model version to the desired stage (with retries)
attempt = 0
while attempt < MAX_POLL_ATTEMPTS:
    try:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version,
            stage=STAGE,
            archive_existing_versions=False
        )
        print(f"✅ Model '{MODEL_NAME}' version {version} transitioned to stage '{STAGE}'")
        break
    except Exception as e:
        attempt += 1
        print(f"Attempt {attempt}/{MAX_POLL_ATTEMPTS} - transition failed: {e}. Retrying in {POLL_SECONDS}s...")
        time.sleep(POLL_SECONDS)
else:
    print("⚠️ Could not transition model version stage within max attempts.")
