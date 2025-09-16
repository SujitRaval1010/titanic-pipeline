# notebooks/04_register_model.py

import os
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# --- Config ---
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/titanic")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "databricks")

# ✅ Unity Catalog path (catalog.schema.model_name)
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "workspace.my_schema.titanic_model")

# Unity Catalog prefers aliases instead of stages
ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "staging")

# Performance threshold
ACCURACY_THRESHOLD = float(os.getenv("ACCURACY_THRESHOLD", 0.8))

# --- Setup client ---
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# --- Get or create experiment ---
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    experiment_id = client.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id

# --- Get best run by accuracy ---
runs = client.search_runs(
    experiment_ids=[experiment_id],
    order_by=["metrics.accuracy DESC"],
    max_results=1,
)

if not runs:
    raise RuntimeError(f"❌ No runs found in experiment '{EXPERIMENT_NAME}'!")

best_run = runs[0]
run_id = best_run.info.run_id
accuracy = best_run.data.metrics.get("accuracy", None)

if accuracy is None:
    raise RuntimeError("❌ Accuracy metric not logged in the best run!")

print(f"🔎 Best run {run_id} | Accuracy = {accuracy:.4f}")

# --- Check accuracy threshold ---
if accuracy < ACCURACY_THRESHOLD:
    print(f"❌ Model accuracy {accuracy:.3f} < {ACCURACY_THRESHOLD}. Skipping registration.")
    exit(0)

# --- Register the model ---
model_uri = f"runs:/{run_id}/model"
print(f"✅ Registering model from {model_uri} into {MODEL_NAME}...")

try:
    mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    version = mv.version
    print(f"✅ Registered model version: {version}")
except MlflowException as e:
    print(f"⚠️ mlflow.register_model failed: {e}. Trying client fallback...")
    try:
        client.create_registered_model(MODEL_NAME)
    except Exception:
        pass
    mv_info = client.create_model_version(
        name=MODEL_NAME,
        source=model_uri,
        run_id=run_id,
    )
    version = mv_info.version
    print(f"✅ Created model version (fallback): {version}")

# --- Assign alias ---
try:
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=ALIAS,
        version=version,
    )
    print(f"✅ Model '{MODEL_NAME}' version {version} assigned alias '{ALIAS}'")
except Exception as e:
    print(f"⚠️ Failed to set alias: {e}")
