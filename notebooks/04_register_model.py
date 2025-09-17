# notebooks/04_register_model.py

import os
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import tempfile
import joblib
import hashlib
import shutil

# --- Config ---
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/titanic")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "databricks")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "workspace.my_schema.titanic_model")
ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "staging")
ACCURACY_THRESHOLD = float(os.getenv("ACCURACY_THRESHOLD", 0.8))

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# --- Get experiment ---
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
experiment_id = experiment.experiment_id if experiment else client.create_experiment(EXPERIMENT_NAME)

# --- Get best run by accuracy ---
runs = client.search_runs([experiment_id], order_by=["metrics.accuracy DESC"], max_results=1)
if not runs:
    raise RuntimeError("❌ No runs found!")

best_run = runs[0]
run_id = best_run.info.run_id
accuracy = best_run.data.metrics.get("accuracy", None)
if accuracy is None or accuracy < ACCURACY_THRESHOLD:
    print(f"❌ Accuracy {accuracy} < {ACCURACY_THRESHOLD}. Skipping registration.")
    exit(0)

print(f"🔎 Best run {run_id} | Accuracy = {accuracy:.4f}")

model_uri = f"runs:/{run_id}/model"

# --- Compute hash of new model ---
def compute_model_hash(model_uri):
    with tempfile.TemporaryDirectory() as tmpdir:
        local_model_path = mlflow.sklearn.load_model(model_uri)
        tmp_file = os.path.join(tmpdir, "model.pkl")
        joblib.dump(local_model_path, tmp_file)
        with open(tmp_file, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

new_hash = compute_model_hash(model_uri)
print(f"ℹ️ New model hash: {new_hash}")

# --- Compare with existing model by alias ---
skip_registration = False
try:
    alias_uri = f"models:/{MODEL_NAME}/{ALIAS}"
    existing_hash = compute_model_hash(alias_uri)
    print(f"ℹ️ Existing model hash (alias='{ALIAS}'): {existing_hash}")

    if new_hash == existing_hash:
        print("✅ Model is identical to existing version. Skipping registration.")
        skip_registration = True
except Exception:
    print(f"ℹ️ No existing model found at alias '{ALIAS}'")

# --- Register model if changed ---
if not skip_registration:
    mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    version = mv.version
    print(f"✅ Registered new model version: {version}")

    # Assign alias
    client.set_registered_model_alias(name=MODEL_NAME, alias=ALIAS, version=version)
    print(f"✅ Alias '{ALIAS}' now points to version {version}")
