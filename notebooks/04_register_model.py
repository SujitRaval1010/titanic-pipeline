# notebooks/04_register_model.py

import os
import mlflow
from mlflow.tracking import MlflowClient
import tempfile
import hashlib

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


# --- Compute hash of full model directory ---
def compute_model_hash(model_uri):
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = mlflow.artifacts.download_artifacts(model_uri, dst_path=tmpdir)

        md5_hash = hashlib.md5()
        for root, _, files in os.walk(local_path):
            for file in sorted(files):  # ensure consistent order
                file_path = os.path.join(root, file)
                with open(file_path, "rb") as f:
                    while chunk := f.read(8192):
                        md5_hash.update(chunk)
        return md5_hash.hexdigest()


# --- Compute new hash ---
new_hash = compute_model_hash(model_uri)
print(f"ℹ️ New model hash: {new_hash}")

# --- Compare with existing alias ---
skip_registration = False
try:
    alias_uri = f"models:/{MODEL_NAME}/{ALIAS}"
    existing_hash = compute_model_hash(alias_uri)
    existing_version = client.get_model_version_by_alias(MODEL_NAME, ALIAS).version
    print(f"ℹ️ Existing model hash (alias='{ALIAS}', version={existing_version}): {existing_hash}")

    if new_hash == existing_hash:
        print("✅ Model is identical to existing version. Skipping registration.")
        skip_registration = True
except Exception:
    print(f"ℹ️ No existing model found at alias '{ALIAS}'")
    existing_version = None

# --- Register model only if different ---
if not skip_registration:
    mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    version = mv.version
    print(f"✅ Registered NEW model version: {version}")

    # Set alias to new version
    client.set_registered_model_alias(MODEL_NAME, ALIAS, version)
    print(f"✅ Alias '{ALIAS}' now points to version {version}")
else:
    print(f"🔄 Alias '{ALIAS}' stays on version {existing_version}, no new version created")
