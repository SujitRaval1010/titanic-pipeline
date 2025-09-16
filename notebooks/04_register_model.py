import os
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# --- Config ---
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/titanic")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "databricks")

# ✅ Unity Catalog path (catalog.schema.model_name)
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "workspace.my_schema.titanic_model")

# Instead of stages, we use aliases in Unity Catalog
ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "staging")

ACCURACY_THRESHOLD = float(os.getenv("ACCURACY_THRESHOLD", 0.8))

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

# Register model
try:
    mv = mlflow.register_model(model_uri, MODEL_NAME)
    version = mv.version
    print(f"Registered model version: {version}")
except MlflowException as e:
    print(f"mlflow.register_model failed with: {e}. Trying client.create_model_version fallback.")
    try:
        client.create_registered_model(MODEL_NAME)
    except Exception:
        pass
    mv_info = client.create_model_version(
        name=MODEL_NAME,
        source=model_uri,
        run_id=run_id
    )
    version = mv_info.version
    print(f"Created model version (fallback): {version}")

# ✅ Assign alias (instead of stage transition)
try:
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=ALIAS,
        version=version
    )
    print(f"✅ Model '{MODEL_NAME}' version {version} assigned alias '{ALIAS}'")
except Exception as e:
    print(f"⚠️ Failed to set alias: {e}")
