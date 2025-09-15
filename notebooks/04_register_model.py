from mlflow.tracking import MlflowClient

# --- Config ---
experiment_name = "/Shared/titanic"
model_name = "titanic_model"
stage = "Staging"
ACCURACY_THRESHOLD = 0.8

client = MlflowClient()

# Get experiment
experiment = client.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = client.create_experiment(experiment_name)
else:
    experiment_id = experiment.experiment_id

# Get best run
runs = client.search_runs(
    experiment_ids=[experiment_id],
    order_by=["metrics.accuracy DESC"],
    max_results=1
)

if not runs:
    raise ValueError("No runs found in experiment!")

best_run = runs[0]
accuracy = best_run.data.metrics.get("accuracy", 0)

if accuracy >= ACCURACY_THRESHOLD:
    model_uri = f"runs:/{best_run.info.run_id}/model"
    try:
        client.create_registered_model(model_name)
    except Exception as e:
        print(f"Model may already exist: {e}")
    client.create_model_version(model_name, model_uri, stage)
    print(f"✅ Model '{model_name}' registered in stage '{stage}' with accuracy {accuracy}")
else:
    print(f"❌ Model accuracy {accuracy} < {ACCURACY_THRESHOLD}, skipping registration")
