# register.py
# DAM405 Practical 6, Task 5: Register the best run's model in the MLflow Model Registry from code.

import mlflow
from mlflow import MlflowClient

# Use the same tracking database as train.py
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Best run from Task 4: n_estimators=100, max_depth=5 (smiling-grouse-446),
# highest cv_accuracy (0.9786) and the simpler of the two tied runs
RUN_ID = "f18cc271baf54c1d8f9559ef9d532623"

# Register the model logged under that run as a new version of 'wine-classifier'
result = mlflow.register_model(f"runs:/{RUN_ID}/model", "wine-classifier")
print("Registered version:", result.version)

# MLflow 3 replaces stages with aliases; point the 'production' alias at this version
client = MlflowClient()
client.set_registered_model_alias("wine-classifier", "production", result.version)
print("Alias 'production' now points to version", result.version)