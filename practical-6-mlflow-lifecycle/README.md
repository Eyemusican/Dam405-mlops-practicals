# DAM405 Practical 6: Managing the ML Lifecycle with MLflow

In this practical I used MLflow to track, compare, register and load a RandomForest model trained on the wine dataset.

## Files

* `train.py` trains 9 models with different `n_estimators` and `max_depth` and logs each run to MLflow (Task 2 and 3)
* `register.py` registers the best model as `wine-classifier` (Task 5)
* `predict.py` loads the registered model and predicts (Task 6)
* `evidence/` has my screenshots

## How to run

Run these inside the `practical-6-mlflow-lifecycle` folder.

```powershell
pip install mlflow scikit-learn
python train.py
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open http://localhost:5000 to compare the runs. Then in another terminal:

```powershell
python register.py
python predict.py
```

`register.py` uses the run ID from my training, so change `RUN_ID` to your own best run if you train again.

Optional REST serving:

```powershell
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
$env:OPENBLAS_NUM_THREADS = "1"
mlflow models serve -m "models:/wine-classifier/latest" -p 5001 --env-manager local
```

## Notes

I used MLflow 3.17.2, so I made a few small changes to the handout code. I set a SQLite tracking URI so the Model Registry works, and added `skops_trusted_types` because `log_model` failed without it. All runs got 1.0 test accuracy, so I also logged `cv_accuracy` to pick the best one.

`mlflow.db` and `mlruns/` are not pushed since `train.py` creates them again.

## Result

Best run was `n_estimators=100`, `max_depth=5` with cv_accuracy 0.9786. It is registered as `wine-classifier` and the loaded model predicted `[0 0 2 0 1]`, which matches the actual labels.