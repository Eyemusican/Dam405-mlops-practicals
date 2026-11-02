# train.py
# DAM405 Practical 6: Track training runs with MLflow
# Trains a RandomForest on the wine dataset with different hyperparameters
# and logs each run (params, metrics, model) to MLflow.

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

# Store runs in a local SQLite database so the Model Registry works
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Load the wine dataset and split it 80/20 with a fixed seed for reproducibility
X, y = load_wine(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

# All runs are grouped under this experiment name
mlflow.set_experiment("wine-classifier")


def run(n_estimators, max_depth):
    """Train one model and log everything about it as a single MLflow run."""
    with mlflow.start_run() as active_run:
        # Train the model with the given hyperparameters
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        ).fit(Xtr, ytr)

        # Evaluate on the test set (the metric the handout asks for)
        acc = accuracy_score(yte, clf.predict(Xte))

        # Extra metric: 5 fold cross validation on the training set.
        # The test set is small, so many runs tie at 1.0; this helps separate them.
        cv_acc = cross_val_score(
            RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42),
            Xtr, ytr, cv=5,
        ).mean()

        # Log the hyperparameters, the metrics and the trained model
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("cv_accuracy", cv_acc)
        mlflow.sklearn.log_model(
            clf,
            name="model",
            input_example=Xtr[:5],
            # MLflow 3 saves with skops; we trained this model ourselves so the tree type is safe to trust
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print(f"n={n_estimators} depth={max_depth} acc={acc:.4f} cv_acc={cv_acc:.4f} run_id={active_run.info.run_id}")


if __name__ == "__main__":
    # Task 3: run a small grid of experiments so there are runs to compare
    for n in (50, 100, 200):
        for d in (3, 5, None):
            run(n, d)