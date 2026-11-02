# predict.py
# DAM405 Practical 6, Task 6: Load the registered model back from the Model Registry and predict.

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Use the same tracking database as train.py and register.py
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Recreate the same test split as train.py (same random_state) so we predict on unseen data
X, y = load_wine(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the latest registered version of 'wine-classifier' (as in the handout)
model = mlflow.sklearn.load_model("models:/wine-classifier/latest")
print("Loaded model prediction:", model.predict(Xte[:5]))
print("Actual labels:          ", yte[:5])

# Load the version the 'production' alias points to (MLflow 3 replacement for stages)
prod_model = mlflow.sklearn.load_model("models:/wine-classifier@production")
print("Production model prediction:", prod_model.predict(Xte[:5]))