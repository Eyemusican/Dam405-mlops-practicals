from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from flaml import AutoML
import time

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

for budget in [10, 60, 180]:
    automl = AutoML()
    start = time.time()
    automl.fit(X_train, y_train,
               task='classification',
               metric='accuracy',
               time_budget=budget,
               verbose=0)
    elapsed = time.time() - start
    test_acc = accuracy_score(y_test, automl.predict(X_test))
    print(f'Budget {budget}s | Best: {automl.best_estimator} | '
          f'CV: {1 - automl.best_loss:.4f} | Test: {test_acc:.4f} | '
          f'Time: {elapsed:.1f}s')


    