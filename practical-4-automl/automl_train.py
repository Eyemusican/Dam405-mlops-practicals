from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Task 1: Load and split the data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print('Train shape:', X_train.shape)
print('Test shape :', X_test.shape)

# Task 2: Baseline model
base = LogisticRegression(max_iter=2000).fit(X_train, y_train)
base_acc = accuracy_score(y_test, base.predict(X_test))
print('Baseline accuracy:', round(base_acc, 4))


# Task 3: Run AutoML
from flaml import AutoML

automl = AutoML()
automl.fit(X_train, y_train,
           task='classification',
           metric='accuracy',
           time_budget=60)  # increase for a better search



# Task 4: Inspect the best model
print('Best estimator :', automl.best_estimator)
print('Best config    :', automl.best_config)
print('Best CV score  :', 1 - automl.best_loss)


# Task 5: Evaluate on the test set
auto_acc = accuracy_score(y_test, automl.predict(X_test))
print('AutoML test accuracy  :', round(auto_acc, 4))
print('Baseline test accuracy:', round(base_acc, 4))


# Task 6: Save the best model
import joblib
joblib.dump(automl, 'automl_model.joblib')
print('Saved automl_model.joblib')