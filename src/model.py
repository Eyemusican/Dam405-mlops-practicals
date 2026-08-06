from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def train_and_score(random_state: int = 42) -> float:
    """Train a model on Iris and return test accuracy."""
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=random_state)
    clf = LogisticRegression(max_iter=200).fit(Xtr, ytr)
    return clf.score(Xte, yte)
