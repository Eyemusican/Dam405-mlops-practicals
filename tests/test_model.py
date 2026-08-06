from src.model import train_and_score


def test_accuracy_is_reasonable():
    assert train_and_score() > 0.8
