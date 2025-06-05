import pytest
from training.train import prepare_data

def test_prepare_data_shape():
    X_train, X_test, y_train, y_test, _, _, _ = prepare_data()
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert y_train.shape[1] == y_test.shape[1]
