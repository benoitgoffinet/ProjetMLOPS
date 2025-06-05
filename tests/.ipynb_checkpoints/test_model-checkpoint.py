import joblib
import os

def test_model_exists():
    assert os.path.exists("models/model.pkl")

def test_model_loads():
    model = joblib.load("models/model.pkl")
    assert hasattr(model, "predict")