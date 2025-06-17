import requests

def test_predict_endpoint():
    response = requests.post("http://localhost:8000/predict", data="How to use pandas?"))
    assert response.status_code == 200
    assert "tags" in response.json()