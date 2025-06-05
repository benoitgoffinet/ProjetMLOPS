import requests

def test_predict_endpoint():
    response = requests.post("http://localhost:8000/predict", json={"question": "How to use pandas?"})
    assert response.status_code == 200
    assert "tags" in response.json()