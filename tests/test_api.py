import requests

def test_predict_endpoint():
    response = requests.post(
        "http://localhost:8000/predict",
        json={"text": "How to use pandas?"}
    )
    print("Status code:", response.status_code)
    print("Response body:", response.text)
    assert response.status_code == 200
    assert "keywords" in response.json()