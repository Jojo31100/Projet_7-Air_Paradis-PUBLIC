# test_api.py
import pytest
from fastapi.testclient import TestClient
from api import app


client = TestClient(app)

#Test unitaire : présente du site en ligne
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "API Air Paradis en ligne" in response.json()["message"]

#Test unitaire : tweet positif
def test_predict_positive():
    payload = {"tweetRecu": "I'm really happy !"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["Prédiction"] == "Tweet positif"

#Test unitaire : tweet négatif
def test_predict_negative():
    payload = {"tweetRecu": "I feel soooo sad today..."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["Prédiction"] == "Tweet négatif"
