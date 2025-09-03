#Tests Unitaires de déploiement CI/CD
import pytest
import os
from fastapi.testclient import TestClient


#Patch du test de variable d'env (pour éviter le plantage lors des tests U)
os.environ["AZURE_STORAGE_CONNECTION_STRING"] = ("DefaultEndpointsProtocol=https;AccountName=fake;AccountKey=fake;EndpointSuffix=core.windows.net")


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
