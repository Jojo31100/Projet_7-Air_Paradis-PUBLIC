#Tests Unitaires de déploiement CI/CD
import pytest
import os
import unittest.mock as mock
from fastapi.testclient import TestClient
import numpy

#Patch de la variable d'env (pour éviter le plantage lors des tests U)
os.environ["AZURE_STORAGE_CONNECTION_STRING"] = ("DefaultEndpointsProtocol=https;AccountName=fake;AccountKey=fake;EndpointSuffix=core.windows.net")

#On simule les deux chargements de modèles
with mock.patch("api.tensorflow.keras.models.load_model") as mock_keras_load_model:
    with mock.patch("api.tensorflow_hub.load") as mock_hub_load:

        #Etape 1 : on simule l'encodeur USE,
        # qui va renvoyer un "embedding" factice
        mock_encoder = mock.MagicMock()
        mock_encoder.return_value = numpy.zeros(512)
        mock_hub_load.return_value = mock_encoder

        #Etape 2 : on simule le modèle USE
        mock_model = mock.MagicMock()

        #Etape 3 : on configure la méthode .predict() du mock pour qu'elle
        # renvoie un résultat spécifique en fonction du texte d'entrée.
        def mock_predict_keras(embedding):
            #Normalement, la prédiction devrait se faire sur l'embedding...
            # Mais pour le test, nous savons que l'embedding est lié au tweet.
            # On simule donc le comportement du modèle réel :
            # - Le test pour "happy" renverra un résultat positif
            # - Le test pour "sad", un résultat négatif
            if("happy" in client.predict_tweet_text):
                return numpy.array([[0.9]]) # Probabilité positive
            elif("sad" in client.predict_tweet_text):
                return numpy.array([[0.1]]) # Probabilité négative
            return np.array([[0.5]])

        #... et on fait en sorte que mock_model.predict() utilise NOTRE
        # fonction de simulation !
        mock_model.predict.side_effect = mock_predict_keras
        mock_keras_load_model.return_value = mock_model

        #On patch également le client Azure
        with mock.patch("azure.storage.blob.ContainerClient") as mock_container_client:
            mock_container_client.return_value.list_blobs.return_value = []

            #Maintenant que tout ça est fait, on peut importer l'application
            from api import app

#On crée le client de test en dehors des blocs with pour qu'il soit
#accessible par toutes les fonctions de test.
client = TestClient(app)
client.predict_tweet_text = "" #Variable de stockage de texte pour le mock

#Test unitaire : présence du site en ligne
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "API Air Paradis en ligne" in response.json()["message"]

#Test unitaire : tweet positif
def test_predict_positive():
    payload = {"tweetRecu": "I'm really happy !"}
    client.predict_tweet_text = payload["tweetRecu"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["Prédiction"] == "Tweet positif"

#Test unitaire : tweet négatif
def test_predict_negative():
    payload = {"tweetRecu": "I feel soooo sad today..."}
    client.predict_tweet_text = payload["tweetRecu"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["Prédiction"] == "Tweet négatif"
