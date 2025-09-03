# Tests unitaires de déploiement CI/CD de l'API Air Paradis

import os
import pytest
from unittest import mock
import numpy
from fastapi.testclient import TestClient

#Patch de la variable d'environnement,
# pour éviter que la connexion à Azure plante lors des tests
os.environ["AZURE_STORAGE_CONNECTION_STRING"] = ("DefaultEndpointsProtocol=https;AccountName=fake;AccountKey=fake;EndpointSuffix=core.windows.net")

with mock.patch("azure.storage.blob.BlobServiceClient") as mock_blob:
    #On simule le client de conteneur
    mock_container = mock.MagicMock()
    #list_blobs renvoie une liste vide pour éviter toute requête réelle
    mock_container.list_blobs.return_value = []
    mock_blob.from_connection_string.return_value.get_container_client.return_value = mock_container

    #On simule le modèle USE
    with mock.patch("tensorflow.keras.models.load_model") as mock_keras:
        mock_model = mock.MagicMock()

        #On configure la méthode .predict() pour qu'elle
        # renvoie un résultat spécifique selon le texte du tweet
        def mock_predict(embedding):
            #On récupère le texte envoyé dans le client de test
            text = getattr(client, "predict_tweet_text", "").lower()
            if "happy" in text:
                return numpy.array([[0.9]])  # Tweet positif
            elif "sad" in text:
                return numpy.array([[0.1]])  # Tweet négatif
            return numpy.array([[0.5]])

        mock_model.predict.side_effect = mock_predict
        mock_keras.return_value = mock_model

        #On simule l'encodeur Universal Sentence Encoder (USE)
        with mock.patch("tensorflow_hub.load") as mock_hub:
            mock_encoder = mock.MagicMock()
            #Le mock renvoie un embedding factice de 512 dimensions
            mock_encoder.return_value = numpy.zeros(512)
            mock_hub.return_value = mock_encoder

            #On importe l'application FastAPI APRÈS tous les patchs
            from api import app

#On crée le client de test en dehors des blocs "with"
# pour qu'il soit accessible par toutes les fonctions de test
client = TestClient(app)
client.predict_tweet_text = ""  # Variable de stockage du texte pour le mock

# Test unitaire : vérification que le site est en ligne
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
