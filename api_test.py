#API - VERSION SBERT DEBUG


import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from collections import Counter
from azure.storage.blob import BlobServiceClient
import numpy
#import tensorflow
#from sentence_transformers import SentenceTransformer


#Fonction de nettoyage de texte adaptée de la fonction "_textCleaning", mais à un string, plutôt qu'à une variable de Dataframe
def _textCleaning_API(_inputText, _inputDropTokenIfLessThanXChars=0, _inputDropTokenIfFoundMoreThanXTimes=0, _inputLanguage="None", _inputLemmatizationOrStemmingChoice="None"):
    #1ère étape : on passe tous les caractères en minuscules et on vire les caractères spéciaux !
    _inputText = str(_inputText).lower()
    for caractere in ["-", "+", "/", "#", "_", "&", "(", ")", "@"]:
        _inputText = _inputText.replace(caractere, " ")

    #2ème étape : on ne garde que les mots constitués de caracètres alphabétiques (Tokenisation)
    tokenizer = nltk.RegexpTokenizer(r"[^\W\d_]+")
    tokens = tokenizer.tokenize(_inputText)

    #3ème étape : on vire les tokens de moins de "_inputDropTokenIfLessThanXChars" caractères
    if _inputDropTokenIfLessThanXChars != 0:
        tokens = [t for t in tokens if len(t) >= _inputDropTokenIfLessThanXChars]

    #4ème étape : on virer les X tokens les plus fréquents
    if _inputDropTokenIfFoundMoreThanXTimes != 0:
        numberOfTokens = Counter(tokens)
        stopWords = {item for item, count in numberOfTokens.items() if count >= _inputDropTokenIfFoundMoreThanXTimes}
        tokens = [token for token in tokens if token not in stopWords]

    #5ème étape : on télécharge les StopWords NLTK les plus courants de la langue "_inputLanguage", ou on passe si "_inputLanguage" == "None"
    if str(_inputLanguage).lower() != "none":
        nltk.download("stopwords", quiet=True)
        stopWords = set(stopwords.words(_inputLanguage))
        tokens = [token for token in tokens if token not in stopWords]

    #6ème étape : on lemmatise ("Lem") ou on racinise ("Stem") les tokens, ou on ne fait rien du tout ("None")
    if str(_inputLemmatizationOrStemmingChoice).lower() == "lem":
        nltk.download("wordnet", quiet=True)
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    elif str(_inputLemmatizationOrStemmingChoice).lower() == "stem":
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]

    #Retour en string
    return " ".join(tokens)


#Définition des variables
#AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
#if(AZURE_STORAGE_CONNECTION_STRING is None):
#    raise ValueError("La variable d'environnement AZURE_STORAGE_CONNECTION_STRING n'est pas définie !")
CONTAINER_NAME = "models"
BLOB_FOLDER = "SBERT-TexteDL/"
LOCAL_MODEL_AND_TOKENIZER_DIR = "./model"

#Etape 1 : recopie du Modèle et du Tokenizer (depuis Azure BlobStorage vers la VM locale)
#Création du dossier local (s'il n'existe pas déjà)
#os.makedirs(LOCAL_MODEL_AND_TOKENIZER_DIR, exist_ok=True)

#Init du client Azure Blob Storage
#blobStorageClient = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
#containerClient = blobStorageClient.get_container_client(CONTAINER_NAME)

#Téléchargement de tous les fichiers du dossier, si le dossier local est vide
#if(not os.listdir(LOCAL_MODEL_AND_TOKENIZER_DIR)):
#    for fichier in containerClient.list_blobs(name_starts_with=BLOB_FOLDER):
#        local_path = os.path.join(LOCAL_MODEL_AND_TOKENIZER_DIR, os.path.basename(fichier.name))
#        with open(local_path, "wb") as fileToCopy:
#            fileToCopy.write(containerClient.download_blob(fichier.name).readall())

#Etape 2 : chargement de l'encodeur et du Modèle
#SBERT_Encoder = SentenceTransformer("all-mpnet-base-v2")
#SBERT_Model = tensorflow.keras.models.load_model(os.path.join(LOCAL_MODEL_AND_TOKENIZER_DIR, "best_model.SBERT.keras"))

#Etape 3 : FastAPI
app = FastAPI(title="air-paradis-api")

#Etape 4 : CORS (pour pouvoir accéder en version web depuis n'importe où)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

#Classe pour requête POST
class TweetRequest(BaseModel):
    tweetRecu: str

#Route racine
@app.get("/")
async def root():
    return {"message": "API Air Paradis en ligne - SBERT & BlobStorage v1.0"}

#Route prédiction
@app.post("/predict")
def predict(request: TweetRequest):
    #Nettoyage du texte
    text = _textCleaning_API(request.tweetRecu, 0, 0, "None", "None")

    #Encodage avec SBERT
#    embedding = SBERT_Encoder.encode([text], convert_to_numpy=True)

    #Prédiction Keras
#    y_proba = SBERT_Model.predict(embedding).flatten()
#    pred_class = int((y_proba >= 0.5)[0])

    #Mapping classes
#    if(pred_class == 1):
#        label = "Tweet positif"
#    else:
#        label = "Tweet négatif"

#    return {"Tweet": request.tweetRecu, "Texte traité": text, "Prédiction": label, "Probabilité": float(numpy.max(y_proba))}
    return {"Tweet": request.tweetRecu, "Texte traité": text}
