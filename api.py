#API

from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow
from transformers import BertTokenizer
import numpy


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
    if(_inputDropTokenIfLessThanXChars != 0):
        tokens = [t for t in tokens if len(t) >= _inputDropTokenIfLessThanXChars]

    #4ème étape : on virer les X tokens les plus fréquents
    if(_inputDropTokenIfFoundMoreThanXTimes != 0):
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

#Connexion à Google Drive
#drive.mount("/content/drive/")

#Etape 1 : Définition du chemin d'accès du modèle et du tokenizer
MODEL_PATH = "///content/drive/My Drive/Colab_Notebooks/Project_7/MLflow_data/734748334327253282/e5688fb58e9c439ebef416e432a216a9/artifacts/Pipeline-BERT-TexteDL/best_model.BERT.keras"
TOKENIZER_PATH = "///content/drive/My Drive/Colab_Notebooks/Project_7/MLflow_data/734748334327253282/e5688fb58e9c439ebef416e432a216a9/artifacts/Pipeline-BERT-TexteDL/best_model.BERT.keras_tokenizer"

#Etape 2 : Chargement du modèle et du tokenizer
model = TFBertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_PATH)

#Etape 3 : FastAPI
app = FastAPI(title="API_Projet_7-Air_Paradis")

class TweetRequest(BaseModel):
    tweetRecu: str

@app.get("/")
async def root():
    return {"message": "API Air Paradis en ligne"}

@app.post("/predict")
def predict(request: TweetRequest):
    #Nettoyage du texte
    text = _textCleaning_API(request.tweetRecu, 0, 0, "None", "None")

    #Tokenisation
    inputs = tokenizer(text, return_tensors="tf", padding="max_length", truncation=True, max_length=128)

    #Prédiction
    preds = model(inputs)[0].numpy()
    probs = tensorflow.nn.softmax(preds, axis=1).numpy()
    pred_class = int(numpy.argmax(probs, axis=1)[0])

    #Mapping classes
    if(pred_class == 1):
        label = "Tweet positif"
    else:
        label = "Tweet négatif"

    return {"Tweet": request.tweetRecu, "Texte traité": text, "Prédiction": label, "Probabilité": float(numpy.max(probs))}
