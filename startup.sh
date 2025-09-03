#!/bin/bash

#Définir la variable d'environnement pour le chemin de l'application
#Le chemin par défaut pour les App Services Linux est /home/site/wwwroot
APP_PATH="/home/site/wwwroot"

#Chercher et activer le Virtual Environment
VENV_PATH=$(find /home -type d -name "antenv")

if [ -n "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "Virtual environment activé : $VENV_PATH"
else
    echo "Pas de virtual environment trouvé. Utilisation de l'environnement système."
fi

#Exécuter l'application avec Gunicorn pour la production
exec gunicorn --bind 0.0.0.0:${PORT:-8000} --workers 1 --threads 2 --worker-class uvicorn.workers.UvicornWorker api_test:app #api:app
