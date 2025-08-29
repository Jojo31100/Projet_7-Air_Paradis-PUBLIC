#!/bin/bash
#Installe les dépendances
pip install -r requirements.txt

#Lance FastAPI via uvicorn
exec python -m uvicorn api:app --host 0.0.0.0 --port 8000
