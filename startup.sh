#!/bin/bash
#Active le VirtualEnv créé par Oryx
source antenv/bin/activate

#Lance FastAPI via uvicorn
exec uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}
