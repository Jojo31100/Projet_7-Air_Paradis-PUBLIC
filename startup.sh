#!/bin/bash
#Lance FastAPI via uvicorn
exec python -m uvicorn api:app --host 0.0.0.0 --port 8000
