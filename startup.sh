#!/bin/bash
#Active le VirtualEnv créé par Oryx
if [ -d "antenv/bin" ]; then
  source antenv/bin/activate
elif [ -d "/home/site/wwwroot/antenv/bin" ]; then
  source /home/site/wwwroot/antenv/bin/activate
elif [ -d "/tmp/8dde94fb1e7b0e7/antenv/bin" ]; then
  source /tmp/8dde94fb1e7b0e7/antenv/bin/activate
fi

#Lance FastAPI via uvicorn
exec uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000} --reload --log-level debug
