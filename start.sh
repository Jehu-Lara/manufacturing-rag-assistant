#!/usr/bin/env bash
set -e

uvicorn api.main:app --host 0.0.0.0 --port 8000 &
streamlit run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true &
nginx -g "daemon off;" &

wait -n
exit $?
