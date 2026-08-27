#!/usr/bin/env bash
set -e

uvicorn src.main:app --host 0.0.0.0 --port 8000 &
streamlit run src/web/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true &
# nginx.conf's `pid` directive is rewritten to /tmp/nginx.pid at build time
# (see the sed step in Dockerfile) so nginx can write its pid file as uid 1000.
nginx -g "daemon off;" &

wait -n
exit $?
