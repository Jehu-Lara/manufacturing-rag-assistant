#!/usr/bin/env bash
set -Eeuo pipefail

uvicorn src.main:app --host 127.0.0.1 --port 8000 &
api_pid=$!

# `python -m streamlit`, not the `streamlit` console script: only `-m` puts the
# WORKDIR (/app) on sys.path, which the app's `from src.web import client` needs
# (the project is not pip-installed in the image). The console script would leave
# only src/web/ on the path and the UI would crash with ModuleNotFoundError.
python -m streamlit run src/web/app.py \
    --server.port 8501 \
    --server.address 127.0.0.1 \
    --server.headless true &
web_pid=$!

# nginx.conf's pid directive is rewritten to /tmp/nginx.pid at build time.
nginx -g "daemon off;" &
nginx_pid=$!

shutting_down=0

shutdown() {
    local status="$1"
    if (( shutting_down )); then
        return
    fi
    shutting_down=1
    trap - TERM INT

    for pid in "$api_pid" "$web_pid" "$nginx_pid"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    wait "$api_pid" "$web_pid" "$nginx_pid" 2>/dev/null || true
    exit "$status"
}

trap 'shutdown 143' TERM
trap 'shutdown 130' INT

set +e
wait -n "$api_pid" "$web_pid" "$nginx_pid"
status=$?
set -e

# Any child exit leaves the combined service incomplete. A clean child exit
# is still unexpected and must fail the container.
if (( status == 0 )); then
    status=1
fi
shutdown "$status"
