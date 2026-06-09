#!/bin/sh
set -e

exec uvicorn main:app \
  --host 0.0.0.0 \
  --port "${UVICORN_PORT:-8005}" \
  "$@"
