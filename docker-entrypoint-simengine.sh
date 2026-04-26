#!/bin/bash
# docker-entrypoint-simengine.sh
# Ray クラスタの head / worker 自動起動

set -e

RAY_MODE="${RAY_MODE:-standalone}"   # standalone | head | worker
RAY_HEAD_ADDR="${RAY_HEAD_ADDR:-}"
RAY_NUM_GPUS="${RAY_NUM_GPUS:-1}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_REDIS_PORT="${RAY_REDIS_PORT:-6379}"

case "$RAY_MODE" in
  head)
    echo "[SimEngine] Starting Ray HEAD node (port ${RAY_REDIS_PORT})"
    ray start --head \
        --port="${RAY_REDIS_PORT}" \
        --dashboard-port="${RAY_DASHBOARD_PORT}" \
        --num-gpus="${RAY_NUM_GPUS}" \
        --block &
    sleep 5
    echo "[SimEngine] Ray HEAD ready at $(hostname -I | awk '{print $1}'):${RAY_REDIS_PORT}"
    ;;
  worker)
    if [ -z "$RAY_HEAD_ADDR" ]; then
        echo "[ERROR] RAY_HEAD_ADDR must be set for worker mode"
        exit 1
    fi
    echo "[SimEngine] Joining Ray cluster at ${RAY_HEAD_ADDR}"
    ray start \
        --address="${RAY_HEAD_ADDR}:${RAY_REDIS_PORT}" \
        --num-gpus="${RAY_NUM_GPUS}" \
        --block &
    sleep 5
    ;;
  standalone)
    echo "[SimEngine] Standalone mode (no Ray auto-start)"
    ;;
esac

exec "$@"
