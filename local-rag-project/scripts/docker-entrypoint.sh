#!/bin/sh
set -e

REDIS_HOST="${REDIS_HOST:-semantic-cache}"
REDIS_PORT="${REDIS_PORT:-6379}"
OLLAMA_HOST="${OLLAMA_HOST:-http://ollama:11434}"

echo "Waiting for Redis at ${REDIS_HOST}:${REDIS_PORT}..."
until python -c "
import redis
redis.Redis(host='${REDIS_HOST}', port=${REDIS_PORT}).ping()
" 2>/dev/null; do
  sleep 2
done
echo "Redis is ready."

echo "Waiting for Ollama at ${OLLAMA_HOST}..."
until curl -sf "${OLLAMA_HOST}/api/tags" >/dev/null; do
  sleep 2
done
echo "Ollama is ready."

exec "$@"
