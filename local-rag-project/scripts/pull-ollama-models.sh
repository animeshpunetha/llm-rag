#!/bin/sh
set -e

OLLAMA_HOST="${OLLAMA_HOST:-http://ollama:11434}"
LLM_MODEL="${LLM_MODEL:-qwen2.5:7b}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-nomic-embed-text}"

echo "Pulling Ollama models via ${OLLAMA_HOST}..."

curl -sf --max-time 3600 "${OLLAMA_HOST}/api/pull" -d "{\"name\": \"${LLM_MODEL}\"}"
echo ""
curl -sf --max-time 3600 "${OLLAMA_HOST}/api/pull" -d "{\"name\": \"${EMBEDDING_MODEL}\"}"
echo ""

echo "Model pull complete."
