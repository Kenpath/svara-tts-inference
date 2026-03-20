#!/bin/bash
# Development startup script
# vLLM engine is embedded in the FastAPI process -- single process only.

set -e

echo "=============================================="
echo "Starting Svara TTS API (Development Mode)"
echo "=============================================="
echo ""

API_PORT=${API_PORT:-8080}
API_HOST=${API_HOST:-0.0.0.0}

echo "Configuration:"
echo "  Model: ${VLLM_MODEL:-kenpath/svara-tts-v1}"
echo "  API: ${API_HOST}:${API_PORT}"
echo "  dtype: ${VLLM_DTYPE:-auto}"
echo "  quantization: ${VLLM_QUANTIZATION:-none}"
echo ""

cd "$(dirname "$0")/../api"

exec python3 -m uvicorn server:app \
    --host "$API_HOST" \
    --port "$API_PORT" \
    --log-level info
