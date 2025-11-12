#!/bin/bash
# Development startup script (uses the old bash-based approach)
# For production, use Docker with supervisord instead

set -e

echo "=============================================="
echo "Starting Svara TTS API (Development Mode)"
echo "=============================================="
echo ""
echo "⚠️  This is for development only!"
echo "   For production, use Docker with supervisord"
echo ""

# Configuration
VLLM_MODEL=${VLLM_MODEL:-kenpath/svara-tts-v1}
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_HOST=${VLLM_HOST:-0.0.0.0}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.9}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-2048}
VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE:-1}

API_PORT=${API_PORT:-8080}
API_HOST=${API_HOST:-0.0.0.0}

echo "Configuration:"
echo "  vLLM Model: $VLLM_MODEL"
echo "  vLLM Port: $VLLM_PORT"
echo "  API Port: $API_PORT"
echo ""

# Function to check if vLLM is ready
wait_for_vllm() {
    echo "Waiting for vLLM server to be ready..."
    local max_attempts=60
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        # Try both health endpoints
        if curl -s http://localhost:${VLLM_PORT}/health > /dev/null 2>&1 || \
           curl -s http://localhost:${VLLM_PORT}/v1/models > /dev/null 2>&1; then
            echo "✓ vLLM server is ready!"
            return 0
        fi
        
        attempt=$((attempt + 1))
        echo "  Attempt $attempt/$max_attempts..."
        
        # Show last 10 lines of log every 10 attempts
        if [ $((attempt % 10)) -eq 0 ]; then
            echo ""
            echo "  Last 10 lines of vLLM log:"
            tail -n 10 /tmp/vllm.log | sed 's/^/    /'
            echo ""
        fi
        
        sleep 5
    done
    
    echo "✗ Error: vLLM server did not become ready in time"
    return 1
}

# Function to handle shutdown
cleanup() {
    echo ""
    echo "=============================================="
    echo "Shutting down..."
    echo "=============================================="
    
    if [ ! -z "$VLLM_PID" ]; then
        echo "Stopping vLLM server (PID: $VLLM_PID)..."
        kill $VLLM_PID 2>/dev/null || true
        wait $VLLM_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$API_PID" ]; then
        echo "Stopping FastAPI server (PID: $API_PID)..."
        kill $API_PID 2>/dev/null || true
        wait $API_PID 2>/dev/null || true
    fi
    
    echo "✓ Shutdown complete"
    exit 0
}

trap cleanup SIGTERM SIGINT SIGQUIT

# Start vLLM
echo "Starting vLLM server..."
python3 -m vllm.entrypoints.openai.api_server \
    --model "$VLLM_MODEL" \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" \
    --trust-remote-code \
    --dtype auto \
    > /tmp/vllm.log 2>&1 &

VLLM_PID=$!
echo "✓ vLLM server started (PID: $VLLM_PID)"

# Wait for vLLM
if ! wait_for_vllm; then
    echo "vLLM logs:"
    tail -n 50 /tmp/vllm.log
    exit 1
fi

# Start FastAPI
echo "Starting FastAPI server..."
cd api
python3 -m uvicorn server:app \
    --host "$API_HOST" \
    --port "$API_PORT" \
    --log-level info \
    > /tmp/api.log 2>&1 &

API_PID=$!
echo "✓ FastAPI server started (PID: $API_PID)"

echo ""
echo "=============================================="
echo "✓ Development server is ready!"
echo "=============================================="
echo "  API: http://localhost:${API_PORT}"
echo "  Press Ctrl+C to stop"
echo "=============================================="

wait $VLLM_PID $API_PID

