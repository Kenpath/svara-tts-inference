# Svara TTS API - Production Dockerfile
# Multi-stage build for vLLM + SNAC + FastAPI deployment

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    libsndfile1 \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# ============================================================================
# Stage 1: Install vLLM with CUDA support
# ============================================================================
FROM base AS vllm-builder

# Install vLLM with CUDA support
RUN pip3 install vllm==0.6.3.post1

# ============================================================================
# Stage 2: Install application dependencies
# ============================================================================
FROM vllm-builder AS app-deps

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Install additional dependencies for audio processing
RUN pip3 install soundfile numpy

# ============================================================================
# Stage 3: Final application image
# ============================================================================
FROM app-deps AS final

# Copy application code
COPY tts_engine/ ./tts_engine/
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY supervisord.conf /etc/supervisor/conf.d/svara-tts.conf

# Make scripts executable
RUN chmod +x ./scripts/*.sh ./scripts/*.py

# Create directories for logs and cache
RUN mkdir -p /var/log/supervisor /root/.cache/huggingface

# Expose ports
# 8000: vLLM server
# 8080: FastAPI server
EXPOSE 8000 8080

# Set default environment variables
ENV VLLM_MODEL=kenpath/svara-tts-v1 \
    VLLM_PORT=8000 \
    VLLM_HOST=0.0.0.0 \
    VLLM_GPU_MEMORY_UTILIZATION=0.9 \
    VLLM_MAX_MODEL_LEN=2048 \
    VLLM_BASE_URL=http://localhost:8000/v1 \
    API_PORT=8080 \
    API_HOST=0.0.0.0 \
    TTS_DEVICE=cuda

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start supervisord to manage all processes
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/svara-tts.conf"]

