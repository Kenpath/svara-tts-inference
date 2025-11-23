# Svara TTS API - Production Dockerfile
# Multi-stage build for vLLM + SNAC + FastAPI deployment
# CUDA 12.8 for NVIDIA Blackwell GPUs (RTX 5090)

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install system dependencies including Python 3.11
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    curl \
    libsndfile1 \
    ffmpeg \
    supervisor \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip and install uv for faster package management
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && pip3 install uv

# Set working directory
WORKDIR /app

# ============================================================================
# Stage 1: Install PyTorch with CUDA 12.8 support
# ============================================================================
FROM base AS pytorch-builder

# Install PyTorch with CUDA 12.8 support
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128

# ============================================================================
# Stage 2: Install vLLM with CUDA 12.8 support
# ============================================================================
FROM pytorch-builder AS vllm-builder

# Set environment variables for faster compilation if building from source
ENV MAX_JOBS=4
ENV NVCC_THREADS=4
ENV TORCH_CUDA_ARCH_LIST="8.9;9.0"

# Install vLLM (will use the PyTorch with CUDA 12.8 we already installed)
# Try to get pre-built wheel first, if not available it will build from source
RUN pip3 install --no-build-isolation vllm || pip3 install vllm

# ============================================================================
# Stage 3: Install application dependencies
# ============================================================================
FROM vllm-builder AS app-deps

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Install additional dependencies for audio processing
RUN pip3 install soundfile numpy

# ============================================================================
# Stage 4: Final application image
# ============================================================================
FROM app-deps AS final

# Copy application code
COPY tts_engine/ ./tts_engine/
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY supervisord.conf /etc/supervisor/conf.d/svara-tts.conf

# Make scripts executable
RUN chmod +x ./scripts/*.sh

# Create directories for logs and cache
RUN mkdir -p /var/log/supervisor /root/.cache/huggingface

# Expose ports
# 8000: vLLM server
# 8080: FastAPI server
EXPOSE 8000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start supervisord to manage all processes
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/svara-tts.conf"]

