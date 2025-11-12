# Deployment Guide - Svara TTS API

This guide provides comprehensive instructions for deploying the Svara TTS API with vLLM and SNAC in a Docker container.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Building the Image](#building-the-image)
- [Running the Container](#running-the-container)
- [API Usage](#api-usage)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

## Prerequisites

### Required Software

1. **Docker** (version 20.10 or later)
   ```bash
   docker --version
   ```

2. **Docker Compose** (version 2.0 or later)
   ```bash
   docker-compose --version
   ```

3. **NVIDIA GPU Drivers** (for GPU acceleration)
   ```bash
   nvidia-smi
   ```

4. **NVIDIA Container Toolkit**
   ```bash
   # Install on Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

### Hardware Requirements

- **Minimum**:
  - GPU: NVIDIA GPU with 16GB VRAM (e.g., Tesla T4, RTX 4070)
  - RAM: 16GB system RAM
  - Storage: 50GB free space

- **Recommended**:
  - GPU: NVIDIA GPU with 24GB+ VRAM (e.g., A100, RTX 4090)
  - RAM: 32GB system RAM
  - Storage: 100GB free space (for model cache)

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd svara-tts-inference
```

### 2. Configure Environment Variables

```bash
# Copy example configuration
cp .env.example .env

# Edit configuration (optional)
nano .env
```

### 3. Build and Run

```bash
# Build the Docker image
docker-compose build

# Start the service
docker-compose up -d

# Check logs
docker-compose logs -f
```

### 4. Verify Deployment

```bash
# Check health
curl http://localhost:8080/health

# List available voices
curl http://localhost:8080/v1/voices

# Test text-to-speech (streaming)
curl -X POST http://localhost:8080/v1/text-to-speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "नमस्ते, मैं स्वरा टीटीएस हूं।",
    "voice_id": "hi_male",
    "stream": true
  }' \
  --output audio.pcm
```

## Configuration

### Environment Variables

The `.env` file contains all configurable parameters. Key variables:

#### vLLM Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_MODEL` | `kenpath/svara-tts-v1` | Hugging Face model repository |
| `VLLM_PORT` | `8000` | vLLM server port |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.9` | GPU memory usage (0.0-1.0) |
| `VLLM_MAX_MODEL_LEN` | `2048` | Maximum context length |
| `VLLM_TENSOR_PARALLEL_SIZE` | `1` | Number of GPUs for parallelism |

#### API Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `API_PORT` | `8080` | FastAPI server port |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM endpoint URL |
| `TTS_DEVICE` | `cuda` | Device for SNAC decoder |

#### Hugging Face Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | (empty) | Hugging Face API token |

## Building the Image

### Standard Build

```bash
docker-compose build
```

### Build with Custom Tag

```bash
docker build -t svara-tts-api:v1.0.0 .
```

### Build for Different Architecture

```bash
docker buildx build --platform linux/amd64 -t svara-tts-api:latest .
```

## Running the Container

### Using Docker Compose (Recommended)

```bash
# Start in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Process Management with Supervisord

The Docker container uses **supervisord** for robust process management. This provides:

- **Automatic restart** on process failure
- **Better logging** with log rotation
- **Health monitoring** for both vLLM and FastAPI
- **Graceful shutdown** handling

**View supervisord status:**
```bash
# Enter container
docker-compose exec svara-tts-api bash

# Check process status
supervisorctl status

# View logs
tail -f /var/log/supervisor/vllm.log
tail -f /var/log/supervisor/fastapi.log

# Restart a service
supervisorctl restart vllm
supervisorctl restart fastapi

# Stop/start all services
supervisorctl stop all
supervisorctl start all
```

**Process startup order:**
1. vLLM server starts first (priority 100)
2. FastAPI starts after vLLM is ready (priority 200)
3. Health check monitor runs continuously (priority 300)

### Using Docker Run

```bash
docker run -d \
  --name svara-tts-api \
  --gpus all \
  -p 8000:8000 \
  -p 8080:8080 \
  -e VLLM_MODEL=kenpath/svara-tts-v1 \
  -e VLLM_GPU_MEMORY_UTILIZATION=0.9 \
  -v huggingface_cache:/root/.cache/huggingface \
  svara-tts-api:latest
```

### Multi-GPU Deployment

```bash
# Use specific GPUs
docker-compose run \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e VLLM_TENSOR_PARALLEL_SIZE=2 \
  svara-tts-api

# Or modify docker-compose.yml:
# environment:
#   - CUDA_VISIBLE_DEVICES=0,1
#   - VLLM_TENSOR_PARALLEL_SIZE=2
```

## API Usage

### Endpoints

#### 1. Health Check

```bash
curl http://localhost:8080/health
```

**Response:**
```json
{
  "status": "healthy",
  "model": "kenpath/svara-tts-v1",
  "vllm_url": "http://localhost:8000/v1"
}
```

#### 2. Get Voices

```bash
# Get all voices
curl http://localhost:8080/v1/voices

# Filter by model
curl http://localhost:8080/v1/voices?model_id=svara-tts-v1
```

**Response:**
```json
{
  "voices": [
    {
      "voice_id": "hi_male",
      "name": "Hindi (Male)",
      "language_code": "hi",
      "model_id": "svara-tts-v1",
      "gender": "male",
      "description": "Hindi voice with male characteristics"
    },
    ...
  ]
}
```

#### 3. Text-to-Speech

**Streaming (default):**

```bash
curl -X POST http://localhost:8080/v1/text-to-speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "नमस्ते दुनिया",
    "voice_id": "hi_male",
    "model_id": "svara-tts-v1",
    "stream": true
  }' \
  --output audio.pcm
```

**Non-streaming:**

```bash
curl -X POST http://localhost:8080/v1/text-to-speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "voice_id": "en_female",
    "model_id": "svara-tts-v1",
    "stream": false
  }' \
  --output audio.pcm
```

### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | string | Yes | - | Text to synthesize (1-5000 chars) |
| `voice_id` | string | Yes | - | Voice identifier (e.g., "hi_male") |
| `model_id` | string | No | `svara-tts-v1` | Model to use |
| `language_code` | string | No | (from voice) | Override language code |
| `voice_settings` | object | No | `{}` | Voice settings (not implemented) |
| `text_normalization` | boolean | No | `false` | Text normalization (not implemented) |
| `reference_audio` | bytes | No | `null` | Reference audio (not implemented) |
| `stream` | boolean | No | `true` | Stream audio response |

### Response Headers

For audio responses:
- `Content-Type: audio/pcm`
- `X-Sample-Rate: 24000`
- `X-Bit-Depth: 16`
- `X-Channels: 1`

### Converting PCM to WAV/MP3

```bash
# PCM to WAV using ffmpeg
ffmpeg -f s16le -ar 24000 -ac 1 -i audio.pcm audio.wav

# PCM to MP3 using ffmpeg
ffmpeg -f s16le -ar 24000 -ac 1 -i audio.pcm -b:a 192k audio.mp3

# Play directly with ffplay
ffplay -f s16le -ar 24000 -ac 1 audio.pcm
```

### Python Client Example

See [`examples/api_client.py`](examples/api_client.py) for a complete Python client implementation.

```python
import requests

# Get voices
response = requests.get("http://localhost:8080/v1/voices")
voices = response.json()["voices"]

# Text-to-speech (streaming)
response = requests.post(
    "http://localhost:8080/v1/text-to-speech",
    json={
        "text": "नमस्ते",
        "voice_id": "hi_male",
        "stream": True
    },
    stream=True
)

with open("output.pcm", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)
```

## Troubleshooting

### Common Issues

#### 1. Container Fails to Start

**Check GPU availability:**
```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base nvidia-smi
```

**Check logs:**
```bash
docker-compose logs -f
cat /tmp/vllm.log  # Inside container
cat /tmp/api.log   # Inside container
```

#### 2. Out of Memory Errors

**Reduce GPU memory utilization:**
```bash
# In .env
VLLM_GPU_MEMORY_UTILIZATION=0.8
VLLM_MAX_MODEL_LEN=1024
```

**Check memory usage:**
```bash
nvidia-smi
```

#### 3. Model Download Issues

**Check Hugging Face token:**
```bash
# For gated models, set HF_TOKEN in .env
HF_TOKEN=hf_xxxxxxxxxxxx
```

**Manual model download:**
```bash
# Pre-download model
docker-compose run svara-tts-api \
  python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('kenpath/svara-tts-v1')"
```

#### 4. Audio Quality Issues

**Adjust decoding parameters:**
```python
# In api/server.py, modify orchestrator settings:
hop_samples=512      # Lower for faster response
prebuffer_seconds=1.5  # Higher for smoother audio
```

#### 5. Slow Response Times

**Enable concurrent decoding:**
```python
concurrent_decode=True
max_workers=4  # Increase workers
```

**Use multiple GPUs:**
```bash
VLLM_TENSOR_PARALLEL_SIZE=2
```

### Debugging

**Access container shell:**
```bash
docker-compose exec svara-tts-api /bin/bash
```

**Check supervisord status:**
```bash
docker-compose exec svara-tts-api supervisorctl status
```

**View process logs:**
```bash
# All logs
docker-compose exec svara-tts-api tail -f /var/log/supervisor/*.log

# Specific service
docker-compose exec svara-tts-api tail -f /var/log/supervisor/vllm.log
docker-compose exec svara-tts-api tail -f /var/log/supervisor/fastapi.log
```

**Check vLLM server:**
```bash
curl http://localhost:8000/v1/models
```

**Test vLLM directly:**
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kenpath/svara-tts-v1",
    "prompt": "Test",
    "max_tokens": 10
  }'
```

**Restart services without container restart:**
```bash
# Restart vLLM
docker-compose exec svara-tts-api supervisorctl restart vllm

# Restart FastAPI
docker-compose exec svara-tts-api supervisorctl restart fastapi

# Restart all
docker-compose exec svara-tts-api supervisorctl restart all
```

## Advanced Configuration

### Custom Model Deployment

```bash
# Use custom model from Hugging Face
VLLM_MODEL=your-username/your-model

# Use local model
docker run -v /path/to/model:/model \
  -e VLLM_MODEL=/model \
  svara-tts-api:latest
```

### Production Deployment

**Using a reverse proxy (nginx):**

```nginx
upstream svara-api {
    server localhost:8080;
}

server {
    listen 80;
    server_name api.example.com;
    
    location / {
        proxy_pass http://svara-api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;  # Important for streaming
    }
}
```

**SSL/TLS with certbot:**

```bash
certbot --nginx -d api.example.com
```

### Monitoring

**Health checks:**
```bash
# Add to docker-compose.yml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
  interval: 30s
  timeout: 10s
  retries: 3
```

**Resource monitoring:**
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor container stats
docker stats svara-tts-api
```

### Scaling

**Horizontal scaling with load balancer:**

Run multiple instances on different ports and use nginx for load balancing:

```yaml
# docker-compose.scale.yml
services:
  svara-tts-api-1:
    extends: svara-tts-api
    ports:
      - "8081:8080"
  
  svara-tts-api-2:
    extends: svara-tts-api
    ports:
      - "8082:8080"
```

## Support

For issues and questions:
- GitHub Issues: [Repository Issues](https://github.com/your-repo/issues)
- Documentation: [README.md](README.md)
- Model Card: [Hugging Face](https://huggingface.co/kenpath/svara-tts-v1)

## License

See [LICENSE](LICENSE) file for details.

