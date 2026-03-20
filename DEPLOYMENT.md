# Deployment Guide - Svara TTS API

This guide provides comprehensive instructions for deploying the Svara TTS API with the embedded vLLM engine in a Docker container.

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
  - GPU: NVIDIA GPU with 24GB+ VRAM (e.g., A100, RTX 4090, H100)
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

# Test text-to-speech (streaming, MP3 output)
curl -N -X POST http://localhost:8080/v1/text-to-speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "नमस्ते, मैं स्वरा टीटीएस हूं।",
    "voice": "Hindi (Male)",
    "response_format": "mp3",
    "stream": true
  }' \
  --output audio.mp3

# Test OpenAI-compatible endpoint
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello from Svara!",
    "voice": "en_male",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

## Configuration

### Environment Variables

The `.env` file contains all configurable parameters:

#### vLLM Engine Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_MODEL` | `kenpath/svara-tts-v1` | Hugging Face model repository |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.9` | GPU memory usage (0.0-1.0) |
| `VLLM_MAX_MODEL_LEN` | `4096` | Maximum context length |
| `VLLM_TENSOR_PARALLEL_SIZE` | `1` | Number of GPUs for parallelism |
| `VLLM_DTYPE` | `auto` | Data type: `auto`, `float16`, `bfloat16`, `float32` |
| `VLLM_QUANTIZATION` | (none) | Quantization: `fp8`, `awq`, `gptq`, or empty for none |
| `VLLM_ENFORCE_EAGER` | `false` | Disable CUDA graphs (useful for debugging) |

#### API Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `API_PORT` | `8080` | FastAPI server port |
| `API_HOST` | `0.0.0.0` | FastAPI bind address |
| `TTS_DEVICE` | `cuda` | Device for SNAC decoder: `cuda`, `mps`, `cpu`, or empty for auto-detect |

#### Hugging Face Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | (empty) | Hugging Face API token (for gated models) |

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

### Using Docker Run

```bash
docker run -d \
  --name svara-tts-api \
  --gpus all \
  -p 8080:8080 \
  -e VLLM_MODEL=kenpath/svara-tts-v1 \
  -e VLLM_GPU_MEMORY_UTILIZATION=0.9 \
  -e VLLM_MAX_MODEL_LEN=4096 \
  -v huggingface_cache:/root/.cache/huggingface \
  svara-tts-api:latest
```

### With Quantization (fp8 on H100)

```bash
docker run -d \
  --name svara-tts-api \
  --gpus all \
  -p 8080:8080 \
  -e VLLM_MODEL=kenpath/svara-tts-v1 \
  -e VLLM_QUANTIZATION=fp8 \
  -e VLLM_DTYPE=auto \
  -v huggingface_cache:/root/.cache/huggingface \
  svara-tts-api:latest
```

### Multi-GPU Deployment

```bash
docker run -d \
  --name svara-tts-api \
  --gpus all \
  -p 8080:8080 \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e VLLM_TENSOR_PARALLEL_SIZE=2 \
  -e VLLM_MODEL=kenpath/svara-tts-v1 \
  -v huggingface_cache:/root/.cache/huggingface \
  svara-tts-api:latest
```

### Process Management with Supervisord

The Docker container uses **supervisord** to manage the FastAPI process with automatic restart on failure.

**View process status:**
```bash
# Enter container
docker-compose exec svara-tts-api bash

# Check process status
supervisorctl status

# Restart the service
supervisorctl restart fastapi
```

## API Usage

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/voices` | GET | List available voices |
| `/v1/text-to-speech` | POST | Full-featured TTS (JSON or multipart) |
| `/v1/audio/speech` | POST | OpenAI-compatible TTS |
| `/debug/timing` | GET | Performance timing statistics |
| `/debug/timing/reset` | POST | Reset timing stats |

### 1. Health Check

```bash
curl http://localhost:8080/health
```

**Response:**
```json
{
  "status": "healthy",
  "model": "kenpath/svara-tts-v1",
  "engine": "embedded"
}
```

### 2. Get Voices

```bash
# Get all voices
curl http://localhost:8080/v1/voices

# Filter by model
curl "http://localhost:8080/v1/voices?model_id=svara-tts-v1"
```

**Response:**
```json
{
  "voices": [
    {
      "voice_id": "hi_male",
      "name": "Hindi (Male)",
      "model_id": "svara-tts-v1",
      "gender": "male",
      "description": "Hindi voice with male characteristics"
    }
  ]
}
```

### Available Voice IDs

| Language | Male | Female |
|----------|------|--------|
| Hindi | `hi_male` | `hi_female` |
| English (Indian) | `en_male` | `en_female` |
| Bengali | `bn_male` | `bn_female` |
| Marathi | `mr_male` | `mr_female` |
| Telugu | `te_male` | `te_female` |
| Kannada | `kn_male` | `kn_female` |
| Tamil | `ta_male` | `ta_female` |
| Gujarati | `gu_male` | `gu_female` |
| Malayalam | `ml_male` | `ml_female` |
| Punjabi | `pa_male` | `pa_female` |
| Assamese | `as_male` | `as_female` |
| Bhojpuri | `bho_male` | `bho_female` |
| Magahi | `mag_male` | `mag_female` |
| Chhattisgarhi | `hne_male` | `hne_female` |
| Maithili | `mai_male` | `mai_female` |
| Bodo | `brx_male` | `brx_female` |
| Dogri | `doi_male` | `doi_female` |
| Nepali | `ne_male` | `ne_female` |
| Sanskrit | `sa_male` | `sa_female` |

### 3. Text-to-Speech (`/v1/text-to-speech`)

The full-featured endpoint supporting streaming, multiple formats, generation parameters, and zero-shot voice cloning.

**Streaming (default):**

```bash
curl -N -X POST http://localhost:8080/v1/text-to-speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "नमस्ते दुनिया",
    "voice": "Hindi (Male)",
    "response_format": "opus",
    "stream": true
  }' \
  --output audio.opus
```

**Non-streaming with format conversion:**

```bash
curl -X POST http://localhost:8080/v1/text-to-speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "voice": "English (Female)",
    "response_format": "wav",
    "stream": false
  }' \
  --output audio.wav
```

**With generation parameters:**

```bash
curl -N -X POST http://localhost:8080/v1/text-to-speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a test with custom parameters",
    "voice": "English (Male)",
    "response_format": "mp3",
    "temperature": 0.8,
    "top_p": 0.95,
    "repetition_penalty": 1.2,
    "max_tokens": 4096
  }' \
  --output audio.mp3
```

**Zero-shot voice cloning (JSON with base64 audio):**

```bash
# Encode reference audio to base64
REF_AUDIO=$(base64 -w0 reference.wav)

curl -X POST http://localhost:8080/v1/text-to-speech \
  -H "Content-Type: application/json" \
  -d "{
    \"text\": \"Hello, this should sound like the reference.\",
    \"reference_audio\": \"${REF_AUDIO}\",
    \"reference_transcript\": \"Original transcript of reference audio.\",
    \"response_format\": \"wav\",
    \"stream\": false
  }" \
  --output cloned.wav
```

**Zero-shot voice cloning (multipart form with file upload):**

```bash
curl -X POST http://localhost:8080/v1/text-to-speech \
  -F "text=Hello, this should sound like the reference." \
  -F "reference_audio=@reference.wav" \
  -F "reference_transcript=Original transcript of reference audio." \
  -F "response_format=wav" \
  -F "stream=false" \
  --output cloned.wav
```

#### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | string | Yes | — | Text to synthesize (1-5000 chars) |
| `voice` | string | Conditional | — | Voice name (e.g., `Hindi (Male)`). Required unless using zero-shot. |
| `reference_audio` | string/file | No | `null` | Base64 audio (JSON) or file upload (multipart) for zero-shot cloning |
| `reference_transcript` | string | No | `null` | Transcript of reference audio (improves cloning quality) |
| `model_id` | string | No | `svara-tts-v1` | Model to use |
| `stream` | boolean | No | `true` | Stream audio response |
| `response_format` | string | No | `opus` | Output format: `mp3`, `opus`, `aac`, `wav`, `pcm` |
| `temperature` | float | No | `0.75` | Sampling temperature (0.0-2.0) |
| `top_p` | float | No | `0.9` | Nucleus sampling (0.0-1.0) |
| `top_k` | int | No | `40` | Top-k sampling |
| `repetition_penalty` | float | No | `1.1` | Repetition penalty (1.0-2.0) |
| `max_tokens` | int | No | `2048` | Max tokens to generate (1-4096) |

### 4. OpenAI-Compatible Endpoint (`/v1/audio/speech`)

Drop-in replacement for OpenAI's TTS API. Works with the OpenAI Python SDK and any OpenAI-compatible client.

**curl:**

```bash
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello from Svara TTS!",
    "voice": "en_male",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

**With streaming:**

```bash
curl -N -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Streaming audio response",
    "voice": "hi_female",
    "response_format": "opus",
    "stream": true
  }' \
  --output speech.opus
```

**OpenAI Python SDK:**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="unused",  # required by SDK but not checked by Svara
)

# Non-streaming (default)
response = client.audio.speech.create(
    model="svara-tts-v1",
    voice="hi_female",
    input="नमस्ते, मैं स्वरा हूं।",
    response_format="mp3",
)
response.stream_to_file("output.mp3")

# Streaming
with client.audio.speech.with_streaming_response.create(
    model="svara-tts-v1",
    voice="en_male",
    input="Streaming audio from Svara TTS.",
    response_format="opus",
    extra_body={"stream": True},
) as response:
    response.stream_to_file("streaming_output.opus")
```

**Node.js (OpenAI SDK):**

```javascript
import OpenAI from "openai";
import fs from "fs";

const client = new OpenAI({
  baseURL: "http://localhost:8080/v1",
  apiKey: "unused",
});

const response = await client.audio.speech.create({
  model: "svara-tts-v1",
  voice: "en_female",
  input: "Hello from Node.js!",
  response_format: "mp3",
});

const buffer = Buffer.from(await response.arrayBuffer());
fs.writeFileSync("output.mp3", buffer);
```

#### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input` | string | Yes | — | Text to synthesize (1-5000 chars) |
| `voice` | string | Yes | — | Voice ID (e.g., `hi_male`, `en_female`) |
| `model` | string | No | `svara-tts-v1` | Model name (accepted, not used for selection) |
| `response_format` | string | No | `mp3` | Output format: `mp3`, `opus`, `aac`, `wav`, `pcm` |
| `speed` | float | No | `1.0` | Playback speed 0.5-2.0 (accepted, not yet implemented) |
| `stream` | boolean | No | `false` | Stream audio response |

### 5. Performance Timing

```bash
# Get timing stats
curl http://localhost:8080/debug/timing

# Reset timing stats
curl -X POST http://localhost:8080/debug/timing/reset
```

**Example response:**
```json
{
  "timing_stats": {
    "vLLM.astream": {
      "calls": 5,
      "total_ms": 2340.12,
      "avg_ms": 468.02,
      "min_ms": 312.45,
      "max_ms": 623.78
    },
    "Orchestrator.astream_one": {
      "calls": 5,
      "total_ms": 2890.56,
      "avg_ms": 578.11,
      "min_ms": 398.23,
      "max_ms": 756.34
    }
  },
  "note": "All times in milliseconds"
}
```

### Response Headers

For audio responses:
- `Content-Type`: Appropriate MIME type (`audio/mpeg`, `audio/ogg`, `audio/aac`, `audio/wav`, `audio/pcm`)
- `X-Sample-Rate: 24000`
- `X-Channels: 1`
- `Content-Length`: (non-streaming responses only)

### Python Client Examples

**Streaming with requests:**
```python
import requests

response = requests.post(
    "http://localhost:8080/v1/text-to-speech",
    json={
        "text": "नमस्ते",
        "voice": "Hindi (Male)",
        "response_format": "mp3",
        "stream": True,
    },
    stream=True,
)

with open("output.mp3", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)
```

**Non-streaming with requests:**
```python
import requests

response = requests.post(
    "http://localhost:8080/v1/text-to-speech",
    json={
        "text": "Hello world",
        "voice": "English (Female)",
        "response_format": "wav",
        "stream": False,
    },
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

**Async with httpx:**
```python
import httpx
import asyncio

async def synthesize():
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://localhost:8080/v1/text-to-speech",
            json={
                "text": "Async streaming example",
                "voice": "English (Male)",
                "response_format": "opus",
                "stream": True,
            },
        ) as response:
            with open("output.opus", "wb") as f:
                async for chunk in response.aiter_bytes():
                    f.write(chunk)

asyncio.run(synthesize())
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
```

#### 2. Out of Memory Errors

**Reduce GPU memory utilization:**
```bash
# In .env
VLLM_GPU_MEMORY_UTILIZATION=0.8
VLLM_MAX_MODEL_LEN=2048
```

**Enable quantization (reduces memory ~2x):**
```bash
VLLM_QUANTIZATION=fp8
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
docker-compose run svara-tts-api \
  python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('kenpath/svara-tts-v1')"
```

#### 4. Slow Response Times

**Check timing stats:**
```bash
curl http://localhost:8080/debug/timing
```

**Enable quantization for faster inference:**
```bash
VLLM_QUANTIZATION=fp8
```

**Use multiple GPUs:**
```bash
VLLM_TENSOR_PARALLEL_SIZE=2
```

#### 5. Empty Audio Output

This was a known bug (now fixed). If audio is very short and the prebuffer threshold is never reached, the `flush()` mechanism ensures the audio is still returned. If you still see this:

- Check that the text is not empty
- Check that the voice ID is valid
- Try a longer text to confirm the pipeline works

### Debugging

**Access container shell:**
```bash
docker-compose exec svara-tts-api /bin/bash
```

**Check supervisord status:**
```bash
docker-compose exec svara-tts-api supervisorctl status
```

**View logs:**
```bash
# All logs stream to stdout
docker-compose logs -f

# Restart the service
docker-compose exec svara-tts-api supervisorctl restart fastapi
```

**Test health:**
```bash
curl http://localhost:8080/health
```

**Check available voices:**
```bash
curl http://localhost:8080/v1/voices | python3 -m json.tool
```

## Advanced Configuration

### Custom Model Deployment

```bash
# Use custom model from Hugging Face
VLLM_MODEL=your-username/your-model

# Use local model (mount into container)
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
        proxy_read_timeout 300s;  # Allow long TTS requests
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
# Built into docker-compose.yml:
# healthcheck:
#   test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
#   interval: 30s
#   timeout: 10s
#   retries: 3
#   start_period: 120s
```

**Resource monitoring:**
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor container stats
docker stats svara-tts-api
```

**Performance monitoring:**
```bash
# Get per-function timing stats
curl http://localhost:8080/debug/timing | python3 -m json.tool
```

### Scaling

**Horizontal scaling with load balancer:**

Run multiple instances on different ports with one GPU each:

```yaml
# docker-compose.scale.yml
services:
  svara-tts-api-1:
    extends: svara-tts-api
    ports:
      - "8081:8080"
    environment:
      - CUDA_VISIBLE_DEVICES=0

  svara-tts-api-2:
    extends: svara-tts-api
    ports:
      - "8082:8080"
    environment:
      - CUDA_VISIBLE_DEVICES=1
```

## Support

For issues and questions:
- GitHub Issues: [Repository Issues](https://github.com/your-repo/issues)
- Documentation: [README.md](README.md)
- Architecture: [ARCHITECTURE.md](ARCHITECTURE.md)
- Model Card: [Hugging Face](https://huggingface.co/kenpath/svara-tts-v1)

## License

See [LICENSE](LICENSE) file for details.
