# svara-tts-inference

Inference and deployment toolkit for Svara-TTS, an open-source multilingual text-to-speech model for Indic languages — includes examples for local GGUF inference, Gradio demo, and production-ready API deployment.

[![🤗 Hugging Face - svara-tts-v1 Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-black)](https://huggingface.co/kenpath/svara-tts-v1)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15YxFo1DzdQNbFUIZ1HJA4AN4oHqKxGtg)
[![🤗 Hugging Face - Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-green)](https://huggingface.co/spaces/kenpath/svara-tts)

## Features

- **38 Voice Profiles**: Support for 19 Indian languages with male and female voices
- **Streaming Audio**: Real-time audio generation with low-latency streaming
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's `/v1/audio/speech` endpoint
- **Production Ready**: Docker deployment with embedded vLLM engine
- **GPU Accelerated**: CUDA-optimized inference with configurable SNAC decoder device
- **Multiple Audio Formats**: Output in MP3, Opus, AAC, WAV, or raw PCM via ffmpeg
- **Zero-Shot Voice Cloning**: Clone any voice with a short audio reference
- **Long-Text Chunking**: Automatic sentence-boundary splitting with crossfade stitching

## Supported Languages

Hindi, Bengali, Marathi, Telugu, Kannada, Bhojpuri, Magahi, Chhattisgarhi, Maithili, Assamese, Bodo, Dogri, Gujarati, Malayalam, Punjabi, Tamil, English (Indian), Nepali, Sanskrit

## Quick Start - API Deployment

Deploy Svara TTS as a production API service with Docker:

```bash
# Clone repository
git clone <repository-url>
cd svara-tts-inference

# Configure (optional)
cp .env.example .env

# Build and start
docker-compose up -d

# Test the API
curl http://localhost:8080/health
curl http://localhost:8080/v1/voices
```

### API Usage

**Get Available Voices:**
```bash
curl http://localhost:8080/v1/voices
```

**OpenAI-Compatible Endpoint:**
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

**Streaming:**
```bash
curl -N -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "नमस्ते, मैं स्वरा टीटीएस हूं",
    "voice": "hi_male",
    "response_format": "wav",
    "stream": true
  }' \
  --output audio.wav
```

**Python Example (OpenAI SDK):**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")

response = client.audio.speech.create(
    model="svara-tts-v1",
    voice="hi_female",
    input="नमस्ते, मैं स्वरा हूं।",
    response_format="mp3",
)

response.stream_to_file("output.mp3")
```

See [examples/api_client.py](examples/api_client.py) for more examples.

## API Documentation

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/voices` | GET | List available voices |
| `/v1/audio/speech` | POST | OpenAI-compatible TTS (supports streaming, zero-shot cloning) |

### Voice IDs

For `svara-tts-v1`, voice IDs follow the format `{language_code}_{gender}`:
- Hindi: `hi_male`, `hi_female`
- English: `en_male`, `en_female`
- Bengali: `bn_male`, `bn_female`
- [See full list in DEPLOYMENT.md](DEPLOYMENT.md)

The `/v1/audio/speech` endpoint also accepts display names like `Hindi (Male)`, `English (Female)`.

### Audio Formats

All endpoints support multiple output formats via the `response_format` parameter:

| Format | MIME Type | Notes |
|--------|-----------|-------|
| `mp3` | `audio/mpeg` | Default |
| `opus` | `audio/ogg` | Great for streaming |
| `aac` | `audio/aac` | ADTS container |
| `wav` | `audio/wav` | Uncompressed, larger files |
| `pcm` | `audio/pcm` | Raw signed 16-bit LE, 24kHz mono |

## Deployment Guide

For detailed deployment instructions, configuration options, and troubleshooting:

**[Read the Full Deployment Guide](DEPLOYMENT.md)**

Topics covered:
- Prerequisites and hardware requirements
- Docker configuration
- Environment variables
- Production deployment with nginx
- Troubleshooting and monitoring
- Multi-GPU setup

## Architecture

The server runs as a **single process** with the vLLM engine embedded directly in the FastAPI application. This eliminates the HTTP hop between API server and LLM engine, reducing latency and operational complexity.

```
┌──────────────────────────────────┐
│         FastAPI Server           │  Port 8080
│                                  │
│  ┌────────────────────────────┐  │
│  │   Embedded vLLM Engine     │  │
│  │   (AsyncLLMEngine)         │  │
│  └──────────┬─────────────────┘  │
│             │                    │
│  ┌──────────▼─────────────────┐  │
│  │   SNAC Decoder             │  │
│  │   Token → PCM Audio        │  │
│  │   (SNAC_DEVICE: cpu/cuda)  │  │
│  └──────────┬─────────────────┘  │
│             │                    │
│  ┌──────────▼─────────────────┐  │
│  │   ffmpeg (format convert)  │  │
│  │   PCM → MP3/Opus/WAV/AAC  │  │
│  └────────────────────────────┘  │
└──────────────────────────────────┘
```

For detailed architecture documentation, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Development

### Project Structure

```
svara-tts-inference/
├── api/                    # FastAPI server
│   ├── server.py           # Main API endpoints + engine init
│   └── models.py           # Pydantic request/response models
├── tts_engine/             # Core TTS engine
│   ├── orchestrator.py     # TTS pipeline orchestration
│   ├── transports.py       # Embedded vLLM transport
│   ├── buffers.py          # Audio prebuffering + crossfade
│   ├── mapper.py           # Token-to-SNAC mapping
│   ├── codec.py            # SNAC encoder/decoder
│   ├── voice_config.py     # Voice profiles
│   ├── encoder.py          # Text-to-token encoding
│   ├── constants.py        # Token IDs and special tokens
│   └── utils.py            # Utilities (chunking, audio processing)
├── assets/                 # Voice config YAML files
├── examples/               # Example scripts
│   └── api_client.py       # API client examples
├── Dockerfile              # Docker image
├── docker-compose.yml      # Docker Compose config
├── supervisord.conf        # Process manager config
├── requirements.txt        # Python dependencies
└── .env.example            # Environment variable template
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment (optional)
cp .env.example .env

# Start the server (vLLM engine starts embedded)
cd api && python server.py
```

The vLLM engine initializes in-process during FastAPI startup — no separate vLLM server needed.

## Requirements

### Hardware
- GPU: NVIDIA GPU with 16GB+ VRAM (recommended: 24GB+)
- RAM: 16GB+ system RAM
- Storage: 50GB+ free space

### Software
- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA GPU Drivers
- NVIDIA Container Toolkit

## License

See [LICENSE](LICENSE) file for details.

## Citation

If you use Svara TTS in your research, please cite:

```bibtex
@misc{svara-tts-v1,
  title={Svara TTS: Multilingual Text-to-Speech for Indic Languages},
  author={Kenpath},
  year={2024},
  url={https://huggingface.co/kenpath/svara-tts-v1}
}
```

## Links

- [Model on Hugging Face](https://huggingface.co/kenpath/svara-tts-v1)
- [Try Demo on Hugging Face Spaces](https://huggingface.co/spaces/kenpath/svara-tts)
- [Colab Notebook](https://colab.research.google.com/drive/15YxFo1DzdQNbFUIZ1HJA4AN4oHqKxGtg)
