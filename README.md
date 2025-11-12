# svara-tts-inference

Inference and deployment toolkit for Svara-TTS, an open-source multilingual text-to-speech model for Indic languages — includes examples for local GGUF inference, Gradio demo, and production-ready API deployment.

[![🤗 Hugging Face - svara-tts-v1 Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-black)](https://huggingface.co/kenpath/svara-tts-v1) 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15YxFo1DzdQNbFUIZ1HJA4AN4oHqKxGtg)
[![🤗 Hugging Face - Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-green)](https://huggingface.co/spaces/kenpath/svara-tts)

## Features

- **38 Voice Profiles**: Support for 19 Indian languages with male and female voices
- **Streaming Audio**: Real-time audio generation with low-latency streaming
- **Production Ready**: Docker deployment with vLLM and FastAPI
- **GPU Accelerated**: CUDA-optimized inference with SNAC decoder
- **API Compatible**: ElevenLabs-style REST API for easy integration

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

**Text-to-Speech:**
```bash
curl -X POST http://localhost:8080/v1/text-to-speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "नमस्ते, मैं स्वरा टीटीएस हूं",
    "voice_id": "hi_male",
    "stream": true
  }' \
  --output audio.pcm

# Convert to WAV
ffmpeg -f s16le -ar 24000 -ac 1 -i audio.pcm output.wav
```

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:8080/v1/text-to-speech",
    json={
        "text": "Hello from Svara TTS",
        "voice_id": "en_female",
        "stream": True
    },
    stream=True
)

with open("output.pcm", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
```

See [examples/api_client.py](examples/api_client.py) for more examples.

## API Documentation

### Endpoints

- `GET /health` - Health check
- `GET /v1/voices` - List available voices
- `POST /v1/text-to-speech` - Generate speech from text

### Voice IDs

For `svara-tts-v1`, voice IDs follow the format `{language_code}_{gender}`:
- Hindi: `hi_male`, `hi_female`
- English: `en_male`, `en_female`
- Bengali: `bn_male`, `bn_female`
- [See full list in DEPLOYMENT.md](DEPLOYMENT.md)

For `svara-tts-v2` (coming soon): `rohit`, `priya`, `arjun`, etc.

## Deployment Guide

For detailed deployment instructions, configuration options, and troubleshooting:

**📖 [Read the Full Deployment Guide →](DEPLOYMENT.md)**

Topics covered:
- Prerequisites and hardware requirements
- Docker configuration
- Environment variables
- Production deployment with nginx
- Troubleshooting and monitoring
- Multi-GPU setup

## Architecture

```
┌─────────────────┐
│   FastAPI       │  Port 8080
│   API Server    │  
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   vLLM Server   │  Port 8000
│   (LLM Engine)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SNAC Decoder   │  CUDA/GPU
│  (Audio Gen)    │
└─────────────────┘
```

## Development

### Project Structure

```
svara-tts-inference/
├── api/                    # FastAPI server
│   └── server.py          # Main API endpoints
├── tts_engine/            # Core TTS engine
│   ├── orchestrator.py    # TTS orchestration
│   ├── decoder_snac.py    # SNAC decoder
│   ├── transports.py      # vLLM transport
│   ├── voice_config.py    # Voice profiles
│   └── utils.py           # Utilities
├── examples/              # Example scripts
│   └── api_client.py      # API client examples
├── scripts/               # Deployment scripts
│   └── start.sh          # Container startup
├── Dockerfile             # Docker image
├── docker-compose.yml     # Docker Compose config
└── requirements.txt       # Python dependencies
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start vLLM server separately
python -m vllm.entrypoints.openai.api_server \
  --model kenpath/svara-tts-v1 \
  --port 8000

# Start FastAPI server
cd api
python server.py
```

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

- 🤗 [Model on Hugging Face](https://huggingface.co/kenpath/svara-tts-v1)
- 🚀 [Try Demo on Hugging Face Spaces](https://huggingface.co/spaces/kenpath/svara-tts)
- 📓 [Colab Notebook](https://colab.research.google.com/drive/15YxFo1DzdQNbFUIZ1HJA4AN4oHqKxGtg)
