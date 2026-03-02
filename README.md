# svara-tts-inference

Inference and deployment toolkit for Svara-TTS, an open-source multilingual text-to-speech model for Indic languages — includes examples for local GGUF inference, Gradio demo, and production-ready API deployment.

[![🤗 Hugging Face - svara-tts-v1 Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-black)](https://huggingface.co/kenpath/svara-tts-v1) 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15YxFo1DzdQNbFUIZ1HJA4AN4oHqKxGtg)
[![🤗 Hugging Face - Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-green)](https://huggingface.co/spaces/kenpath/svara-tts)

## Features

- **V1 First, V0.5 Optional**: Default setup targets `svara-tts-v1`; optional `v0.5` mode is also supported
- **19 Indic Languages**: Multilingual synthesis across major Indic languages
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

**Text-to-Speech (v1 model style: language + gender):**
```bash
curl -X POST http://localhost:8080/v1/text-to-speech \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "नमस्ते, मैं स्वरा टीटीएस हूं",
    "language": "hi",
    "gender": "male",
    "stream": true,
    "response_format": "wav"
  }' \
  --output output_v1.wav
```

**Python Example (streaming):**
```python
import requests

response = requests.post(
    "http://localhost:8080/v1/text-to-speech",
    json={
        "transcript": "Hello from Svara TTS",
        "language": "en",
        "gender": "female",
        "stream": True,
        "response_format": "wav",
    },
    stream=True
)

with open("output.wav", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
```

See [examples/api_client.py](examples/api_client.py) for more examples.

## API Documentation

### Endpoints

- `GET /health` - Health check
- `GET /v1/voices` - List available voices for active `.env` model (`VLLM_MODEL`)
- `POST /v1/text-to-speech` - Generate speech from text (`stream=true/false`)

### Model Modes

Set model mode via `.env` `VLLM_MODEL`:

- `kenpath/svara-tts-v1` (v1 mode)
  - Input style: `transcript + language + gender`
  - Voice selection is inferred from language/gender
  - `GET /v1/voices` returns v1 voice IDs (example: `en_male`, `hi_female`)

- `kenpath/voice-svara-tts-v1-fft-v0.5` (v0.5 mode)
  - Input style: `transcript + voice_name` (with language/gender fields still present)
  - Voice selection is explicit using `voice_name`
  - `GET /v1/voices` returns name-based voices (example: `Prakash`, `Aaradhya`)
  - Optional mode for users who need the raw/modal-style voice-name flow

After changing `VLLM_MODEL`, restart/rebuild the service.

### Voice Selection

- For `kenpath/svara-tts-v1`, requests use `language + gender`.
- For `voice-svara-tts-v1-fft-v0.5` (and v2-style voice sets), requests use `voice_name`.
- `GET /v1/voices` requires no params and returns the valid voices for the active model.

Examples:
- v1 voice-like selection: `{"language":"en","gender":"male"}`
- v0.5/v2 voice-name selection: `{"voice_name":"Prakash","language":"en","gender":"male"}`

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
┌───────────────────────────────┐
│ Embedded vLLM AsyncLLMEngine │
│ + SNAC Decoder (CUDA/GPU)    │
└───────────────────────────────┘
```

## Development

### Project Structure

```
svara-tts-inference/
├── api/                    # FastAPI server
│   └── server.py          # Main API endpoints
├── tts_engine/            # Core TTS engine
│   ├── orchestrator.py    # TTS orchestration
│   ├── codec.py           # SNAC codec
│   ├── transports.py      # vLLM transport
│   ├── voice_config.py    # Voice profiles
│   └── utils.py           # Utilities
├── examples/              # Example scripts
│   └── api_client.py      # API client examples
├── Dockerfile             # Docker image
├── docker-compose.yml     # Docker Compose config
└── requirements.txt       # Python dependencies
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

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
