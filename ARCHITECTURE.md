# Svara TTS API - Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Container                        │
│                                                             │
│  ┌────────────────────────────────────────────────────┐   │
│  │              Supervisord                           │   │
│  │         (Process Manager)                          │   │
│  └───┬─────────────┬─────────────┬────────────────────┘   │
│      │             │             │                         │
│      │             │             │                         │
│  ┌───▼───────┐ ┌───▼────────┐ ┌─▼─────────────────┐      │
│  │   vLLM    │ │  FastAPI   │ │  Health Check     │      │
│  │  Server   │ │   Server   │ │    Monitor        │      │
│  │           │ │            │ │                   │      │
│  │ Port 8000 │ │ Port 8080  │ │   (Background)    │      │
│  └───┬───────┘ └───┬────────┘ └───────────────────┘      │
│      │             │                                       │
│      │             │                                       │
│  ┌───▼─────────────▼─────────────────────────────────┐   │
│  │                                                     │   │
│  │            TTS Engine Components                   │   │
│  │                                                     │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────┐ │   │
│  │  │ Orchestrator │  │ SNAC Decoder │  │  Voice  │ │   │
│  │  │              │  │   (CUDA)     │  │ Config  │ │   │
│  │  └──────────────┘  └──────────────┘  └─────────┘ │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Supervisord (Process Manager)

**Purpose:** Manages multiple processes within a single Docker container

**Features:**
- Automatic process restart on failure
- Log rotation and management
- Health monitoring
- Graceful shutdown handling
- Process priority control

**Configuration:** [`supervisord.conf`](supervisord.conf)

**Managed Processes:**
- **vLLM Server** (Priority 100) - Starts first
- **FastAPI Server** (Priority 200) - Starts after vLLM
- **Health Check Monitor** (Priority 300) - Runs continuously

### 2. vLLM Server

**Purpose:** Serves the Svara TTS language model for token generation

**Port:** 8000  
**API:** OpenAI-compatible `/v1/completions` endpoint

**Features:**
- GPU-accelerated inference
- Tensor parallelism for multi-GPU
- Configurable memory utilization
- OpenAI API compatibility

**Key Configuration:**
- Model: `kenpath/svara-tts-v1`
- GPU Memory: 90% utilization (configurable)
- Max Context: 2048 tokens (configurable)
- Trust remote code: Yes (for custom model code)

### 3. FastAPI Server

**Purpose:** Public-facing REST API for text-to-speech synthesis

**Port:** 8080  
**Framework:** FastAPI (async Python)

**Endpoints:**
- `GET /health` - Health check
- `GET /v1/voices` - List available voices
- `POST /v1/text-to-speech` - Generate speech

**Features:**
- Async/await for high concurrency
- Streaming audio response
- Request validation with Pydantic
- Automatic API documentation (OpenAPI/Swagger)

### 4. TTS Engine Components

#### Orchestrator (`tts_engine/orchestrator.py`)

**Purpose:** Coordinates the TTS pipeline

**Flow:**
1. Accepts text and speaker_id
2. Formats prompt for vLLM
3. Streams tokens from vLLM
4. Decodes tokens to audio via SNAC
5. Streams PCM audio chunks

**Features:**
- Sync and async interfaces
- Concurrent decoding with thread pool
- Audio prebuffering for smooth playback
- Hop-only decoding for low latency

#### SNAC Decoder (`tts_engine/decoder_snac.py`)

**Purpose:** Converts model tokens to audio waveforms

**Specifications:**
- Sample Rate: 24 kHz
- Bit Depth: 16-bit PCM
- Channels: Mono
- Device: CUDA/MPS/CPU (auto-detected)

**Process:**
1. Receives 7-code frames from model
2. Reconstructs 3-layer SNAC codes
3. Decodes to audio using SNAC neural codec
4. Returns PCM16 bytes

#### Voice Configuration (`tts_engine/voice_config.py`)

**Purpose:** Manages voice profiles and metadata

**Voice Models:**
- **svara-tts-v1:** 38 voices (19 languages × 2 genders)
- **svara-tts-v2:** Custom voice profiles (future)

**Key Functions:**
- `get_all_voices()` - List all voices
- `get_voice(voice_id)` - Get specific voice
- `get_speaker_id(voice_id)` - Get speaker ID for prompt

### 5. Health Check Monitor

**Purpose:** Continuous monitoring of service health

**Checks:**
- vLLM server responsiveness
- FastAPI server responsiveness
- Logs status every 60 seconds

**Benefits:**
- Early problem detection
- Automatic restart on failure (via supervisord)
- Health status logging

## Data Flow

### Text-to-Speech Request Flow

```
1. HTTP Request
   POST /v1/text-to-speech
   ↓
2. FastAPI Server
   - Validate request
   - Get voice config
   - Get speaker_id
   ↓
3. TTS Orchestrator
   - Format prompt: "<|audio|> Hindi (Male): नमस्ते<|eot_id|>"
   - Create async audio stream
   ↓
4. vLLM Server
   - Generate tokens from prompt
   - Stream custom tokens
   ↓
5. SNAC Decoder (CUDA)
   - Decode tokens to audio frames
   - Convert to PCM16 bytes
   ↓
6. HTTP Response
   - Stream audio chunks to client
   - Content-Type: audio/pcm
```

### Startup Sequence

```
1. Docker Container Starts
   ↓
2. Supervisord Initializes
   ↓
3. vLLM Server Starts (Priority 100)
   - Loads model weights (~8-12GB)
   - Initializes CUDA kernels
   - Opens API endpoint on port 8000
   ↓
4. Health Check Waits (~30-60s)
   - Polls vLLM /health endpoint
   - Waits for "ready" status
   ↓
5. FastAPI Server Starts (Priority 200)
   - Loads voice configuration (38+ voices)
   - Initializes TTS orchestrator
   - Opens API endpoint on port 8080
   ↓
6. Health Monitor Starts (Priority 300)
   - Begins continuous monitoring
   - Logs status every 60s
   ↓
7. System Ready
   - Container health check passes
   - API accessible to clients
```

## Technology Stack

### Core Technologies

- **Python 3.10** - Primary language
- **FastAPI** - Web framework
- **Uvicorn** - ASGI server
- **vLLM** - LLM inference engine
- **PyTorch** - Deep learning framework
- **SNAC** - Neural audio codec

### Infrastructure

- **Docker** - Containerization
- **Supervisord** - Process management
- **NVIDIA CUDA** - GPU acceleration
- **Ubuntu 22.04** - Base OS

### Libraries

- **aiohttp** - Async HTTP client
- **pydantic** - Data validation
- **langcodes** - Language utilities
- **requests** - HTTP client

## Configuration

### Environment Variables

All configurable via `.env` file:

**vLLM:**
- `VLLM_MODEL` - Model repository
- `VLLM_GPU_MEMORY_UTILIZATION` - GPU memory %
- `VLLM_MAX_MODEL_LEN` - Max context length
- `VLLM_TENSOR_PARALLEL_SIZE` - GPU count

**API:**
- `API_PORT` - FastAPI port
- `VLLM_BASE_URL` - vLLM endpoint
- `TTS_DEVICE` - SNAC device

**See [`.env.example`](.env.example) for full list**

## Scalability Considerations

### Current Architecture (Single Container)

**Pros:**
- Simple deployment
- Low latency (no network overhead)
- Easy development and debugging

**Cons:**
- Single point of failure
- Limited horizontal scaling
- Coupled services

### Future Architecture Options

#### 1. Separate Containers

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   FastAPI   │─────▶│  vLLM Server │─────▶│    SNAC     │
│  Container  │      │   Container  │      │  Container  │
└─────────────┘      └──────────────┘      └─────────────┘
```

**Benefits:**
- Independent scaling
- Service isolation
- Easier updates

#### 2. Load Balanced

```
       ┌──────────────┐
       │ Load Balancer│
       └──────┬───────┘
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
┌────────┐ ┌────────┐ ┌────────┐
│ API #1 │ │ API #2 │ │ API #3 │
└────────┘ └────────┘ └────────┘
    │         │         │
    └─────────┼─────────┘
              ▼
       ┌──────────────┐
       │ vLLM Cluster │
       └──────────────┘
```

**Benefits:**
- High availability
- Horizontal scaling
- Traffic distribution

#### 3. Kubernetes Deployment

**Benefits:**
- Auto-scaling
- Self-healing
- Service discovery
- Rolling updates

## Security Considerations

### Current Implementation

- No authentication (internal deployment)
- No rate limiting
- Trust remote code enabled

### Production Recommendations

1. **API Authentication**
   - JWT tokens
   - API keys
   - OAuth2

2. **Rate Limiting**
   - Per-user quotas
   - Request throttling
   - DDoS protection

3. **Network Security**
   - HTTPS/TLS
   - Firewall rules
   - VPC isolation

4. **Model Security**
   - Verify model checksums
   - Scan for vulnerabilities
   - Regular updates

## Monitoring and Observability

### Current Logging

- Supervisord logs: `/var/log/supervisor/`
- vLLM logs: `/var/log/supervisor/vllm.log`
- FastAPI logs: `/var/log/supervisor/fastapi.log`
- Health check logs: `/var/log/supervisor/healthcheck.log`

### Future Monitoring

- **Metrics:** Prometheus + Grafana
- **Tracing:** OpenTelemetry
- **Logging:** ELK Stack (Elasticsearch, Logstash, Kibana)
- **Alerts:** PagerDuty, Slack notifications

## Performance Optimization

### Current Optimizations

- Concurrent SNAC decoding (thread pool)
- Audio prebuffering (1.2s)
- Hop-only decoding (512 samples)
- GPU memory optimization (90%)

### Future Optimizations

- Model quantization (INT8/INT4)
- KV cache optimization
- Batch inference
- Response caching
- CDN for audio delivery

## Development vs Production

### Development Mode

```bash
./scripts/start-dev.sh
```

- Single bash script
- Manual process management
- Live code reload
- Verbose logging

### Production Mode

```bash
docker-compose up -d
```

- Supervisord process management
- Automatic restart
- Log rotation
- Health monitoring
- Container orchestration

## Further Reading

- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment instructions
- [README.md](README.md) - Getting started
- [REFACTORING_NOTES.md](REFACTORING_NOTES.md) - Code changes
- [supervisord.conf](supervisord.conf) - Process configuration

