# Svara TTS API - Architecture Overview

## System Architecture

The Svara TTS API runs as a **single process** with the vLLM inference engine embedded directly in the FastAPI application. This eliminates the HTTP hop between services, reducing latency and simplifying operations.

```
┌───────────────────────────────────────────────────────────┐
│                     Docker Container                       │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              Supervisord (Process Manager)           │  │
│  └───────────────────────┬─────────────────────────────┘  │
│                          │                                 │
│  ┌───────────────────────▼─────────────────────────────┐  │
│  │              FastAPI Server (Port 8080)              │  │
│  │                                                     │  │
│  │  ┌───────────────────────────────────────────────┐  │  │
│  │  │        Embedded vLLM Engine                   │  │  │
│  │  │        (AsyncLLMEngine singleton)             │  │  │
│  │  └───────────────────┬───────────────────────────┘  │  │
│  │                      │                              │  │
│  │  ┌───────────────────▼───────────────────────────┐  │  │
│  │  │           TTS Engine Components               │  │  │
│  │  │                                               │  │  │
│  │  │  ┌──────────────┐  ┌────────────┐  ┌───────┐ │  │  │
│  │  │  │ Orchestrator │  │   SNAC     │  │ Voice │ │  │  │
│  │  │  │              │  │  Decoder   │  │Config │ │  │  │
│  │  │  └──────────────┘  └────────────┘  └───────┘ │  │  │
│  │  │                                               │  │  │
│  │  └───────────────────────────────────────────────┘  │  │
│  │                                                     │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Supervisord (Process Manager)

**Purpose:** Manages the FastAPI process within the Docker container

**Features:**
- Automatic process restart on failure
- Log streaming to stdout/stderr
- Graceful shutdown handling

**Configuration:** [`supervisord.conf`](supervisord.conf)

### 2. Embedded vLLM Engine

**Purpose:** Runs the Svara TTS language model for token generation in-process

**Key Design:**
- **Singleton pattern**: `VLLMEmbeddedTransport` holds a class-level `AsyncLLMEngine` instance — GPU resources are allocated once at startup
- **`initialize_engine()`**: Called during FastAPI lifespan startup with env-var-driven configuration
- **Direct generation**: `engine.generate()` yields `RequestOutput` objects with accumulated text; the transport computes deltas
- **Request cancellation**: On error or client disconnect, `engine.abort(request_id)` frees vLLM resources

**Configuration (via environment variables):**

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_MODEL` | `kenpath/svara-tts-v1` | Hugging Face model repository |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.9` | GPU memory usage (0.0-1.0) |
| `VLLM_MAX_MODEL_LEN` | `4096` | Maximum context length |
| `VLLM_TENSOR_PARALLEL_SIZE` | `1` | Number of GPUs for parallelism |
| `VLLM_DTYPE` | `auto` | Data type (`auto`, `float16`, `bfloat16`) |
| `VLLM_QUANTIZATION` | (none) | Quantization method (`fp8`, `awq`, `gptq`) |
| `VLLM_ENFORCE_EAGER` | `false` | Disable CUDA graphs (for debugging) |

### 3. FastAPI Server

**Purpose:** Public-facing REST API for text-to-speech synthesis

**Port:** 8080
**Framework:** FastAPI (async Python)

**Endpoints:**
- `GET /health` — Health check
- `GET /v1/voices` — List available voices
- `POST /v1/text-to-speech` — Full-featured TTS (supports zero-shot cloning, gen params)
- `POST /v1/audio/speech` — OpenAI-compatible TTS endpoint
- `GET /debug/timing` — Performance timing statistics
- `POST /debug/timing/reset` — Reset timing stats

**Features:**
- Async/await for high concurrency
- Streaming audio response with format conversion
- Request validation with Pydantic
- Automatic API documentation (OpenAPI/Swagger at `/docs`)
- OpenAI SDK compatibility

### 4. TTS Engine Components

#### Orchestrator (`tts_engine/orchestrator.py`)

**Purpose:** Coordinates the TTS pipeline

**Flow:**
1. Accepts text and speaker_id
2. Encodes prompt via `svara_text_to_tokens()`
3. Streams tokens from embedded vLLM engine
4. Maps tokens through `SvaraMapper` (7-token SNAC frames)
5. Decodes frames to PCM audio via SNAC codec
6. Prebuffers and streams PCM int16 chunks

**Features:**
- Sync and async interfaces (`stream()` / `astream()`)
- Per-request `SvaraMapper` instances (thread-safe for concurrent requests)
- Concurrent SNAC decoding with thread pool
- Audio prebuffering for smooth playback
- Buffer flush for short utterances

#### Transport (`tts_engine/transports.py`)

**Purpose:** Interface between orchestrator and vLLM engine

**Class:** `VLLMEmbeddedTransport`
- `initialize_engine()` — class method, called once at startup
- `astream(prompt, **gen_kwargs)` — async generator yielding text deltas
- `stream(prompt, **gen_kwargs)` — sync wrapper (runs async code on a separate thread)

#### SNAC Decoder (`tts_engine/codec.py`)

**Purpose:** Converts model tokens to audio waveforms

**Specifications:**
- Sample Rate: 24 kHz
- Bit Depth: 16-bit PCM
- Channels: Mono
- Device: CUDA/MPS/CPU (auto-detected via `TTS_DEVICE` env var)

**Process:**
1. Receives 7-code frames from mapper
2. Reconstructs 3-layer SNAC codes
3. Decodes to audio using SNAC neural codec
4. Returns PCM16 bytes

#### Audio Buffer (`tts_engine/buffers.py`)

**Purpose:** Manages prebuffering for smooth streaming playback

**Features:**
- Accumulates audio until a threshold (default 0.5s) before starting to yield
- `flush()` method ensures short utterances are not lost

#### Voice Configuration (`tts_engine/voice_config.py`)

**Purpose:** Manages voice profiles and metadata

**Voice Models:**
- **svara-tts-v1:** 38 voices (19 languages x 2 genders)
- **svara-tts-v2:** Custom voice profiles (future)

**Key Functions:**
- `get_all_voices()` — List all voices
- `get_voice(voice_id)` — Get specific voice by ID
- `get_speaker_id(voice_id)` — Convert voice ID (e.g. `hi_male`) to speaker ID (e.g. `Hindi (Male)`)

#### Audio Format Converter (`api/server.py`)

**Purpose:** Converts raw PCM stream to requested audio format

**Implementation:** Pipes PCM data through ffmpeg subprocess
- Supports MP3, Opus, AAC, WAV output
- Streams conversion (no full audio buffering required)
- PCM passthrough when no conversion needed

## Data Flow

### Text-to-Speech Request Flow

```
1. HTTP Request
   POST /v1/text-to-speech  or  POST /v1/audio/speech
   ↓
2. FastAPI Server
   - Validate request (Pydantic)
   - Resolve voice → speaker_id
   - (Optional) Encode reference audio for zero-shot cloning
   ↓
3. TTS Orchestrator
   - Encode text + speaker_id into prompt tokens
   - Create async audio stream
   ↓
4. Embedded vLLM Engine
   - engine.generate(prompt, sampling_params, request_id)
   - Yields RequestOutput with accumulated text
   - Transport computes text deltas
   ↓
5. Token Mapper (SvaraMapper)
   - Extracts custom token numbers from text deltas
   - Groups into 7-token SNAC frames
   ↓
6. SNAC Decoder (CUDA)
   - Decodes 7-token frames to audio waveform
   - Concurrent decoding via ThreadPoolExecutor
   - Returns PCM16 bytes
   ↓
7. Audio Buffer
   - Prebuffers ~0.5s for smooth playback
   - Flushes remaining audio at end
   ↓
8. Format Converter (ffmpeg)
   - PCM → MP3/Opus/AAC/WAV (if requested)
   - Streams output chunks
   ↓
9. HTTP Response
   - StreamingResponse with chunked transfer encoding
   - Or complete file for non-streaming requests
```

### Startup Sequence

```
1. Docker Container Starts
   ↓
2. Supervisord Initializes
   ↓
3. FastAPI/Uvicorn Process Starts
   ↓
4. Lifespan: VLLMEmbeddedTransport.initialize_engine()
   - Loads model weights from HuggingFace (~8-12GB)
   - Initializes CUDA kernels, KV cache
   - Allocates GPU memory
   ↓
5. Lifespan: SvaraTTSOrchestrator initialized
   - Loads SNAC codec
   - Loads tokenizer
   ↓
6. Lifespan: Voice config loaded (38 voices)
   ↓
7. Server Ready
   - API accessible on port 8080
   - Container health check passes
```

## Technology Stack

### Core Technologies

- **Python 3.11** — Primary language
- **FastAPI** — Web framework
- **Uvicorn** — ASGI server
- **vLLM** (>=0.17.0) — LLM inference engine (embedded `AsyncLLMEngine`)
- **PyTorch** — Deep learning framework
- **SNAC** — Neural audio codec
- **ffmpeg** — Audio format conversion

### Infrastructure

- **Docker** — Containerization
- **Supervisord** — Process management (auto-restart)
- **NVIDIA CUDA** — GPU acceleration
- **Ubuntu 22.04** — Base OS

### Libraries

- **pydantic** — Data validation and request models
- **python-dotenv** — Environment variable loading
- **langcodes** — Language utilities
- **transformers** — Tokenizer loading

## Configuration

### Environment Variables

All configurable via `.env` file:

**vLLM Engine:**
- `VLLM_MODEL` — Model repository
- `VLLM_GPU_MEMORY_UTILIZATION` — GPU memory %
- `VLLM_MAX_MODEL_LEN` — Max context length
- `VLLM_TENSOR_PARALLEL_SIZE` — GPU count
- `VLLM_DTYPE` — Data type (auto, float16, bfloat16)
- `VLLM_QUANTIZATION` — Quantization (fp8, awq, gptq)
- `VLLM_ENFORCE_EAGER` — Disable CUDA graphs

**API:**
- `API_PORT` — FastAPI port
- `API_HOST` — FastAPI bind host
- `TTS_DEVICE` — SNAC decoder device (cuda, mps, cpu)

**See [`.env.example`](.env.example) for full list**

## Scalability Considerations

### Current Architecture (Single Process)

**Pros:**
- Minimal latency (no HTTP hop between LLM and API)
- Simple deployment (one process, one port)
- Easy development and debugging
- Efficient GPU memory use (single process owns GPU)

**Cons:**
- Single point of failure (mitigated by supervisord auto-restart)
- Scaling requires running multiple container instances

### Future Architecture Options

#### 1. Load Balanced

```
       ┌──────────────┐
       │ Load Balancer │
       └──────┬───────┘
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
┌────────┐ ┌────────┐ ┌────────┐
│ Svara  │ │ Svara  │ │ Svara  │
│  #1    │ │  #2    │ │  #3    │
│(GPU 0) │ │(GPU 1) │ │(GPU 2) │
└────────┘ └────────┘ └────────┘
```

**Benefits:**
- High availability
- Horizontal scaling (one GPU per instance)
- Traffic distribution

#### 2. Decoupled LLM + Decoder (Advanced)

```
┌─────────────┐      ┌──────────────┐
│   FastAPI    │─────▶│  vLLM Pool   │ (token generation)
│  Frontends  │      └──────┬───────┘
└──────┬──────┘             │
       │                    ▼
       └────────────▶┌──────────────┐
                     │  SNAC Pool   │ (audio decoding)
                     └──────────────┘
```

**Benefits:**
- Independent scaling of LLM and decoder
- fp8 KV cache quantization on LLM nodes
- Adaptive micro-batching

#### 3. Kubernetes Deployment

**Benefits:**
- Auto-scaling based on GPU utilization
- Self-healing
- Service discovery
- Rolling updates

## Security Considerations

### Current Implementation

- No authentication (designed for internal/private deployment)
- No rate limiting
- Trust remote code enabled (required for model)

### Production Recommendations

1. **API Authentication** — JWT tokens, API keys, or OAuth2
2. **Rate Limiting** — Per-user quotas, request throttling
3. **Network Security** — HTTPS/TLS, firewall rules, VPC isolation
4. **Model Security** — Verify model checksums, regular updates

## Monitoring and Observability

### Built-in Monitoring

- **Timing stats**: `GET /debug/timing` returns per-function call counts, avg/min/max latency
- **Health check**: `GET /health` for container orchestration
- **Logs**: Streamed to stdout/stderr via supervisord

### Future Monitoring

- **Metrics:** Prometheus + Grafana
- **Tracing:** OpenTelemetry
- **Logging:** ELK Stack
- **Alerts:** PagerDuty, Slack notifications

## Performance Optimization

### Current Optimizations

- Embedded vLLM engine (zero network latency for token generation)
- Concurrent SNAC decoding (ThreadPoolExecutor)
- Audio prebuffering (0.5s default)
- Per-request mapper instances (safe concurrent access)
- Request cancellation with `engine.abort()` (frees GPU resources)

### Tuning Knobs

| Setting | Effect |
|---------|--------|
| `VLLM_GPU_MEMORY_UTILIZATION` | Higher = more KV cache, more concurrent requests |
| `VLLM_QUANTIZATION=fp8` | ~2x throughput on H100, slight quality trade-off |
| `VLLM_DTYPE=bfloat16` | Better precision than float16 on Ampere+ GPUs |
| `prebuffer_seconds` | Lower = faster TTFB, higher = smoother playback |
| `max_workers` | More workers = more concurrent SNAC decoding |

### Reference Latency

Based on Orpheus TTS ecosystem benchmarks:
- ~130ms TTFB on H100 (with fp8 quantization)
- ~150-200ms TTFB on H100 (baseline)
- ~250-300ms TTFB on A100 (baseline)

## Development vs Production

### Development Mode

```bash
./scripts/start-dev.sh
```

- Single uvicorn process
- .env file auto-loaded via python-dotenv
- Verbose logging

### Production Mode

```bash
docker-compose up -d
```

- Supervisord process management with auto-restart
- Log streaming to container stdout
- Health monitoring via Docker healthcheck
- Container orchestration ready

## Further Reading

- [DEPLOYMENT.md](DEPLOYMENT.md) — Deployment instructions
- [README.md](README.md) — Getting started
- [supervisord.conf](supervisord.conf) — Process configuration
- [.env.example](.env.example) — All configuration options
