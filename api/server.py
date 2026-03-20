"""
FastAPI server for Svara TTS API.

Provides ElevenLabs-style text-to-speech endpoints with support for
Indian language voices and streaming audio generation.
"""
from __future__ import annotations
import os
import sys
import logging
from pathlib import Path
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager
import asyncio

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse


logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tts_engine.voice_config import get_all_voices, get_speaker_id
from tts_engine.orchestrator import SvaraTTSOrchestrator
from tts_engine.transports import VLLMEmbeddedTransport
from tts_engine.utils import load_audio_from_bytes
from tts_engine.codec import SNACCodec
from api.models import VoiceResponse, VoicesResponse, OpenAISpeechRequest


# ============================================================================
# Configuration
# ============================================================================

VLLM_MODEL = os.getenv("VLLM_MODEL", "kenpath/svara-tts-v1")
TOKENIZER_MODEL = os.getenv("TOKENIZER_MODEL", VLLM_MODEL)  # Defaults to VLLM_MODEL
SNAC_DEVICE = os.getenv("SNAC_DEVICE", None)  # None = auto-detect (CUDA/MPS/CPU). Device for SNAC audio decoder.
VLLM_GPU_MEMORY_UTILIZATION = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9"))
VLLM_MAX_MODEL_LEN = int(os.getenv("VLLM_MAX_MODEL_LEN", "4096"))
VLLM_TENSOR_PARALLEL_SIZE = int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1"))
VLLM_QUANTIZATION = os.getenv("VLLM_QUANTIZATION") or None
VLLM_ENFORCE_EAGER = os.getenv("VLLM_ENFORCE_EAGER", "false").lower() in ("true", "1", "yes")
VLLM_DTYPE = os.getenv("VLLM_DTYPE", "auto")
# HF_TOKEN is checked in codec.get_or_load_tokenizer() for private models

# Global instances (initialized in lifespan)
orchestrator: Optional[SvaraTTSOrchestrator] = None


# ============================================================================
# Application Lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global orchestrator
    
    print(f"🚀 Initializing Svara TTS API...")
    print(f"   Model: {VLLM_MODEL}")
    print(f"   Tokenizer Model: {TOKENIZER_MODEL}")
    print(f"   Device: {SNAC_DEVICE or 'auto-detect'}")
    print(f"   dtype: {VLLM_DTYPE}, quantization: {VLLM_QUANTIZATION}")
    print(f"   max_model_len: {VLLM_MAX_MODEL_LEN}, enforce_eager: {VLLM_ENFORCE_EAGER}")
    print(f"   HF_TOKEN: {'set' if os.getenv('HF_TOKEN') else 'not set'}")

    # Initialize embedded vLLM engine (singleton)
    VLLMEmbeddedTransport.initialize_engine(
        model=VLLM_MODEL,
        gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        max_model_len=VLLM_MAX_MODEL_LEN,
        tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
        dtype=VLLM_DTYPE,
        quantization=VLLM_QUANTIZATION,
        enforce_eager=VLLM_ENFORCE_EAGER,
    )
    print(f"✓ vLLM engine initialized (embedded)")

    transport = VLLMEmbeddedTransport(model=VLLM_MODEL)

    # Initialize orchestrator with default settings
    orchestrator = SvaraTTSOrchestrator(
        transport=transport,
        model=VLLM_MODEL,
        speaker_id="English (Male)",  # Default, will be overridden per request
        device=SNAC_DEVICE,
        prebuffer_seconds=0.5,
        concurrent_decode=True,
    )

    print(f"✓ Orchestrator initialized")
    print(f"✓ Loaded {len(get_all_voices())} voices")
    
    yield
    
    print("🛑 Shutting down Svara TTS API...")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Svara TTS API",
    description="Text-to-speech API for Indian languages with streaming support",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================================
# Helpers
# ============================================================================

async def audio_stream_converter(
    pcm_stream: AsyncGenerator[bytes, None],
    format: str,
    sample_rate: int = 24000,
    channels: int = 1
) -> AsyncGenerator[bytes, None]:
    """
    Convert PCM stream to target format using ffmpeg.
    
    Args:
        pcm_stream: Async iterator yielding PCM bytes
        format: Target format ('mp3', 'opus', 'aac', 'wav', 'pcm')
        sample_rate: Input sample rate
        channels: Input channels
        
    Yields:
        Encoded audio bytes
    """
    if format == "pcm":
        async for chunk in pcm_stream:
            yield chunk
        return

    # Setup ffmpeg command
    cmd = [
        "ffmpeg",
        "-f", "s16le",       # Input format: signed 16-bit little-endian
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-i", "pipe:0",      # Read from stdin
        "-loglevel", "error" # Suppress output
    ]

    # Format specific flags
    if format == "mp3":
        cmd.extend(["-f", "mp3", "pipe:1"])
    elif format == "opus":
        cmd.extend(["-f", "opus", "pipe:1"])
    elif format == "aac":
        cmd.extend(["-f", "adts", "pipe:1"]) # ADTS is streamable AAC container
    elif format == "wav":
        cmd.extend(["-f", "wav", "pipe:1"])
    else:
        # Fallback to PCM if unknown format
        logger.warning(f"Unknown format '{format}', falling back to PCM")
        async for chunk in pcm_stream:
            yield chunk
        return

    # Start ffmpeg process
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    async def write_stdin():
        try:
            async for chunk in pcm_stream:
                if process.stdin:
                    process.stdin.write(chunk)
                    await process.stdin.drain()
            if process.stdin:
                process.stdin.close()
        except Exception as e:
            logger.error(f"Error writing to ffmpeg stdin: {e}")
            try:
                process.kill()
            except:
                pass

    # Create task to write to stdin
    write_task = asyncio.create_task(write_stdin())

    # Read from stdout
    try:
        if process.stdout:
            while True:
                chunk = await process.stdout.read(4096)
                if not chunk:
                    break
                yield chunk
    except Exception as e:
        logger.error(f"Error reading from ffmpeg stdout: {e}")
        raise
    finally:
        # Ensure process is cleaned up
        if not write_task.done():
            write_task.cancel()
            try:
                await write_task
            except asyncio.CancelledError:
                pass
        
        if process.returncode is None:
            try:
                process.kill()
            except:
                pass
            await process.wait()


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return {
        "status": "healthy",
        "model": VLLM_MODEL,
        "engine": "embedded",
    }


@app.get("/v1/voices", response_model=VoicesResponse)
async def get_voices(model_id: Optional[str] = None):
    """
    Get list of available voices.
    
    Args:
        model_id: Optional filter by model ID (e.g., "svara-tts-v1")
    
    Returns:
        List of available voices with metadata
    """
    voices = get_all_voices(model_id=model_id)
    return VoicesResponse(
        voices=[VoiceResponse(**voice.to_dict()) for voice in voices]
    )


@app.post("/v1/audio/speech")
async def openai_speech(req: OpenAISpeechRequest):
    """
    OpenAI-compatible text-to-speech endpoint.

    Drop-in replacement for OpenAI's /v1/audio/speech. Works with the OpenAI SDK
    out of the box. Extended features (zero-shot cloning, generation params) can
    be passed via the SDK's `extra_body` parameter.

    Supports:
    - Standard TTS with any available voice
    - Zero-shot voice cloning via `reference_audio` (base64)
    - Generation parameter tuning (temperature, top_p, etc.)
    - Streaming and non-streaming responses
    - Multiple audio formats (mp3, opus, aac, wav, pcm)
    """
    # Resolve voice: try voice_id lookup first, then treat as direct speaker name
    speaker_id = None
    if not req.reference_audio:
        try:
            speaker_id = get_speaker_id(req.voice)
        except ValueError:
            # Maybe it's already a speaker name like "Hindi (Male)"
            # Pass it through and let the orchestrator handle it
            speaker_id = req.voice

    # Zero-shot voice cloning
    audio_tokens = None
    if req.reference_audio:
        logger.info(f"Loading reference audio from bytes ({len(req.reference_audio)} bytes)")
        audio_tensor, sample_rate = load_audio_from_bytes(req.reference_audio, device=SNAC_DEVICE)
        logger.info(f"Audio loaded: shape={audio_tensor.shape}, sr={sample_rate}Hz")

        codec = SNACCodec(device=SNAC_DEVICE)
        audio_tokens = codec.encode_audio(audio_tensor, input_sample_rate=sample_rate, add_token_offsets=True)
        logger.info(f"Audio tokens encoded: {len(audio_tokens)} tokens")

    # Build generation kwargs
    gen_kwargs = {}
    if req.temperature is not None:
        gen_kwargs["temperature"] = req.temperature
    if req.top_p is not None:
        gen_kwargs["top_p"] = req.top_p
    if req.top_k is not None:
        gen_kwargs["top_k"] = req.top_k
    if req.repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = req.repetition_penalty
    if req.max_tokens is not None:
        gen_kwargs["max_tokens"] = req.max_tokens

    # Map format to media type
    format_media_types = {
        "mp3": "audio/mpeg",
        "opus": "audio/ogg",
        "aac": "audio/aac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }
    media_type = format_media_types.get(req.response_format, "audio/mpeg")

    # Generate audio
    pcm_generator = orchestrator.astream(
        text=req.input,
        audio_reference=audio_tokens,
        reference_text=req.reference_transcript,
        speaker_id=speaker_id,
        **gen_kwargs,
    )

    audio_stream = audio_stream_converter(
        pcm_generator,
        format=req.response_format,
    )

    if req.stream:
        return StreamingResponse(
            audio_stream,
            media_type=media_type,
            headers={
                "Content-Type": media_type,
                "X-Sample-Rate": "24000",
                "X-Channels": "1",
            }
        )
    else:
        audio_chunks = []
        async for chunk in audio_stream:
            audio_chunks.append(chunk)

        complete_audio = b"".join(audio_chunks)

        return Response(
            content=complete_audio,
            media_type=media_type,
            headers={
                "Content-Type": media_type,
                "Content-Length": str(len(complete_audio)),
            }
        )


# ============================================================================
# Main Entry Point (for local development only)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", "8080"))
    host = os.getenv("API_HOST", "0.0.0.0")
    
    print(f"Starting Svara TTS API on {host}:{port}")
    print("Note: For production, use supervisord to manage processes")
    
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
