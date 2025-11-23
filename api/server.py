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
import subprocess

from fastapi import FastAPI, HTTPException, Response, File, UploadFile, Form
from fastapi.responses import StreamingResponse


logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tts_engine.voice_config import get_all_voices
from tts_engine.orchestrator import SvaraTTSOrchestrator
from tts_engine.timing import get_timing_stats, reset_timing_stats
from tts_engine.utils import load_audio_from_bytes
from tts_engine.codec import SNACCodec, get_or_load_tokenizer
from api.models import VoiceResponse, VoicesResponse, TTSRequest


# ============================================================================
# Configuration
# ============================================================================

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.getenv("VLLM_MODEL", "kenpath/svara-tts-v1")
TOKENIZER_MODEL = os.getenv("TOKENIZER_MODEL", VLLM_MODEL)  # Defaults to VLLM_MODEL
TTS_DEVICE = os.getenv("TTS_DEVICE", None)  # None = auto-detect (CUDA/MPS/CPU)
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
    print(f"   vLLM URL: {VLLM_BASE_URL}")
    print(f"   Model: {VLLM_MODEL}")
    print(f"   Tokenizer Model: {TOKENIZER_MODEL}")
    print(f"   Device: {TTS_DEVICE or 'auto-detect'}")
    print(f"   HF_TOKEN: {'set' if os.getenv('HF_TOKEN') else 'not set'}")
    
    # Initialize orchestrator with default settings
    # We'll create new instances per request with specific voice settings
    orchestrator = SvaraTTSOrchestrator(
        base_url=VLLM_BASE_URL,
        model=VLLM_MODEL,
        speaker_id="English (Male)",  # Default, will be overridden per request
        device=TTS_DEVICE,
        prebuffer_seconds=0.5,
        concurrent_decode=True,
        max_workers=2,
    )
    
    # Note: Tokenizer is loaded on-demand and cached globally in codec.py
    print(f"✓ Orchestrator initialized")
    print(f"✓ Loaded {len(get_all_voices())} voices")
    print(f"✓ Tokenizer will be lazy-loaded on first zero-shot request")
    
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
        "vllm_url": VLLM_BASE_URL,
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


@app.post("/v1/text-to-speech")
async def text_to_speech(
    # Accept both JSON (via TTSRequest) and multipart/form-data
    text: Optional[str] = Form(None),
    voice: Optional[str] = Form(None),
    reference_audio: Optional[UploadFile] = File(None),
    reference_transcript: Optional[str] = Form(None),
    model_id: str = Form(default="svara-tts-v1"),
    stream: bool = Form(default=True),
    response_format: str = Form(default="opus"),  # Default to opus for streaming
    temperature: Optional[float] = Form(None),
    top_p: Optional[float] = Form(None),
    top_k: Optional[int] = Form(None),
    repetition_penalty: Optional[float] = Form(None),
    max_tokens: Optional[int] = Form(None),
    # JSON body (used when Content-Type is application/json)
    json_body: Optional[TTSRequest] = None,
):
    """
    Convert text to speech with streaming or non-streaming response.
    
    Supports two modes:
    1. Standard TTS: Provide 'voice' parameter
    2. Zero-shot cloning: Provide 'reference_audio' file (and optionally 'reference_transcript')
    
    Accepts both:
    - JSON (Content-Type: application/json) with base64-encoded reference_audio
    - Multipart form data (Content-Type: multipart/form-data) with file upload
    
    Returns:
        Audio bytes in requested format (streaming or complete)
    """
    # Handle both JSON and multipart/form-data
    if json_body is not None:
        # JSON request
        request_text = json_body.text
        request_voice = json_body.voice
        request_reference_audio_bytes = json_body.reference_audio
        request_reference_transcript = json_body.reference_transcript
        request_model_id = json_body.model_id
        request_stream = json_body.stream
        request_response_format = json_body.response_format
        request_temperature = json_body.temperature
        request_top_p = json_body.top_p
        request_top_k = json_body.top_k
        request_repetition_penalty = json_body.repetition_penalty
        request_max_tokens = json_body.max_tokens
    else:
        # Multipart form data request
        if text is None:
            raise HTTPException(status_code=400, detail="'text' field is required")
        request_text = text
        request_voice = voice
        request_reference_transcript = reference_transcript
        request_model_id = model_id
        request_stream = stream
        request_response_format = response_format
        request_temperature = temperature
        request_top_p = top_p
        request_top_k = top_k
        request_repetition_penalty = repetition_penalty
        request_max_tokens = max_tokens
        
        # Handle file upload for reference_audio
        if reference_audio is not None:
            request_reference_audio_bytes = await reference_audio.read()
        else:
            request_reference_audio_bytes = None
    
    # Validate that either voice or reference_audio is provided
    if not request_voice and not request_reference_audio_bytes:
        raise HTTPException(
            status_code=400,
            detail="Either 'voice' or 'reference_audio' must be provided"
        )
    
    # Determine mode: zero-shot or standard
    zero_shot_mode = request_reference_audio_bytes is not None
    audio_tokens = None
    
    if zero_shot_mode:
        logger.info(f"Loading reference audio from bytes ({len(request_reference_audio_bytes)} bytes)")
        audio_tensor, sample_rate = load_audio_from_bytes(request_reference_audio_bytes, device=TTS_DEVICE)
        logger.info(f"Audio loaded: shape={audio_tensor.shape}, sr={sample_rate}Hz, min={audio_tensor.min():.3f}, max={audio_tensor.max():.3f}")
        
        # Encode audio to SNAC tokens
        logger.info(f"Encoding audio to SNAC tokens")
        codec = SNACCodec(device=TTS_DEVICE)
        
        # Encode with offsets (128266+) for use in prompt
        audio_tokens = codec.encode_audio(audio_tensor, input_sample_rate=sample_rate, add_token_offsets=True)
        logger.info(f"Audio tokens encoded to {len(audio_tokens)} tokens")
        logger.info(f"First 10 tokens: {audio_tokens[:10]}")
        logger.info(f"Last 10 tokens: {audio_tokens[-10:]}")
    else:
        # Standard TTS mode
        if not request_voice:
            raise HTTPException(
                status_code=400,
                detail="'voice' parameter is required for standard TTS mode"
            )
    
    # Use global orchestrator (already initialized, SNAC model cached)
    request_orchestrator = orchestrator
    
    # Build generation kwargs from request parameters
    gen_kwargs = {}
    if request_temperature is not None:
        gen_kwargs["temperature"] = request_temperature
    if request_top_p is not None:
        gen_kwargs["top_p"] = request_top_p
    if request_top_k is not None:
        gen_kwargs["top_k"] = request_top_k
    if request_repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = request_repetition_penalty
    if request_max_tokens is not None:
        gen_kwargs["max_tokens"] = request_max_tokens
    
    # Map format to media type
    format_media_types = {
        "mp3": "audio/mpeg",
        "opus": "audio/ogg",
        "aac": "audio/aac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }
    media_type = format_media_types.get(request_response_format, "audio/pcm")
    
    # Create async generator for raw PCM
    pcm_generator = request_orchestrator.astream(
        text=request_text,
        audio_reference=audio_tokens,
        reference_text=request_reference_transcript,
        speaker_id=request_voice,
        **gen_kwargs
    )
    
    # Convert to requested format
    audio_stream = audio_stream_converter(
        pcm_generator,
        format=request_response_format,
    )
    
    # Handle streaming vs non-streaming
    if request_stream:
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
        # Non-streaming: collect all audio chunks
        try:
            audio_chunks = []
            async for chunk in audio_stream:
                audio_chunks.append(chunk)
            
            complete_audio = b"".join(audio_chunks)
            
            return Response(
                content=complete_audio,
                media_type=media_type,
                headers={
                    "Content-Type": media_type,
                    "X-Sample-Rate": "24000",
                    "X-Channels": "1",
                    "Content-Length": str(len(complete_audio)),
                }
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error generating audio: {str(e)}"
            )


# ============================================================================
# Debug Endpoints
# ============================================================================

@app.get("/debug/timing")
async def get_timing():
    """
    Get timing statistics for all tracked functions.
    
    Returns detailed performance metrics including call counts, average times,
    min/max times for each tracked function.
    """
    stats = get_timing_stats()
    
    # Convert to more readable format
    formatted_stats = {}
    for func_name, data in stats.items():
        # Skip functions that haven't been called yet
        if data["count"] == 0:
            continue
            
        avg_time = data["total_time"] / data["count"]
        # Handle inf values (when count is 0)
        min_time = data["min_time"] if data["min_time"] != float('inf') else 0
        max_time = data["max_time"] if data["max_time"] != float('-inf') else 0
        
        formatted_stats[func_name] = {
            "calls": data["count"],
            "total_ms": round(data["total_time"] * 1000, 2),
            "avg_ms": round(avg_time * 1000, 2),
            "min_ms": round(min_time * 1000, 2),
            "max_ms": round(max_time * 1000, 2),
        }
    
    return {
        "timing_stats": formatted_stats,
        "note": "All times in milliseconds"
    }


@app.post("/debug/timing/reset")
async def reset_timing():
    """
    Reset all timing statistics.
    
    Clears all accumulated timing data. Useful for starting fresh measurements.
    """
    reset_timing_stats()
    return {"status": "success", "message": "Timing statistics have been reset"}


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
