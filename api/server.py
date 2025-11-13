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
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response, File, UploadFile, Form
from fastapi.responses import StreamingResponse


logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tts_engine.voice_config import get_all_voices
from tts_engine.orchestrator import SvaraTTSOrchestrator
from tts_engine.timing import get_timing_stats, reset_timing_stats
from tts_engine.utils import load_audio_from_bytes, svara_zero_shot_prompt, svara_prompt
from tts_engine.snac_codec import SNACCodec
from api.models import VoiceResponse, VoicesResponse, TTSRequest


# ============================================================================
# Configuration
# ============================================================================

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.getenv("VLLM_MODEL", "kenpath/svara-tts-v1")
TTS_DEVICE = os.getenv("TTS_DEVICE", None)  # None = auto-detect (CUDA/MPS/CPU)

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
    print(f"   Device: {TTS_DEVICE or 'auto-detect'}")
    
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
        Raw PCM16 audio bytes (streaming or complete)
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
    
    # Currently only v1 is implemented
    # TODO: Implement other models when svara-tts-v2 is released
    # if request_model_id != "svara-tts-v1":
    #     raise HTTPException(
    #         status_code=501,
    #         detail=f"Model '{request_model_id}' is not yet implemented. Currently only 'svara-tts-v1' is supported."
    #     )
    
    # Determine mode: zero-shot or standard
    zero_shot_mode = request_reference_audio_bytes is not None
    prompt = None
    
    if zero_shot_mode:
        logger.info(f"Loading reference audio from bytes ({len(request_reference_audio_bytes)} bytes)")
        audio_tensor, sample_rate = load_audio_from_bytes(request_reference_audio_bytes, device=TTS_DEVICE)
        logger.info(f"Audio loaded: shape={audio_tensor.shape}, sr={sample_rate}Hz, min={audio_tensor.min():.3f}, max={audio_tensor.max():.3f}")
        
        # Encode audio to SNAC tokens
        logger.info(f"Encoding audio to SNAC tokens")
        codec = SNACCodec(device=TTS_DEVICE)
        audio_tokens = codec.encode_audio(audio_tensor, input_sample_rate=sample_rate, add_token_offsets=True)
        logger.info(f"Audio tokens encoded to {len(audio_tokens)} tokens")
        logger.info(f"First 10 tokens: {audio_tokens[:10]}")
        logger.info(f"Last 10 tokens: {audio_tokens[-10:]}")
        
        # Build zero-shot prompt (returns token IDs directly)
        prompt = svara_zero_shot_prompt(
            text=request_text,
            audio_tokens=audio_tokens,
            transcript=request_reference_transcript,
        )
        if isinstance(prompt, list):
            logger.info(f"Prompt built: {len(prompt)} token IDs")
            logger.info(f"Token ID preview (first 50): {prompt[:50]}")
            logger.info(f"Token ID preview (last 50): {prompt[-50:]}")
        else:
            logger.info(f"Prompt built (length: {len(prompt)} chars)")
            logger.info(f"Prompt preview (first 500 chars): {prompt[:500]}")
            logger.info(f"Prompt preview (last 200 chars): {prompt[-200:]}")
    else:
        # Standard TTS mode - build standard prompt
        if not request_voice:
            raise HTTPException(
                status_code=400,
                detail="'voice' parameter is required for standard TTS mode"
            )
        
        prompt = svara_prompt(request_text, request_voice)
        logger.info(f"Standard prompt built (length: {len(prompt)} chars)")
        logger.info(f"Prompt: {prompt}")
    
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
    
    # Handle streaming vs non-streaming
    if request_stream:
        # Streaming response
        async def audio_stream():
            """Stream audio chunks as they're generated."""
            try:
                async for chunk in request_orchestrator.astream(request_text, prompt=prompt, **gen_kwargs):
                    yield chunk
            except Exception as e:
                print(f"Error during streaming: {e}")
                raise
        
        return StreamingResponse(
            audio_stream(),
            media_type="audio/pcm",
            headers={
                "Content-Type": "audio/pcm",
                "X-Sample-Rate": "24000",
                "X-Bit-Depth": "16",
                "X-Channels": "1",
            }
        )
    else:
        # Non-streaming: collect all audio chunks
        try:
            audio_chunks = []
            async for chunk in request_orchestrator.astream(request_text, prompt=prompt, **gen_kwargs):
                audio_chunks.append(chunk)
            
            complete_audio = b"".join(audio_chunks)
            
            return Response(
                content=complete_audio,
                media_type="audio/pcm",
                headers={
                    "Content-Type": "audio/pcm",
                    "X-Sample-Rate": "24000",
                    "X-Bit-Depth": "16",
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

