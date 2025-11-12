"""
FastAPI server for Svara TTS API.

Provides ElevenLabs-style text-to-speech endpoints with support for
Indian language voices and streaming audio generation.
"""
from __future__ import annotations
import os
import asyncio
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tts_engine.voice_config import get_all_voices
from tts_engine.orchestrator import SvaraTTSOrchestrator


# ============================================================================
# Configuration
# ============================================================================

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.getenv("VLLM_MODEL", "kenpath/svara-tts-v1")
TTS_DEVICE = os.getenv("TTS_DEVICE", None)  # None = auto-detect (CUDA/MPS/CPU)

# Global orchestrator instance (initialized in lifespan)
orchestrator: Optional[SvaraTTSOrchestrator] = None


# ============================================================================
# Request/Response Models
# ============================================================================

class VoiceResponse(BaseModel):
    """Voice metadata response."""
    voice_id: str
    name: str
    language_code: str
    model_id: str
    gender: Optional[str] = None
    description: Optional[str] = None


class VoicesResponse(BaseModel):
    """Response for GET /v1/voices endpoint."""
    voices: list[VoiceResponse]


class TTSRequest(BaseModel):
    """Request model for text-to-speech endpoint."""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    voice: str = Field(..., description="Voice in 'Language (Gender)' format (e.g., 'Hindi (Male)', 'English (Female)')")
    model_id: str = Field(default="svara-tts-v1", description="Model to use for synthesis")
    voice_settings: Dict[str, Any] = Field(default_factory=dict, description="Voice settings (not implemented yet)")
    text_normalization: bool = Field(default=False, description="Enable text normalization (not implemented yet)")
    reference_audio: Optional[bytes] = Field(None, description="Reference audio for cloning (not implemented yet)")
    stream: bool = Field(default=True, description="Stream audio response")


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
        hop_only=True,
        hop_samples=512,
        prebuffer_seconds=1.2,
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
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech with streaming or non-streaming response.
    
    Args:
        request: TTS request with text, voice, and options
    
    Returns:
        Raw PCM16 audio bytes (streaming or complete)
    """
    # Use the voice directly as speaker_id
    speaker_id = request.voice
    
    # Currently only v1 is implemented
    if request.model_id != "svara-tts-v1":
        raise HTTPException(
            status_code=501,
            detail=f"Model '{request.model_id}' is not yet implemented. Currently only 'svara-tts-v1' is supported."
        )
    
    # Create orchestrator instance for this request
    request_orchestrator = SvaraTTSOrchestrator(
        base_url=VLLM_BASE_URL,
        model=VLLM_MODEL,
        speaker_id=speaker_id,
        device=TTS_DEVICE,
        hop_only=False,  # Disable hop-only for full audio quality
        hop_samples=2048,  # Use full synthesis region
        prebuffer_seconds=0.5,
        concurrent_decode=True,
        max_workers=2,
    )
    
    # Handle streaming vs non-streaming
    if request.stream:
        # Streaming response
        async def audio_stream():
            """Stream audio chunks as they're generated."""
            try:
                async for chunk in request_orchestrator.astream(request.text):
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
            async for chunk in request_orchestrator.astream(request.text):
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

