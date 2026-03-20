"""
Pydantic models for API request/response schemas.

Contains all data models used by the Svara TTS API endpoints.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field, field_validator
import base64


class VoiceResponse(BaseModel):
    """Voice metadata response."""
    voice_id: str
    name: str
    model_id: str
    gender: Optional[str] = None
    description: Optional[str] = None


class VoicesResponse(BaseModel):
    """Response for GET /v1/voices endpoint."""
    voices: list[VoiceResponse]


class OpenAISpeechRequest(BaseModel):
    """
    OpenAI-compatible speech request (POST /v1/audio/speech).

    Core fields match the OpenAI TTS API. Extended fields (generation params,
    zero-shot cloning) can be passed via the OpenAI SDK's `extra_body` parameter.
    """
    # --- OpenAI-standard fields ---
    model: str = Field(default="svara-tts-v1", description="Model name (accepted but not used for model selection)")
    input: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    voice: str = Field(..., description="Voice ID (e.g. 'hi_male', 'en_female') or speaker name (e.g. 'Hindi (Male)')")
    response_format: str = Field(default="mp3", description="Audio format: mp3, opus, aac, wav, pcm")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Playback speed (accepted but not yet implemented)")

    # --- Extended fields ---
    stream: bool = Field(default=False, description="Stream audio response")

    # Zero-shot voice cloning
    reference_audio: Optional[bytes] = Field(None, description="Reference audio as base64-encoded string for zero-shot voice cloning. Supports WAV, MP3, FLAC, OGG, etc.")
    reference_transcript: Optional[str] = Field(None, description="Transcript of the reference audio (improves cloning quality)")

    # Generation parameters
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Sampling temperature (default: 0.75)")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Nucleus sampling probability (default: 0.9)")
    top_k: Optional[int] = Field(None, ge=-1, description="Top-k sampling (default: 40)")
    repetition_penalty: Optional[float] = Field(None, ge=1.0, le=2.0, description="Repetition penalty (default: 1.1)")
    max_tokens: Optional[int] = Field(None, ge=1, le=4096, description="Maximum tokens to generate (default: 2048)")

    @field_validator('reference_audio', mode='before')
    @classmethod
    def decode_reference_audio(cls, v):
        """Decode base64-encoded audio string to bytes."""
        if v is None:
            return None
        if isinstance(v, bytes):
            return v
        if isinstance(v, str):
            try:
                return base64.b64decode(v)
            except Exception as e:
                raise ValueError(f"Invalid base64 audio data: {str(e)}")
        raise ValueError("reference_audio must be a base64-encoded string")
