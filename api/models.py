"""
Pydantic models for API request/response schemas.

Contains all data models used by the Svara TTS API endpoints.
"""
from __future__ import annotations
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


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
    """Request model for text-to-speech endpoint.
    
    Supports two modes:
    1. Standard TTS: Provide 'voice' parameter
    2. Zero-shot cloning: Provide 'reference_audio' (and optionally 'reference_transcript')
    """
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    voice: Optional[str] = Field(None, description="Voice in 'Language (Gender)' format (e.g., 'Hindi (Male)', 'English (Female)'). Required for standard TTS, not used in zero-shot mode.")
    model_id: str = Field(default="svara-tts-v1", description="Model to use for synthesis")
    stream: bool = Field(default=True, description="Stream audio response")
    
    # Zero-shot voice cloning parameters
    reference_audio: Optional[bytes] = Field(None, description="Reference audio bytes (WAV, MP3, FLAC, OGG, etc.) for zero-shot voice cloning. When provided, 'voice' parameter is ignored.")
    reference_transcript: Optional[str] = Field(None, description="Optional transcript of the reference audio. Providing this improves voice cloning quality. Only used when reference_audio is provided.")
    
    # Generation parameters (optional)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Sampling temperature (default: 0.75)")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Nucleus sampling probability (default: 0.9)")
    top_k: Optional[int] = Field(None, ge=-1, description="Top-k sampling (default: -1, disabled)")
    repetition_penalty: Optional[float] = Field(None, ge=1.0, le=2.0, description="Repetition penalty (default: 1.1)")
    max_tokens: Optional[int] = Field(None, ge=1, le=4096, description="Maximum tokens to generate (default: 2048)")
    
    # Future features (not implemented yet)
    voice_settings: Dict[str, Any] = Field(default_factory=dict, description="Voice settings (not implemented yet)")
    text_normalization: bool = Field(default=False, description="Enable text normalization (not implemented yet)")

