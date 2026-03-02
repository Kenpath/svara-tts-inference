"""
Pydantic models for API request/response schemas.

Contains all data models used by the Svara TTS API endpoints.
"""
from __future__ import annotations
from typing import Optional, Dict, Any, Literal
from pydantic import AliasChoices, BaseModel, Field, field_validator


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


class TTSRequest(BaseModel):
    """Request model for text-to-speech endpoint."""
    transcript: str = Field(..., min_length=1, max_length=5000, description="Transcript text to synthesize")
    language: str = Field(..., min_length=2, max_length=8, description="ISO language code (e.g., 'en', 'hi', 'ta')")
    gender: Literal["male", "female"] = Field(..., description="Speaker gender")
    stream: bool = Field(default=True, description="Stream audio response")
    voice_name: Optional[str] = Field(
        default=None,
        description="Optional explicit voice name (used by v0.5 raw/modal-style models, e.g. 'Prakash')",
    )

    # Generation parameters (optional)
    response_format: str = Field(
        default="opus",
        validation_alias=AliasChoices("response_format", "audio_format"),
        description="Audio format for response. Options: 'opus' (default), 'mp3', 'aac', 'wav', 'pcm'",
    )
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Sampling temperature (default: 0.75)")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Nucleus sampling probability (default: 0.9)")
    top_k: Optional[int] = Field(None, ge=-1, description="Top-k sampling (default: -1, disabled)")
    repetition_penalty: Optional[float] = Field(None, ge=1.0, le=2.0, description="Repetition penalty (default: 1.1)")
    max_tokens: Optional[int] = Field(None, ge=1, le=4096, description="Maximum tokens to generate (default: 2048)")
    
    # Future features (not implemented yet)
    voice_settings: Dict[str, Any] = Field(default_factory=dict, description="Voice settings (not implemented yet)")
    text_normalization: bool = Field(default=False, description="Enable text normalization (not implemented yet)")

    @field_validator("language")
    @classmethod
    def normalize_language(cls, v: str) -> str:
        return v.strip().lower()
