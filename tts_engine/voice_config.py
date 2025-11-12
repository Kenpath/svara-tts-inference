"""
Voice configuration system for Svara TTS API.

Manages voice profiles across different models with extensible structure
for future custom voice profiles.
"""
from __future__ import annotations
from typing import List, Dict, Optional, Literal
from dataclasses import dataclass, asdict

@dataclass
class Voice:
    """Voice profile with metadata."""
    voice_id: str
    name: str
    language_code: str
    model_id: str
    gender: Optional[Literal["male", "female"]] = None
    description: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API responses."""
        return {k: v for k, v in asdict(self).items() if v is not None}


# Supported languages for svara-tts-v1
SVARA_V1_LANGUAGES = [
    ("hi", "Hindi"),
    ("bn", "Bengali"),
    ("mr", "Marathi"),
    ("te", "Telugu"),
    ("kn", "Kannada"),
    ("bh", "Bhojpuri"),
    ("mag", "Magahi"),
    ("hne", "Chhattisgarhi"),
    ("mai", "Maithili"),
    ("as", "Assamese"),
    ("brx", "Bodo"),
    ("doi", "Dogri"),
    ("gu", "Gujarati"),
    ("ml", "Malayalam"),
    ("pa", "Punjabi"),
    ("ta", "Tamil"),
    ("en", "English (Indian)"),
    ("ne", "Nepali"),
    ("sa", "Sanskrit"),
]

# Generate svara-tts-v1 voices (19 languages × 2 genders = 38 voices)
SVARA_V1_VOICES: List[Voice] = []
for lang_code, lang_name in SVARA_V1_LANGUAGES:
    for gender in ["male", "female"]:
        voice_id = f"{lang_code}_{gender}"
        SVARA_V1_VOICES.append(
            Voice(
                voice_id=voice_id,
                name=f"{lang_name} ({gender.capitalize()})",
                language_code=lang_code,
                model_id="svara-tts-v1",
                gender=gender,
                description=f"{lang_name} voice with {gender} characteristics"
            )
        )

# Placeholder voices for svara-tts-v2 (custom voice profiles)
SVARA_V2_VOICES: List[Voice] = [
    Voice(
        voice_id="rohit",
        name="Rohit",
        language_code="hi",
        model_id="svara-tts-v2",
        description="Deep, professional male voice"
    ),
    Voice(
        voice_id="priya",
        name="Priya",
        language_code="hi",
        model_id="svara-tts-v2",
        description="Warm, friendly female voice"
    ),
    Voice(
        voice_id="arjun",
        name="Arjun",
        language_code="en",
        model_id="svara-tts-v2",
        description="Energetic, youthful male voice"
    ),
    Voice(
        voice_id="ananya",
        name="Ananya",
        language_code="en",
        model_id="svara-tts-v2",
        description="Clear, articulate female voice"
    ),
    Voice(
        voice_id="vikram",
        name="Vikram",
        language_code="hi",
        model_id="svara-tts-v2",
        description="Authoritative, confident male voice"
    ),
    Voice(
        voice_id="kavya",
        name="Kavya",
        language_code="hi",
        model_id="svara-tts-v2",
        description="Gentle, soothing female voice"
    ),
]

# Combined voice registry
ALL_VOICES: List[Voice] = SVARA_V1_VOICES + SVARA_V2_VOICES

# Voice lookup dictionary for fast access
VOICE_REGISTRY: Dict[str, Voice] = {voice.voice_id: voice for voice in ALL_VOICES}


def get_all_voices(model_id: Optional[str] = None) -> List[Voice]:
    """
    Get all available voices, optionally filtered by model_id.
    
    Args:
        model_id: Optional model ID to filter voices (e.g., "svara-tts-v1")
    
    Returns:
        List of Voice objects
    """
    if model_id is None:
        return ALL_VOICES
    return [v for v in ALL_VOICES if v.model_id == model_id]


def get_voice(voice_id: str) -> Optional[Voice]:
    """
    Get a specific voice by ID.
    
    Args:
        voice_id: Voice identifier (e.g., "hi_male" or "rohit")
    
    Returns:
        Voice object if found, None otherwise
    """
    return VOICE_REGISTRY.get(voice_id)


def parse_voice_for_v1(voice_id: str) -> tuple[str, Literal["male", "female"]]:
    """
    Parse a v1 voice_id to extract language code and gender.
    
    Args:
        voice_id: Voice ID in format "{lang_code}_{gender}"
    
    Returns:
        Tuple of (language_code, gender)
    
    Raises:
        ValueError: If voice_id is invalid or not found
    """
    voice = get_voice(voice_id)
    if voice is None:
        raise ValueError(f"Voice ID '{voice_id}' not found")
    
    if voice.model_id != "svara-tts-v1":
        raise ValueError(f"Voice '{voice_id}' is not a svara-tts-v1 voice")
    
    if voice.gender is None:
        raise ValueError(f"Voice '{voice_id}' does not have gender information")
    
    return voice.language_code, voice.gender


def get_speaker_id(voice_id: str) -> str:
    """
    Get the speaker ID for a given voice.
    
    Args:
        voice_id: Voice identifier (e.g., "hi_male" or "rohit")
    
    Returns:
        Speaker ID string (e.g., "Hindi (Male)" for v1, "rohit" for v2)
    
    Raises:
        ValueError: If voice_id is invalid or not found
    """
    voice = get_voice(voice_id)
    if voice is None:
        raise ValueError(f"Voice ID '{voice_id}' not found")
    
    # For v1 voices, construct speaker ID from language and gender
    # (model was trained with "Language (Gender)" format)
    if voice.model_id == "svara-tts-v1":
        from .utils import create_speaker_id
        if voice.gender is None:
            raise ValueError(f"Voice '{voice_id}' does not have gender information")
        return create_speaker_id(voice.language_code, voice.gender)
    
    # For v2 and future voices, use the voice_id directly as speaker ID
    # This allows custom voice profiles
    return voice.voice_id

