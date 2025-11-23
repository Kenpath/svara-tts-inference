"""
Voice configuration system for Svara TTS API.

Manages voice profiles across different models with extensible structure
for future custom voice profiles.
"""
from __future__ import annotations
import os
import yaml
from typing import List, Dict, Optional, Literal
from dataclasses import dataclass, asdict

@dataclass
class Voice:
    """Voice profile with metadata."""
    voice_id: str
    name: str
    model_id: str
    languages: List[str]
    gender: Optional[Literal["male", "female"]] = None
    description: Optional[str] = None
    
    @property
    def language_code(self) -> str:
        """
        Get the primary language code.
        Maintained for backward compatibility.
        """
        if self.languages:
            return self.languages[0]
        raise ValueError(f"Voice {self.voice_id} has no languages defined")

    def to_dict(self) -> Dict:
        """Convert to dictionary for API responses."""
        data = {k: v for k, v in asdict(self).items() if v is not None}
        # Add language_code for backward compatibility in API response
        if self.languages:
            data['language_code'] = self.languages[0]
        return data


def load_voices_from_yaml(path: str) -> List[Voice]:
    """Load voices from a YAML file."""
    if not os.path.exists(path):
        # Return empty list if file not found to allow partial setup
        # Log warning in production
        print(f"Warning: Voice config file not found at {path}")
        return []
        
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
        
    voices = []
    if data:
        for item in data:
            voices.append(Voice(**item))
    return voices


# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICES_DIR = os.path.join(BASE_DIR, "assets", "voices")
V1_YAML_PATH = os.path.join(VOICES_DIR, "svara-tts-v1.yaml")
V2_YAML_PATH = os.path.join(VOICES_DIR, "svara-tts-v2.yaml")

# Load svara-tts-v1 voices from YAML
SVARA_V1_VOICES: List[Voice] = load_voices_from_yaml(V1_YAML_PATH)

# Load svara-tts-v2 voices from YAML
SVARA_V2_VOICES: List[Voice] = load_voices_from_yaml(V2_YAML_PATH)

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
    
    # For v2 and future voices, use the name directly (capitalized as in the YAML)
    # The name field in the voice object matches the key in the model's speaker list
    return voice.name
