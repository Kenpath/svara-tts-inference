# Refactoring Notes - Speaker ID Implementation

## Overview

Refactored the `SvaraTTSOrchestrator` to accept a `speaker_id` parameter directly instead of constructing it internally from `lang_code` and `gender`. This makes the system more flexible and prepares it for future voice models (like svara-tts-v2) that don't follow the language+gender pattern.

## Changes Made

### 1. `tts_engine/utils.py`

**Before:**
```python
def svara_prompt(text: str, lang_code: str, gender: Literal["male", "female"]) -> str:
    language = Language.get(lang_code).display_name()    
    gender   = gender.capitalize()
    voice   = f"{language} ({gender})"        
    base = f"<|audio|> {voice}: {text}<|eot_id|>"
    return "<custom_token_3>" + base + "<custom_token_4><custom_token_5>"
```

**After:**
```python
def svara_prompt(text: str, speaker_id: str) -> str:
    """Format the prompt for the Svara-TTS model."""
    base = f"<|audio|> {speaker_id}: {text}<|eot_id|>"
    return "<custom_token_3>" + base + "<custom_token_4><custom_token_5>"

def create_speaker_id(lang_code: str, gender: Literal["male", "female"]) -> str:
    """Create a speaker ID from language code and gender."""
    language = Language.get(lang_code).display_name()
    return f"{language} ({gender.capitalize()})"
```

**Changes:**
- Simplified `svara_prompt()` to accept `speaker_id` directly
- Added `create_speaker_id()` helper function for backward compatibility
- Speaker ID construction logic is now decoupled from prompt formatting

### 2. `tts_engine/orchestrator.py`

**Before:**
```python
def __init__(self,
             base_url: str,
             model: str = "kenpath/svara-tts-v1",
             lang_code: str = "en",
             gender: Literal["male", "female"] = "male",
             ...):
    self.lang_code = lang_code
    self.gender = gender
```

**After:**
```python
def __init__(self,
             base_url: str,
             model: str = "kenpath/svara-tts-v1",
             speaker_id: Optional[str] = None,
             lang_code: str = "en",
             gender: Literal["male", "female"] = "male",
             ...):
    if speaker_id is None:
        self.speaker_id = create_speaker_id(lang_code, gender)
    else:
        self.speaker_id = speaker_id
```

**Changes:**
- Added `speaker_id` parameter (optional, for direct specification)
- Kept `lang_code` and `gender` for backward compatibility
- If `speaker_id` is provided, it's used directly; otherwise, it's constructed from `lang_code` and `gender`
- Updated `_stream_one()` and `_astream_one()` to use `self.speaker_id`

### 3. `tts_engine/voice_config.py`

**New Function:**
```python
def get_speaker_id(voice_id: str) -> str:
    """Get the speaker ID for a given voice."""
    voice = get_voice(voice_id)
    if voice is None:
        raise ValueError(f"Voice ID '{voice_id}' not found")
    
    # For v1 voices, construct from language and gender
    if voice.model_id == "svara-tts-v1":
        from .utils import create_speaker_id
        return create_speaker_id(voice.language_code, voice.gender)
    
    # For v2+ voices, use the voice name as speaker ID
    return voice.name
```

**Changes:**
- Added `get_speaker_id()` function to get speaker ID from voice ID
- For v1 voices: Returns "Language (Gender)" format (e.g., "Hindi (Male)")
- For v2+ voices: Returns the voice name directly (e.g., "Rohit", "Priya")
- This enables flexibility for future voice models

### 4. `api/server.py`

**Before:**
```python
lang_code, gender = parse_voice_for_v1(request.voice_id)

request_orchestrator = SvaraTTSOrchestrator(
    base_url=VLLM_BASE_URL,
    model=VLLM_MODEL,
    lang_code=lang_code,
    gender=gender,
    ...
)
```

**After:**
```python
speaker_id = get_speaker_id(request.voice_id)

# Allow language_code override if provided
if request.language_code and voice.gender:
    from tts_engine.utils import create_speaker_id
    speaker_id = create_speaker_id(request.language_code, voice.gender)

request_orchestrator = SvaraTTSOrchestrator(
    base_url=VLLM_BASE_URL,
    model=VLLM_MODEL,
    speaker_id=speaker_id,
    ...
)
```

**Changes:**
- Replaced `parse_voice_for_v1()` with `get_speaker_id()`
- Pass `speaker_id` directly to orchestrator
- Maintain language_code override functionality

## Benefits

1. **Flexibility**: Can now use any speaker ID format, not just language+gender combinations
2. **Future-Proof**: Ready for svara-tts-v2 with custom voice profiles (Rohit, Priya, etc.)
3. **Cleaner Architecture**: Speaker ID construction is centralized in `voice_config.py`
4. **Backward Compatible**: Existing code using `lang_code` and `gender` still works
5. **Easier Testing**: Can directly pass speaker IDs for testing without language/gender parsing

## Usage Examples

### Direct Speaker ID
```python
orchestrator = SvaraTTSOrchestrator(
    base_url="http://localhost:8000/v1",
    model="kenpath/svara-tts-v1",
    speaker_id="Hindi (Male)"
)
```

### Backward Compatible
```python
orchestrator = SvaraTTSOrchestrator(
    base_url="http://localhost:8000/v1",
    model="kenpath/svara-tts-v1",
    lang_code="hi",
    gender="male"
)
# Internally creates speaker_id = "Hindi (Male)"
```

### Future V2 Voices
```python
orchestrator = SvaraTTSOrchestrator(
    base_url="http://localhost:8000/v1",
    model="kenpath/svara-tts-v2",
    speaker_id="Rohit"  # Custom voice name
)
```

## Migration Guide

Existing code will continue to work without changes. To take advantage of the new feature:

```python
# Old way (still works)
orchestrator = SvaraTTSOrchestrator(..., lang_code="hi", gender="male")

# New way (more flexible)
orchestrator = SvaraTTSOrchestrator(..., speaker_id="Hindi (Male)")

# For v2 voices (future)
orchestrator = SvaraTTSOrchestrator(..., speaker_id="Rohit")
```

## Testing

All refactored code has been linted and no errors were found. The changes maintain backward compatibility while adding new functionality.

