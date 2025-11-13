
from __future__ import annotations
import logging
from typing import List, Literal, Optional, Tuple, Union
from langcodes import Language
import torch
import torchaudio
from io import BytesIO

logger = logging.getLogger(__name__)

BEGIN_OF_TEXT   = "<|begin_of_text|>" # 128000
END_OF_TEXT     = "<|end_of_text|>" # 128001
AUDIO           = "<|audio|>" # 156939
EOT_ID          = "<|eot_id|>" # 128009
START_OF_SPEECH = "<custom_token_1>" # 128257
END_OF_SPEECH   = "<custom_token_2>" # 128258
START_OF_HUMAN  = "<custom_token_3>" # 128259
END_OF_HUMAN    = "<custom_token_4>" # 128260
START_OF_AI     = "<custom_token_5>" # 128261
END_OF_AI       = "<custom_token_6>" # 128262
PAD_TOKEN       = "<custom_token_7>" # 128263


_DEFAULT_SEPARATORS = [
    "\n\n",   # paragraphs
    "\n",     # lines
    "। ",      # Hindi danda (sentence end)
    ". ", "? ", "! ", "… ",  # sentence enders
    ",",      # comma only if no space available
    " ",      # space (preferred over comma)
    "",       # hard fallback (character-level)
]



def svara_prompt(text: str, speaker_id: str) -> str:
    """Format the prompt for the Svara-TTS model.
    
    Args:
        text: The text to synthesize.
        speaker_id: The speaker identifier in "Language (Gender)" format (e.g., "Hindi (Male)").
    """
    return f"{START_OF_HUMAN}{AUDIO} {speaker_id}: {text}{EOT_ID}{END_OF_HUMAN}{START_OF_AI}"


def create_speaker_id(lang_code: str, gender: Literal["male", "female"]) -> str:
    """
    Create a speaker ID from language code and gender.
    
    Args:
        lang_code: An ISO 639-1 language code.
        gender: The gender of the voice.
    
    Returns:
        Speaker ID in "Language (Gender)" format (e.g., "Hindi (Male)").
    """
    language = Language.get(lang_code).display_name()
    return f"{language} ({gender.capitalize()})"


def svara_zero_shot_prompt(
    text: str, 
    audio_tokens: List[int], 
    transcript: Optional[str] = None,
) -> Union[str, List[int]]:
    """
    Format the zero-shot voice cloning prompt for the Svara-TTS model.    
    Args:
        text: The target text to synthesize.
        audio_tokens: SNAC token sequence from the reference audio (with offsets).
        transcript: Optional transcript of the reference audio. Including this
                   improves voice cloning quality.
    
    Returns:
    """

    # Correct token ID -> content mapping (from tokenizer config):
    # 128259 -> <custom_token_3>, 128009 -> <|eot_id|>, 128260 -> <custom_token_4>,
    # 128261 -> <custom_token_5>, 128257 -> <custom_token_1>, 128258 -> <custom_token_2>,
    # 128262 -> <custom_token_6>, 128266+ -> audio tokens
    audio_token_str = "".join([f"<custom_token_{token}>" for token in audio_tokens])
    previous_audio_prompt = f"{START_OF_SPEECH}{audio_token_str}{END_OF_SPEECH}"
    previous_text_prompt  = f"{START_OF_HUMAN}{AUDIO} {transcript}{EOT_ID}{END_OF_HUMAN}" if transcript and transcript.strip() else ""
    current_text_prompt   = f"{START_OF_HUMAN}{AUDIO} {text}{EOT_ID}{END_OF_HUMAN}{START_OF_AI}"
    
    print(f"Previous text prompt: {previous_text_prompt}")
    print(f"Previous audio prompt: {previous_audio_prompt}")
    print(f"Current text prompt: {current_text_prompt}")
    
    zero_shot_prompt = ''.join([
        previous_text_prompt, 
        previous_audio_prompt, 
        current_text_prompt
        ])

    return zero_shot_prompt


def _split_text_recursive(
    text: str,
    max_len: int,
    overlap: int,
    separators: List[str],
) -> List[str]:
    """
    Recursively split text using a hierarchy of separators.
    
    Args:
        text: Text to split
        max_len: Maximum chunk size
        overlap: Number of characters to overlap between chunks
        separators: List of separators to try in order of preference
    """
    if len(text) <= max_len:
        return [text]
    
    chunks: List[str] = []
    
    # Try each separator in order
    for separator in separators:
        if separator == "":
            # Fallback: character-level split
            break
            
        # Split by the current separator
        if separator in text:
            parts = text.split(separator)
            current_chunk = ""
            
            for i, part in enumerate(parts):
                # Re-add separator (except for last part)
                if i < len(parts) - 1:
                    part_with_sep = part + separator
                else:
                    part_with_sep = part
                
                # If this part alone is too long, recursively split it
                if len(part_with_sep) > max_len:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                    # Recursively split with remaining separators
                    remaining_seps = separators[separators.index(separator) + 1:]
                    if remaining_seps:
                        sub_chunks = _split_text_recursive(
                            part_with_sep, max_len, overlap, remaining_seps
                        )
                        chunks.extend(sub_chunks)
                    else:
                        # Hard split if no more separators
                        chunks.extend(_hard_split(part_with_sep, max_len, overlap))
                    continue
                
                # Check if adding this part would exceed max_len
                if len(current_chunk) + len(part_with_sep) > max_len:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        # Start new chunk with overlap
                        if overlap > 0 and len(current_chunk) >= overlap:
                            current_chunk = current_chunk[-overlap:] + part_with_sep
                        else:
                            current_chunk = part_with_sep
                    else:
                        current_chunk = part_with_sep
                else:
                    current_chunk += part_with_sep
            
            # Add remaining chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            return [c for c in chunks if c]
    
    # If no separator worked, do hard split
    return _hard_split(text, max_len, overlap)


def _hard_split(text: str, max_len: int, overlap: int) -> List[str]:
    """Split text at character boundaries when no separator is found."""
    chunks: List[str] = []
    start = 0
    
    while start < len(text):
        end = min(start + max_len, len(text))
        chunks.append(text[start:end])
        start = end - overlap if overlap > 0 else end
        
        # Prevent infinite loop
        if start >= len(text) or (overlap > 0 and start == end - overlap and end == len(text)):
            break
    
    return chunks


def chunk_text(
    text: str,
    max_len: int = 128,
    overlap: int = 0,
    separators: List[str] | None = None,
) -> List[str]:
    """
    Split text into chunks using a hierarchy of separators.

    Args:
        text: input text
        max_len: desired max chunk size (characters)
        overlap: desired overlap between chunks (characters)
        separators: override the default separator preference order
    """
    seps = separators or _DEFAULT_SEPARATORS
    
    if not text:
        return []
    
    if len(text) <= max_len:
        return [text]
    
    chunks = _split_text_recursive(text, max_len, overlap, seps)
    return [c.strip() for c in chunks if c.strip()]


# ============================================================================
# Audio Processing Utilities
# ============================================================================

def load_audio_from_bytes(
    audio_bytes: bytes,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, int]:
    """
    Load audio from bytes (WAV, MP3, FLAC, OGG, etc.) into a torch tensor.
    
    Automatically handles:
    - Multiple audio formats (via torchaudio backend)
    - Stereo to mono conversion
    - Returns float32 tensor normalized to [-1, 1]
    
    Args:
        audio_bytes: Raw audio file bytes (any format supported by torchaudio)
        device: Device to load tensor on ('cuda', 'mps', 'cpu', or None for auto-detect).
                Auto-detect tries cuda -> mps -> cpu.
    
    Returns:
        Tuple of (audio_tensor, sample_rate):
        - audio_tensor: Float32 tensor of shape (samples,) in range [-1, 1]
        - sample_rate: Sample rate in Hz
    
    Raises:
        RuntimeError: If audio format is invalid or cannot be loaded
        
    Example:
        >>> with open("audio.wav", "rb") as f:
        ...     audio_bytes = f.read()
        >>> audio, sr = load_audio_from_bytes(audio_bytes)
        >>> audio.shape, sr
        (torch.Size([48000]), 24000)
    """
    # Handle device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    try:
        # Load audio from bytes using BytesIO
        audio_buffer = BytesIO(audio_bytes)
        waveform, sample_rate = torchaudio.load(audio_buffer)
        
        # Convert stereo to mono if needed (average channels)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=False)
        else:
            # Squeeze channel dimension for mono
            waveform = waveform.squeeze(0)
        
        # Move to device and ensure float32
        waveform = waveform.to(device=device, dtype=torch.float32)
        
        return waveform, sample_rate
        
    except Exception as e:
        raise RuntimeError(f"Failed to load audio from bytes: {str(e)}") from e


def resample_audio(
    audio: torch.Tensor,
    original_sr: int,
    target_sr: int,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Resample audio to a target sample rate using torchaudio.
    
    Supports GPU acceleration for faster processing on CUDA/MPS devices.
    
    Args:
        audio: Audio tensor of shape (channels, samples) or (samples,).
               If 1D, will be converted to (1, samples).
        original_sr: Original sample rate in Hz.
        target_sr: Target sample rate in Hz.
        device: Device to use ('cuda', 'mps', 'cpu', or None for auto-detect).
                Auto-detect tries cuda -> mps -> cpu.
    
    Returns:
        Resampled audio tensor with shape matching input (channels, samples).
    
    Example:
        >>> audio = torch.randn(1, 48000)  # 1 second at 48kHz
        >>> resampled = resample_audio(audio, 48000, 24000)
        >>> resampled.shape
        torch.Size([1, 24000])
    """
    # Handle device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    # Ensure 2D tensor (channels, samples)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Move to device
    audio = audio.to(device)
    
    # No resampling needed if sample rates match
    if original_sr == target_sr:
        return audio.squeeze(0) if squeeze_output else audio
    
    # Create resampler and apply
    resampler = torchaudio.transforms.Resample(
        orig_freq=original_sr,
        new_freq=target_sr,
    ).to(device)
    
    resampled = resampler(audio)
    
    # Return with original shape
    return resampled.squeeze(0) if squeeze_output else resampled


def change_audio_speed(
    audio: torch.Tensor,
    speed_factor: float,
    sample_rate: int,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Change audio playback speed without altering pitch.
    
    Uses resampling to adjust speed. Values > 1.0 speed up, < 1.0 slow down.
    This implementation changes both speed and pitch together (like tape speed).
    
    Args:
        audio: Audio tensor of shape (channels, samples) or (samples,).
               If 1D, will be converted to (1, samples).
        speed_factor: Speed multiplier. 1.0 = original speed, 1.5 = 1.5x faster,
                     0.75 = slower (0.75x speed).
        sample_rate: Sample rate of the input audio in Hz.
        device: Device to use ('cuda', 'mps', 'cpu', or None for auto-detect).
                Auto-detect tries cuda -> mps -> cpu.
    
    Returns:
        Speed-adjusted audio tensor with shape matching input (channels, samples).
        The returned audio will have fewer samples if sped up, more if slowed down.
    
    Example:
        >>> audio = torch.randn(1, 24000)  # 1 second at 24kHz
        >>> faster = change_audio_speed(audio, 1.5, 24000)
        >>> faster.shape  # ~0.67 seconds of audio
        torch.Size([1, 16000])
    """
    # Handle device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    # Ensure 2D tensor (channels, samples)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Move to device
    audio = audio.to(device)
    
    # No speed change needed
    if speed_factor == 1.0:
        return audio.squeeze(0) if squeeze_output else audio
    
    # Calculate the target sample rate
    # Speed up: increase sample rate (fewer samples for same duration)
    # Slow down: decrease sample rate (more samples for same duration)
    target_sr = int(sample_rate * speed_factor)
    
    # Use functional API for one-time resampling
    adjusted = torchaudio.functional.resample(
        audio,
        orig_freq=sample_rate,
        new_freq=target_sr,
    )
    
    # Return with original shape
    return adjusted.squeeze(0) if squeeze_output else adjusted


def normalize_audio_volume(
    audio: torch.Tensor,
    target_peak: float = 0.95,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Normalize audio volume to a target peak amplitude.
    
    Scales the audio so its maximum absolute value reaches the target peak.
    This prevents clipping while maximizing volume. Default target of 0.95
    leaves some headroom to avoid clipping.
    
    Args:
        audio: Audio tensor of shape (channels, samples) or (samples,).
               If 1D, will be converted to (1, samples).
        target_peak: Target peak amplitude (0.0 to 1.0). Default 0.95 to
                    leave headroom and avoid clipping. Use 1.0 for maximum
                    volume (risk of clipping).
        device: Device to use ('cuda', 'mps', 'cpu', or None for auto-detect).
                Auto-detect tries cuda -> mps -> cpu.
    
    Returns:
        Normalized audio tensor with shape matching input (channels, samples).
        Peak amplitude will be scaled to target_peak.
    
    Example:
        >>> audio = torch.randn(1, 24000) * 0.3  # Quiet audio
        >>> normalized = normalize_audio_volume(audio, target_peak=0.95)
        >>> normalized.abs().max()  # Should be close to 0.95
        tensor(0.9500)
    """
    # Handle device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    # Ensure 2D tensor (channels, samples)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Move to device
    audio = audio.to(device)
    
    # Find the current peak amplitude
    current_peak = audio.abs().max()
    
    # Avoid division by zero
    if current_peak == 0:
        return audio.squeeze(0) if squeeze_output else audio
    
    # Calculate scaling factor and normalize
    scale = target_peak / current_peak
    normalized = audio * scale
    
    # Return with original shape
    return normalized.squeeze(0) if squeeze_output else normalized