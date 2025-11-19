# tts_engine/codec.py
from snac import SNAC
from typing import List, Optional
import numpy as np
import torch
import os
from transformers import AutoTokenizer
from .utils import resample_audio
from .timing import track_time
from .constants import AUDIO_TOKEN_OFFSETS


# Global model cache to avoid reloading SNAC model for each instance
_SNAC_MODEL_CACHE: dict[str, SNAC] = {}

# Global tokenizer cache to avoid reloading tokenizer for each request
_TOKENIZER_CACHE: dict[str, AutoTokenizer] = {}


def _get_or_load_snac_model(device: str, model_name: str = "hubertsiuzdak/snac_24khz") -> SNAC:
    """
    Get cached SNAC model or load it if not cached.
    
    This prevents repeated model loading when creating multiple codec instances.
    Models are cached per device to handle multi-GPU scenarios.
    
    Args:
        device: Device to load model on ('cuda', 'mps', 'cpu')
        model_name: HuggingFace model identifier
    
    Returns:
        Cached or newly loaded SNAC model
    """
    cache_key = f"{model_name}_{device}"
    
    if cache_key not in _SNAC_MODEL_CACHE:
        print(f"[DEBUG] Loading SNAC model: {model_name} on device: {device}")
        model = SNAC.from_pretrained(model_name).eval().to(device)
        print(f"[DEBUG] SNAC model loaded. Type: {type(model)}, Config: {model.config if hasattr(model, 'config') else 'N/A'}")
        _SNAC_MODEL_CACHE[cache_key] = model
    else:
        print(f"[DEBUG] Using cached SNAC model: {cache_key}")
    
    return _SNAC_MODEL_CACHE[cache_key]


def get_or_load_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Get cached tokenizer or load it if not cached.
    
    This prevents repeated tokenizer loading when processing multiple requests.
    Automatically uses HF_TOKEN environment variable if available for private models.
    
    Args:
        model_name: HuggingFace model identifier
    
    Returns:
        Cached or newly loaded tokenizer
    """
    if model_name not in _TOKENIZER_CACHE:
        print(f"[DEBUG] Loading tokenizer: {model_name}")
        
        # Check for HuggingFace token for private model access
        hf_token = os.getenv("HF_TOKEN")
        
        if hf_token:
            print(f"[DEBUG] Using HF_TOKEN for authentication")
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"[DEBUG] Tokenizer loaded. Vocab size: {len(tokenizer)}")
        _TOKENIZER_CACHE[model_name] = tokenizer
    else:
        print(f"[DEBUG] Using cached tokenizer: {model_name}")
    
    return _TOKENIZER_CACHE[model_name]


class SNACCodec:
    """
    Unified SNAC codec for encoding audio to tokens and decoding tokens to audio.
    
    Supports both:
    - Encoding: audio waveform → SNAC tokens (for zero-shot voice cloning)
    - Decoding: SNAC tokens → PCM16 audio (for TTS synthesis)
    
    Uses a global model cache to avoid reloading the SNAC model when creating
    multiple instances, which significantly improves initialization time.
    """
    
    def __init__(self, device: Optional[str] = None, model_name: str = "hubertsiuzdak/snac_24khz"):
        """
        Initialize SNAC codec.
        
        Args:
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto-detect)
            model_name: HuggingFace model identifier for SNAC
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        self.model_name = model_name
        self.sample_rate = 24000  # SNAC 24kHz model
        
        # Get or load model from cache
        self.model = _get_or_load_snac_model(device, model_name)

    @track_time("SNAC.encode_audio")
    def encode_audio(
        self,
        audio: torch.Tensor,
        input_sample_rate: int = 24000,
        add_token_offsets: bool = True
    ) -> List[int]:
        """
        Encode audio waveform to SNAC tokens for zero-shot voice cloning.
        
        This method takes audio and converts it to a sequence of tokens that
        can be used as a voice reference in the Svara-TTS model.
        Automatically resamples audio to 24kHz if needed.
        
        Args:
            audio: Audio tensor of shape (channels, samples) or (samples,).
                   If 1D, will be converted to (1, 1, samples) for SNAC.
            input_sample_rate: Sample rate of the input audio in Hz. If not 24000,
                              audio will be automatically resampled to 24kHz.
            add_token_offsets: If True, adds Svara-TTS token offsets (128266 + vocab offsets)
                              to make tokens ready for model input. If False, returns raw
                              SNAC codes in range [0, 4096].
        
        Returns:
            List of token IDs. Length will be 7 * num_frames where num_frames depends
            on input audio length. Each frame represents ~10ms of audio.
        
        Example:
            >>> codec = SNACCodec()
            >>> # Load 1 second of audio at 48kHz - will be resampled to 24kHz
            >>> audio = torch.randn(48000)
            >>> tokens = codec.encode_audio(audio, input_sample_rate=48000)
            >>> len(tokens) # ~7 tokens per frame, ~100 frames per second
            700
        """
        # Resample to 24kHz if needed
        if input_sample_rate != self.sample_rate:            
            audio = resample_audio(audio, input_sample_rate, self.sample_rate, self.device)
        
        # Debug: Check audio after resampling
        print(f"[DEBUG] Audio shape after resample: {audio.shape}")
        
        # Ensure proper shape: SNAC expects (batch, channels, samples)
        if audio.dim() == 1:
            # (samples,) -> (1, 1, samples)
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            # (channels, samples) -> (1, channels, samples)
            audio = audio.unsqueeze(0)
        
        # Move to device and ensure float32
        audio = audio.to(dtype=torch.float32, device=self.device)
        
        print(f"[DEBUG] Audio shape going into SNAC encode: {audio.shape}")
        
        # Encode with SNAC
        with torch.inference_mode():
            codes = self.model.encode(audio)
        
        print(f"[DEBUG] SNAC codes shapes: codes[0]={codes[0].shape}, codes[1]={codes[1].shape}, codes[2]={codes[2].shape}")
        
        # SNAC produces hierarchical codes with different temporal resolutions:
        # codes[0]: coarsest (e.g., 100 frames for 1 sec)
        # codes[1]: 2x finer (e.g., 200 frames)
        # codes[2]: 4x finer (e.g., 400 frames)
        # 
        # Interleave pattern per finest-resolution frame:
        # For every 4 frames in codes[2], we get:
        # - 1 code from codes[0]
        # - 2 codes from codes[1] 
        # - 4 codes from codes[2]
        # Output order per coarse frame: c0, c1, c2, c3, c4, c5, c6
        
        all_codes = []
        num_coarse_frames = codes[0].shape[1]
        
        for i in range(num_coarse_frames):
            # Get indices for hierarchical codes
            # Each coarse frame i corresponds to:
            # - codes[0][i]
            # - codes[1][2*i : 2*i+2] (2 codes)
            # - codes[2][4*i : 4*i+4] (4 codes)
            
            c0 = codes[0][0][i].item()
            c1 = codes[1][0][2 * i].item()
            c2 = codes[2][0][4 * i].item()
            c3 = codes[2][0][4 * i + 1].item()
            c4 = codes[1][0][2 * i + 1].item()
            c5 = codes[2][0][4 * i + 2].item()
            c6 = codes[2][0][4 * i + 3].item()
            
            if add_token_offsets:
                # Add Svara-TTS vocabulary offsets using constants
                all_codes.append(c0 + AUDIO_TOKEN_OFFSETS[0])
                all_codes.append(c1 + AUDIO_TOKEN_OFFSETS[1])
                all_codes.append(c2 + AUDIO_TOKEN_OFFSETS[2])
                all_codes.append(c3 + AUDIO_TOKEN_OFFSETS[3])
                all_codes.append(c4 + AUDIO_TOKEN_OFFSETS[4])
                all_codes.append(c5 + AUDIO_TOKEN_OFFSETS[5])
                all_codes.append(c6 + AUDIO_TOKEN_OFFSETS[6])
            else:
                # Raw SNAC codes
                all_codes.extend([c0, c1, c2, c3, c4, c5, c6])
        
        return all_codes
    
    @track_time("SNAC.decode_window")
    def decode_window(self, window: List[int]) -> bytes:
        """
        Decode a sliding window of Svara-TTS codes into PCM16 bytes.
        
        Args:
            window: Flat list of int codes, length multiple of 7 (>= 28 recommended).
                   These should be raw SNAC codes in range [0, 4096], NOT with
                   token offsets added.
        
        Returns:
            PCM16 mono bytes; empty bytes if invalid input.
        """
        if not window or len(window) < 7:
            return b""
        
        # Use only full frames
        F = len(window) // 7
        frame = window[: F * 7]
        
        # Build code streams: [c0], [c1,c4], [c2,c3,c5,c6]
        t = torch.tensor(frame, dtype=torch.int32, device=self.device)
        t = t.view(F, 7)
        
        codes_0 = t[:, 0].reshape(1, -1)
        codes_1 = t[:, [1, 4]].reshape(1, -1)
        codes_2 = t[:, [2, 3, 5, 6]].reshape(1, -1)
        
        # Validate range [0, 4096]
        if (
            torch.any((codes_0 < 0) | (codes_0 > 4096)) or
            torch.any((codes_1 < 0) | (codes_1 > 4096)) or
            torch.any((codes_2 < 0) | (codes_2 > 4096))
        ):            
            return b""
        
        with torch.inference_mode():
            audio = self.model.decode([codes_0, codes_1, codes_2])  # [1, 1, T]
            # Keep the synthesis region (matches SNAC examples)
            audio = audio[:, :, 2048:4096]
        
        x = audio.detach().float().cpu().numpy().reshape(-1)
        print(x.shape)
        pcm16 = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
        return pcm16.tobytes()