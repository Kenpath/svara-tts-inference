# tts_engine/decoder_snac.py
from snac import SNAC
from typing import List, Optional
import numpy as np
import torch
from .timing import track_time

class SNACDecoder:
    """
    Decodes Svara-TTS 7-code frames to PCM16 using SNAC (24 kHz). Accepts sliding 28-token windows (4 frames) and, by default, returns only the last hop (hop-only streaming) for low-latency playback.
    """
    def __init__(self, device: Optional[str] = None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device      = device
        self.model       = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(self.device)
        self.sample_rate = 24000 # Default sample rate for Svara-TTS

    @track_time("SNAC.decode_window")
    def decode_window(self, window: List[int]) -> bytes:
        """
        Decode a sliding window of Svara-TTS codes into PCM16 bytes.

        Args:
            window: flat list of int codes, length multiple of 7 (>= 28 recommended).

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
        # (Use &/| with parentheses to avoid precedence gotchas.)
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
        pcm16 = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
        return pcm16.tobytes()