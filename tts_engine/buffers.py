
from __future__ import annotations
from typing import Optional
import numpy as np


def crossfade_pcm(a: bytes, b: bytes, overlap_ms: int = 50, sample_rate: int = 24000) -> bytes:
    """
    Crossfade two PCM16 mono byte buffers.

    Applies a linear fade-out to the tail of `a` and fade-in to the head of `b`,
    then blends them. Returns the stitched result: a_head + blended_overlap + b_tail.

    Args:
        a: First PCM16 mono buffer.
        b: Second PCM16 mono buffer.
        overlap_ms: Crossfade duration in milliseconds.
        sample_rate: Audio sample rate.

    Returns:
        Stitched PCM16 mono bytes.
    """
    overlap_samples = int(sample_rate * overlap_ms / 1000)

    arr_a = np.frombuffer(a, dtype=np.int16).astype(np.float32)
    arr_b = np.frombuffer(b, dtype=np.int16).astype(np.float32)

    # If either buffer is shorter than the overlap, just concatenate
    if len(arr_a) < overlap_samples or len(arr_b) < overlap_samples:
        return a + b

    fade_out = np.linspace(1.0, 0.0, overlap_samples, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, overlap_samples, dtype=np.float32)

    blended = arr_a[-overlap_samples:] * fade_out + arr_b[:overlap_samples] * fade_in

    result = np.concatenate([
        arr_a[:-overlap_samples],
        blended,
        arr_b[overlap_samples:],
    ])

    return np.clip(result, -32768, 32767).astype(np.int16).tobytes()


class AudioBuffer:
    """Helper to manage prebuffering logic for streaming audio."""
    
    def __init__(self, prebuffer_samples: int):
        self.buf = bytearray()
        self.started = False
        self.prebuffer_samples = prebuffer_samples
    
    def process(self, pcm: bytes) -> Optional[bytes]:
        """
        Add PCM data and return bytes to yield (if any).
        
        Accumulates audio until prebuffer_samples is reached, then
        switches to passthrough mode.
        
        Args:
            pcm: PCM audio bytes (int16)
            
        Returns:
            Bytes to yield, or None if still buffering
        """
        if not pcm:
            return None
            
        if not self.started:
            self.buf.extend(pcm)
            # Each sample is 2 bytes (int16)
            if len(self.buf) // 2 >= self.prebuffer_samples:
                self.started = True
                result = bytes(self.buf)
                self.buf.clear()
                return result
            return None
        else:
            return pcm

    def flush(self) -> Optional[bytes]:
        """Flush any remaining buffered audio. Call after all data has been processed."""
        if not self.buf:
            return None
        result = bytes(self.buf)
        self.buf.clear()
        self.started = True
        return result


class SyncFuture:
    """Wrapper to make synchronous results look like futures."""
    
    def __init__(self, result):
        self._result = result
    
    def result(self):
        return self._result

