
from __future__ import annotations
from typing import Optional


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


class SyncFuture:
    """Wrapper to make synchronous results look like futures."""
    
    def __init__(self, result):
        self._result = result
    
    def result(self):
        return self._result

