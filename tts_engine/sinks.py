
from __future__ import annotations
from typing import Optional
import wave
import pyaudio
class WavSink:
    """Simple WAV writer sink for PCM int16 mono @ sample_rate."""
    def __init__(self, path: str, sample_rate: int = 24000):
        self.path = path
        self.sample_rate = sample_rate
        self._wf: Optional[wave.Wave_write] = None

    def open(self):
        self._wf = wave.open(self.path, "wb")
        self._wf.setnchannels(1)
        self._wf.setsampwidth(2)  # int16
        self._wf.setframerate(self.sample_rate)

    def write(self, pcm_bytes: bytes):
        if not self._wf:
            self.open()
        self._wf.writeframes(pcm_bytes)

    def close(self):
        if self._wf:
            self._wf.close()
            self._wf = None

class PyAudioSink:
    """Live playback sink (optional dependency)."""
    def __init__(self, sample_rate: int = 24000):
        if pyaudio is None:
            raise RuntimeError("PyAudio is not installed. pip install pyaudio")
        self.sample_rate = sample_rate
        self._p = pyaudio.PyAudio()
        self._stream = self._p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, output=True)

    def write(self, pcm_bytes: bytes):
        self._stream.write(pcm_bytes)

    def close(self):
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._p:
            self._p.terminate()
