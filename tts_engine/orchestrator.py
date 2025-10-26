
from __future__ import annotations
from typing import Iterator, AsyncIterator, List, Optional, Literal
import concurrent.futures
import asyncio
from .transports import VLLMCompletionsTransport, VLLMCompletionsTransportAsync
from .mapper import SvaraMapper, extract_custom_token_numbers
from .decoder_snac import SNACDecoder
from .utils import svara_prompt, chunk_text
from .buffers import AudioBuffer, SyncFuture

class SvaraTTSOrchestrator:
    """
    Sync/Async TTS orchestrator:
    transport -> mapper -> decoder -> PCM int16 chunks.
    
    Args:
        base_url: The base URL of the VLLM server.
        model: The model name.
        voice: The voice name.
        lang: The language code.
        headers: The headers for the VLLM server.
        hop_only: If True, return only the last hop_samples for streaming.
        hop_samples: The number of samples to keep when hop_only=True.
        prebuffer_seconds: The number of seconds to prebuffer.
        concurrent_decode: If True, decode concurrently.
        max_workers: The number of workers to use for decoding.
    """
    def __init__(self,
                 base_url: str,
                 model: str,
                 lang_code: str,
                 gender: Literal["male", "female"],
                 headers: Optional[dict] = None,
                 hop_only: bool = False,
                 hop_samples: int = 2048,
                 prebuffer_seconds: float = 1.2,
                 concurrent_decode: bool = True,
                 max_workers: int = 2):
        self.lang_code = lang_code
        self.gender = gender
        self.transport    = VLLMCompletionsTransport(base_url, model, headers)
        self.transport_async = None  # lazy
        self.mapper     = SvaraMapper()
        self.decoder    = SNACDecoder()
        self.hop_only      = hop_only
        self.hop_samples    = hop_samples
        self.prebuffer_samples = int(self.decoder.sample_rate * prebuffer_seconds)
        self.concurrent_decode = concurrent_decode
        self.max_workers    = max_workers

    # ------------ SYNC path ------------
    def stream(self, text: str, *, chunk_long_text: bool = False) -> Iterator[bytes]:
        """Stream the TTS output.
        
        Args:
            text: The text to synthesize.
            chunk_long_text: If True, chunk the text into smaller chunks.
        """
        if chunk_long_text:
            # Sequentially synthesize chunks.
            for t in chunk_text(text, max_len=280, overlap=24):
                yield from self._stream_one(t)
        else:
            yield from self._stream_one(text)

    def _stream_one(self, text: str) -> Iterator[bytes]:
        prompt = svara_prompt(text, self.lang_code, self.gender)
        audio_buf = AudioBuffer(self.prebuffer_samples)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) if self.concurrent_decode else None
        pending: List[concurrent.futures.Future] = []

        def decode(win: List[int]) -> bytes:
            return self.decoder.decode_window(win, hop_only=self.hop_only, hop_samples=self.hop_samples)

        def submit(win: List[int]):
            return executor.submit(decode, win) if executor else SyncFuture(decode(win))

        try:
            for token_text in self.transport.stream(prompt):
                for n in extract_custom_token_numbers(token_text):
                    win = self.mapper.feed_raw(n)
                    if win is not None:
                        pending.append(submit(win))
                        
                    # Yield when we have enough pending
                    while len(pending) > 2:
                        result = audio_buf.process(pending.pop(0).result())
                        if result:
                            yield result
            
            # Flush remaining
            for fut in pending:
                result = audio_buf.process(fut.result())
                if result:
                    yield result
        finally:
            if executor:
                executor.shutdown(wait=True)

    # ------------ ASYNC path ------------
    async def astream(self, text: str, *, chunk_long_text: bool = False) -> AsyncIterator[bytes]:
        if self.transport_async is None:
            base_url = self.transport.url[:-12]  # remove '/completions'
            self.transport_async = VLLMCompletionsTransportAsync(
                base_url, self.transport.model, self.transport.headers
            )
        
        if chunk_long_text:
            for t in chunk_text(text, max_len=280, overlap=24):
                async for b in self._astream_one(t):
                    yield b
        else:
            async for b in self._astream_one(text):
                yield b

    async def _astream_one(self, text: str) -> AsyncIterator[bytes]:
        prompt = svara_prompt(text, self.lang_code, self.gender)
        audio_buf = AudioBuffer(self.prebuffer_samples)
        loop = asyncio.get_running_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) if self.concurrent_decode else None
        pending: List[asyncio.Task] = []

        def decode(win: List[int]) -> bytes:
            return self.decoder.decode_window(win, hop_only=self.hop_only, hop_samples=self.hop_samples)

        async def submit_async(win: List[int]) -> bytes:
            if executor:
                return await loop.run_in_executor(executor, decode, win)
            else:
                return decode(win)

        try:
            async for token_text in self.transport_async.astream(prompt):
                for n in extract_custom_token_numbers(token_text):
                    win = self.mapper.feed_raw(n)
                    if win is not None:
                        pending.append(asyncio.create_task(submit_async(win)))
                        
                    # Yield when we have enough pending
                    while len(pending) > 2:
                        result = audio_buf.process(await pending.pop(0))
                        if result:
                            yield result
            
            # Flush remaining
            for task in pending:
                result = audio_buf.process(await task)
                if result:
                    yield result
        finally:
            if executor:
                executor.shutdown(wait=True)
