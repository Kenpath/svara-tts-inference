
from __future__ import annotations
import os
from typing import Iterator, AsyncIterator, List, Optional, Literal, Union
import concurrent.futures
import asyncio
import logging
from .transports import VLLMCompletionsTransport, VLLMCompletionsTransportAsync
from .mapper import SvaraMapper, extract_custom_token_numbers
from .codec import SNACCodec, get_or_load_tokenizer
from .encoder import svara_text_to_tokens
from .utils import create_speaker_id
from .buffers import AudioBuffer, SyncFuture
from .timing import track_time

logger = logging.getLogger(__name__)

class SvaraTTSOrchestrator:
    """
    Sync/Async TTS orchestrator:
    transport -> mapper -> decoder -> PCM int16 chunks.
    
    Args:
        base_url: The base URL of the VLLM server.
        model: The model name.
        speaker_id: The speaker identifier (e.g., "Hindi (Male)", "English (Female)").
                    If not provided, will be constructed from lang_code and gender.
        lang_code: An ISO 639-1 language code (used if speaker_id not provided).
        gender: The gender of the voice (used if speaker_id not provided).
        headers: The headers for the VLLM server.
        prebuffer_seconds: The number of seconds to prebuffer before yielding audio.
        concurrent_decode: If True, decode concurrently.
        max_workers: The number of workers to use for decoding.
        device: Device for SNAC decoder (cuda, mps, cpu, or None for auto).
    """
    def __init__(self,
                 base_url: str,
                 model: str = "kenpath/svara-tts-v1",
                 speaker_id: Optional[str] = None,
                 lang_code: str = "en",
                 gender: Literal["male", "female"] = "male",
                 headers: Optional[dict] = None,
                 prebuffer_seconds: float = 0.5,
                 concurrent_decode: bool = True,
                 max_workers: int = 2,
                 device: Optional[str] = None):
        # If speaker_id is provided, use it; otherwise construct from lang_code and gender
        if speaker_id is None:
            self.speaker_id = create_speaker_id(lang_code, gender)
        else:
            self.speaker_id = speaker_id

        self.model_name = model
        self.tokenizer_model = os.getenv("TOKENIZER_MODEL", os.getenv("VLLM_MODEL", "kenpath/svara-tts-v1"))            
        self.tokenizer      = get_or_load_tokenizer(self.tokenizer_model)       
        
        self.transport      = VLLMCompletionsTransport(base_url, model, headers)
        self.transport_async = None  # lazy
        self.mapper     = SvaraMapper()
        self.codec      = SNACCodec(device)
        self.prebuffer_samples = int(self.codec.sample_rate * prebuffer_seconds)
        self.concurrent_decode = concurrent_decode
        self.max_workers    = max_workers
        
    # ------------ SYNC path ------------
    def stream(self, 
               text: str, 
               audio_reference: Optional[List[int]] = None,
               reference_text: Optional[str] = None,
               speaker_id: Optional[str] = None,
               **gen_kwargs) -> Iterator[bytes]:
        """Stream the TTS output.
        
        Args:
            text: The text to synthesize.
            audio_reference: Optional SNAC tokens for zero-shot voice cloning.
            reference_text: Optional transcript for the reference audio.
            speaker_id: Optional speaker ID to override the default.
            gen_kwargs: Additional keyword arguments to pass to the transport.
        """
        yield from self._stream_one(text, audio_reference=audio_reference, reference_text=reference_text, speaker_id=speaker_id, **gen_kwargs)

    @track_time("Orchestrator.stream_one")
    def _stream_one(self, 
                    text: str, 
                    audio_reference: Optional[List[int]] = None,
                    reference_text: Optional[str] = None,
                    speaker_id: Optional[str] = None,
                    **gen_kwargs) -> Iterator[bytes]:
        
        # Generate prompt using svara_text_to_tokens
        prompt = svara_text_to_tokens(
            text=text,
            speaker_id=speaker_id or self.speaker_id,
            audio_tokens=audio_reference,
            transcript=reference_text,
            tokenizer=self.tokenizer,
            return_decoded=True
        )
        
        # Log prompt details
        logger.info(f"Final prompt before inference: {len(prompt)} chars")
        logger.debug(f"Full prompt: {prompt}")
        
        audio_buf = AudioBuffer(self.prebuffer_samples)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) if self.concurrent_decode else None
        pending: List[concurrent.futures.Future] = []

        def decode(win: List[int]) -> bytes:
            return self.codec.decode_window(win)

        def submit(win: List[int]):
            return executor.submit(decode, win) if executor else SyncFuture(decode(win))

        try:
            for token_text in self.transport.stream(prompt, **gen_kwargs):
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
    async def astream(self, 
                      text: str, 
                      audio_reference: Optional[List[int]] = None,
                      reference_text: Optional[str] = None,
                      speaker_id: Optional[str] = None,
                      **gen_kwargs) -> AsyncIterator[bytes]:
        """Async stream the TTS output.
        
        Args:
            text: The text to synthesize.
            audio_reference: Optional SNAC tokens for zero-shot voice cloning.
            reference_text: Optional transcript for the reference audio.
            speaker_id: Optional speaker ID to override the default.
            gen_kwargs: Additional keyword arguments to pass to the transport.
        """
        if self.transport_async is None:
            base_url = self.transport.url[:-12]  # remove '/completions'
            self.transport_async = VLLMCompletionsTransportAsync(
                base_url, self.transport.model, self.transport.headers
            )
        
        async for b in self._astream_one(text, audio_reference=audio_reference, reference_text=reference_text, speaker_id=speaker_id, **gen_kwargs):
            yield b

    @track_time("Orchestrator.astream_one")
    async def _astream_one(self, 
                           text: str, 
                           audio_reference: Optional[List[int]] = None,
                           reference_text: Optional[str] = None,
                           speaker_id: Optional[str] = None,
                           **gen_kwargs) -> AsyncIterator[bytes]:
        # Generate prompt using svara_text_to_tokens
        prompt = svara_text_to_tokens(
            text=text,
            speaker_id=speaker_id or self.speaker_id,
            audio_tokens=audio_reference,
            transcript=reference_text,
            tokenizer=self.tokenizer,
            return_decoded=True
        )
        
        # Log prompt details
        logger.info(f"Final prompt before inference: {len(prompt)} chars")
        logger.debug(f"Full prompt: {prompt}")
        
        audio_buf = AudioBuffer(self.prebuffer_samples)
        loop = asyncio.get_running_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) if self.concurrent_decode else None
        pending: List[asyncio.Task] = []

        def decode(win: List[int]) -> bytes:
            return self.codec.decode_window(win)

        async def submit_async(win: List[int]) -> bytes:
            if executor:
                return await loop.run_in_executor(executor, decode, win)
            else:
                return decode(win)

        try:
            async for token_text in self.transport_async.astream(prompt, **gen_kwargs):
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
