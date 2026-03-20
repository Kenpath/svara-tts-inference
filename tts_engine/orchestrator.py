
from __future__ import annotations
import os
from typing import Iterator, AsyncIterator, List, Optional, Literal, Union
import concurrent.futures
import asyncio
import logging
import torch
from .transports import VLLMEmbeddedTransport
from .mapper import SvaraMapper, extract_custom_token_numbers
from .codec import SNACCodec, get_or_load_tokenizer
from .encoder import svara_text_to_tokens
from .utils import create_speaker_id
from .buffers import AudioBuffer, SyncFuture, crossfade_pcm
from .utils import chunk_text
from .timing import track_time

logger = logging.getLogger(__name__)


def _detect_gpu_tier() -> tuple[int, int]:
    """
    Detect GPU capability and return recommended (max_workers, decode_batch_size).

    High-end GPU (16GB+ VRAM, compute 8.0+): 4 workers, batch of 4
    Standard GPU: 2 workers, batch of 4
    CPU/MPS: 2 workers, batch of 2
    """
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            compute_cap = props.major
            vram_gb = props.total_mem / (1024 ** 3)
            logger.info(f"GPU detected: {props.name}, {vram_gb:.1f}GB VRAM, compute {props.major}.{props.minor}")
            if vram_gb >= 16 and compute_cap >= 8:
                return 4, 4  # High-end (A100, H100, RTX 4090, etc.)
            return 2, 4      # Standard GPU
        except Exception:
            return 2, 4
    return 2, 2               # CPU or MPS

class SvaraTTSOrchestrator:
    """
    Sync/Async TTS orchestrator:
    transport -> mapper -> decoder -> PCM int16 chunks.

    Args:
        transport: The VLLMEmbeddedTransport instance.
        model: The model name (for tokenizer lookup).
        speaker_id: The speaker identifier (e.g., "Hindi (Male)", "English (Female)").
                    If not provided, will be constructed from lang_code and gender.
        lang_code: An ISO 639-1 language code (used if speaker_id not provided).
        gender: The gender of the voice (used if speaker_id not provided).
        prebuffer_seconds: The number of seconds to prebuffer before yielding audio.
        concurrent_decode: If True, decode concurrently.
        max_workers: The number of workers to use for decoding.
        device: Device for SNAC decoder (cuda, mps, cpu, or None for auto).
    """
    def __init__(self,
                 transport: VLLMEmbeddedTransport,
                 model: str = "kenpath/svara-tts-v1",
                 speaker_id: Optional[str] = None,
                 lang_code: str = "en",
                 gender: Literal["male", "female"] = "male",
                 prebuffer_seconds: float = 0.5,
                 concurrent_decode: bool = True,
                 max_workers: Optional[int] = None,
                 decode_batch_size: Optional[int] = None,
                 device: Optional[str] = None):
        # If speaker_id is provided, use it; otherwise construct from lang_code and gender
        if speaker_id is None:
            self.speaker_id = create_speaker_id(lang_code, gender)
        else:
            self.speaker_id = speaker_id

        self.model_name = model
        self.tokenizer_model = os.getenv("TOKENIZER_MODEL", os.getenv("VLLM_MODEL", "kenpath/svara-tts-v1"))
        self.tokenizer      = get_or_load_tokenizer(self.tokenizer_model)

        self.transport      = transport
        self.codec      = SNACCodec(device)
        self.prebuffer_samples = int(self.codec.sample_rate * prebuffer_seconds)
        self.concurrent_decode = concurrent_decode

        # Auto-detect optimal workers and batch size from GPU capability
        auto_workers, auto_batch = _detect_gpu_tier()
        self.max_workers = max_workers if max_workers is not None else auto_workers
        self.decode_batch_size = decode_batch_size if decode_batch_size is not None else auto_batch
        self.max_chunk_chars = 1000    # Split text longer than this
        self.crossfade_ms = 50         # Crossfade overlap between chunks
        logger.info(f"Orchestrator: max_workers={self.max_workers}, decode_batch_size={self.decode_batch_size}")
        
    # ------------ SYNC path ------------
    def stream(self,
               text: str,
               audio_reference: Optional[List[int]] = None,
               reference_text: Optional[str] = None,
               speaker_id: Optional[str] = None,
               **gen_kwargs) -> Iterator[bytes]:
        """Stream the TTS output, automatically chunking long texts.

        For texts longer than max_chunk_chars, splits at sentence boundaries
        and crossfades between chunks for smooth audio stitching.
        """
        chunks = chunk_text(text, max_len=self.max_chunk_chars)

        if len(chunks) <= 1:
            # Short text — no chunking needed, stream directly
            yield from self._stream_one(text, audio_reference=audio_reference, reference_text=reference_text, speaker_id=speaker_id, **gen_kwargs)
            return

        # Long text — synthesize each chunk, crossfade between them
        logger.info(f"Long text ({len(text)} chars) split into {len(chunks)} chunks")
        prev_audio: Optional[bytes] = None

        for i, chunk_text_str in enumerate(chunks):
            # Collect full audio for this chunk (non-streaming per chunk)
            chunk_pcm = b"".join(
                self._stream_one(chunk_text_str, audio_reference=audio_reference, reference_text=reference_text, speaker_id=speaker_id, **gen_kwargs)
            )
            if not chunk_pcm:
                continue

            if prev_audio is None:
                prev_audio = chunk_pcm
            else:
                # Crossfade previous chunk with current
                stitched = crossfade_pcm(prev_audio, chunk_pcm, overlap_ms=self.crossfade_ms, sample_rate=self.codec.sample_rate)
                prev_audio = stitched

        if prev_audio:
            yield prev_audio

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
        
        mapper = SvaraMapper()
        audio_buf = AudioBuffer(self.prebuffer_samples)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) if self.concurrent_decode else None
        pending: List[concurrent.futures.Future] = []
        window_batch: List[List[int]] = []

        def decode_batch(batch: List[List[int]]) -> List[bytes]:
            return self.codec.decode_windows_batch(batch)

        def submit_batch(batch: List[List[int]]):
            b = list(batch)
            return executor.submit(decode_batch, b) if executor else SyncFuture(decode_batch(b))

        def flush_batch():
            if window_batch:
                pending.append(submit_batch(window_batch))
                window_batch.clear()

        try:
            for token_text in self.transport.stream(prompt, **gen_kwargs):
                for n in extract_custom_token_numbers(token_text):
                    win = mapper.feed_raw(n)
                    if win is not None:
                        window_batch.append(win)
                        if len(window_batch) >= self.decode_batch_size:
                            pending.append(submit_batch(window_batch))
                            window_batch.clear()

                    # Yield when we have enough pending
                    while len(pending) > 2:
                        for pcm in pending.pop(0).result():
                            result = audio_buf.process(pcm)
                            if result:
                                yield result

            # Flush remaining window batch
            flush_batch()

            # Drain pending futures
            for fut in pending:
                for pcm in fut.result():
                    result = audio_buf.process(pcm)
                    if result:
                        yield result

            # Flush any prebuffered audio that never hit the threshold
            tail = audio_buf.flush()
            if tail:
                yield tail
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
        """Async stream the TTS output, automatically chunking long texts.

        For texts longer than max_chunk_chars, splits at sentence boundaries
        and crossfades between chunks for smooth audio stitching.
        """
        chunks = chunk_text(text, max_len=self.max_chunk_chars)

        if len(chunks) <= 1:
            # Short text — no chunking needed, stream directly
            async for b in self._astream_one(text, audio_reference=audio_reference, reference_text=reference_text, speaker_id=speaker_id, **gen_kwargs):
                yield b
            return

        # Long text — synthesize each chunk, crossfade between them
        logger.info(f"Long text ({len(text)} chars) split into {len(chunks)} chunks")
        prev_audio: Optional[bytes] = None

        for chunk_text_str in chunks:
            # Collect full audio for this chunk
            chunk_parts = []
            async for b in self._astream_one(chunk_text_str, audio_reference=audio_reference, reference_text=reference_text, speaker_id=speaker_id, **gen_kwargs):
                chunk_parts.append(b)
            chunk_pcm = b"".join(chunk_parts)
            if not chunk_pcm:
                continue

            if prev_audio is None:
                prev_audio = chunk_pcm
            else:
                prev_audio = crossfade_pcm(prev_audio, chunk_pcm, overlap_ms=self.crossfade_ms, sample_rate=self.codec.sample_rate)

        if prev_audio:
            yield prev_audio

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
        
        mapper = SvaraMapper()
        audio_buf = AudioBuffer(self.prebuffer_samples)
        loop = asyncio.get_running_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) if self.concurrent_decode else None
        pending: List[asyncio.Task] = []
        window_batch: List[List[int]] = []

        def decode_batch(batch: List[List[int]]) -> List[bytes]:
            return self.codec.decode_windows_batch(batch)

        async def submit_batch_async(batch: List[List[int]]) -> List[bytes]:
            b = list(batch)
            if executor:
                return await loop.run_in_executor(executor, decode_batch, b)
            else:
                return decode_batch(b)

        def flush_batch():
            if window_batch:
                pending.append(asyncio.create_task(submit_batch_async(list(window_batch))))
                window_batch.clear()

        try:
            async for token_text in self.transport.astream(prompt, **gen_kwargs):
                for n in extract_custom_token_numbers(token_text):
                    win = mapper.feed_raw(n)
                    if win is not None:
                        window_batch.append(win)
                        if len(window_batch) >= self.decode_batch_size:
                            pending.append(asyncio.create_task(submit_batch_async(list(window_batch))))
                            window_batch.clear()

                    # Yield when we have enough pending
                    while len(pending) > 2:
                        for pcm in await pending.pop(0):
                            result = audio_buf.process(pcm)
                            if result:
                                yield result

            # Flush remaining window batch
            flush_batch()

            # Drain pending futures
            for task in pending:
                for pcm in await task:
                    result = audio_buf.process(pcm)
                    if result:
                        yield result

            # Flush any prebuffered audio that never hit the threshold
            tail = audio_buf.flush()
            if tail:
                yield tail
        finally:
            if executor:
                executor.shutdown(wait=True)
