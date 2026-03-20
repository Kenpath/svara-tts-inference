
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

logger = logging.getLogger(__name__)


def _detect_gpu_tier() -> int:
    """
    Detect GPU capability and return recommended max_workers.

    High-end GPU (16GB+ VRAM, compute 8.0+): 4 workers
    Standard GPU: 2 workers
    CPU/MPS: 2 workers
    """
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            compute_cap = props.major
            vram_gb = props.total_mem / (1024 ** 3)
            logger.info(f"GPU detected: {props.name}, {vram_gb:.1f}GB VRAM, compute {props.major}.{props.minor}")
            if vram_gb >= 16 and compute_cap >= 8:
                return 4  # High-end (A100, H100, RTX 4090, etc.)
        except Exception:
            pass
    return 2


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
        max_workers: The number of workers to use for decoding (None = auto-detect).
        snac_window_size: SNAC mapper window size in codes. Must be multiple of 7.
                          28 = 4 frames (default, fast TTFB), 56 = 8 frames (fewer
                          decode calls, better throughput). Set via SNAC_WINDOW_SIZE env var.
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
                 snac_window_size: Optional[int] = None,
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

        # Auto-detect optimal workers from GPU capability
        self.max_workers = max_workers if max_workers is not None else _detect_gpu_tier()

        # SNAC window size: configurable via constructor or env var
        if snac_window_size is not None:
            self.snac_window_size = snac_window_size
        else:
            self.snac_window_size = int(os.getenv("SNAC_WINDOW_SIZE", "28"))

        # Long-text chunking config
        # Each SNAC frame = 7 tokens ≈ 10ms audio. With max_tokens=2048,
        # the model can produce ~292 frames ≈ 3s of speech. At ~15 chars/s
        # speaking rate, that's ~45 chars. Use 200 chars for safety margin
        # and to allow for varying speech rates across languages.
        self.max_chunk_chars = 200
        self.crossfade_ms = 50         # Crossfade overlap between chunks

        logger.info(f"Orchestrator: max_workers={self.max_workers}, "
                     f"snac_window_size={self.snac_window_size}, "
                     f"prebuffer={prebuffer_seconds}s")

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
        Streams audio progressively within each chunk — only holds back
        the last overlap_ms for crossfading with the next chunk.
        """
        chunks = chunk_text(text, max_len=self.max_chunk_chars)

        if len(chunks) <= 1:
            yield from self._stream_one(text, audio_reference=audio_reference, reference_text=reference_text, speaker_id=speaker_id, **gen_kwargs)
            return

        logger.info(f"Long text ({len(text)} chars) split into {len(chunks)} chunks")
        overlap_bytes = int(self.codec.sample_rate * self.crossfade_ms / 1000) * 2  # 2 bytes per sample
        prev_tail: Optional[bytes] = None

        for chunk_text_str in chunks:
            is_last_chunk = (chunk_text_str is chunks[-1])
            trailing = bytearray()

            for b in self._stream_one(chunk_text_str, audio_reference=audio_reference, reference_text=reference_text, speaker_id=speaker_id, **gen_kwargs):
                trailing.extend(b)

                if prev_tail is not None:
                    head = bytes(trailing[:overlap_bytes]) if len(trailing) >= overlap_bytes else bytes(trailing)
                    if len(head) >= overlap_bytes:
                        blended = crossfade_pcm(prev_tail, head,
                                                overlap_ms=self.crossfade_ms, sample_rate=self.codec.sample_rate)
                        yield blended
                        trailing = bytearray(trailing[overlap_bytes:])
                        prev_tail = None
                    continue

                if len(trailing) > overlap_bytes:
                    to_yield = bytes(trailing[:-overlap_bytes])
                    trailing = bytearray(trailing[-overlap_bytes:])
                    yield to_yield

            if prev_tail is not None:
                if trailing:
                    blended = crossfade_pcm(prev_tail, bytes(trailing),
                                            overlap_ms=self.crossfade_ms, sample_rate=self.codec.sample_rate)
                    yield blended
                else:
                    yield prev_tail
                prev_tail = None
                trailing = bytearray()

            if not is_last_chunk and len(trailing) > overlap_bytes:
                yield bytes(trailing[:-overlap_bytes])
                prev_tail = bytes(trailing[-overlap_bytes:])
            elif not is_last_chunk:
                prev_tail = bytes(trailing) if trailing else None
            else:
                if trailing:
                    yield bytes(trailing)

    def _stream_one(self,
                    text: str,
                    audio_reference: Optional[List[int]] = None,
                    reference_text: Optional[str] = None,
                    speaker_id: Optional[str] = None,
                    **gen_kwargs) -> Iterator[bytes]:

        prompt = svara_text_to_tokens(
            text=text,
            speaker_id=speaker_id or self.speaker_id,
            audio_tokens=audio_reference,
            transcript=reference_text,
            tokenizer=self.tokenizer,
            return_decoded=True
        )

        logger.info(f"Final prompt before inference: {len(prompt)} chars")
        logger.debug(f"Full prompt: {prompt}")

        mapper = SvaraMapper(window_size=self.snac_window_size)
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
                    win = mapper.feed_raw(n)
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
        Streams audio progressively within each chunk — only holds back
        the last overlap_ms for crossfading with the next chunk.
        """
        chunks = chunk_text(text, max_len=self.max_chunk_chars)

        if len(chunks) <= 1:
            async for b in self._astream_one(text, audio_reference=audio_reference, reference_text=reference_text, speaker_id=speaker_id, **gen_kwargs):
                yield b
            return

        logger.info(f"Long text ({len(text)} chars) split into {len(chunks)} chunks")
        overlap_bytes = int(self.codec.sample_rate * self.crossfade_ms / 1000) * 2  # 2 bytes per sample
        prev_tail: Optional[bytes] = None
        is_first_chunk = True

        for chunk_text_str in chunks:
            is_last_chunk = (chunk_text_str is chunks[-1])
            # Accumulate a trailing buffer per chunk so we can hold back overlap_bytes
            trailing = bytearray()

            async for b in self._astream_one(chunk_text_str, audio_reference=audio_reference, reference_text=reference_text, speaker_id=speaker_id, **gen_kwargs):
                trailing.extend(b)

                # For the first piece of the first non-first chunk, crossfade with prev_tail
                if prev_tail is not None:
                    head = bytes(trailing[:overlap_bytes]) if len(trailing) >= overlap_bytes else bytes(trailing)
                    if len(head) >= overlap_bytes:
                        blended = crossfade_pcm(prev_tail, head,
                                                overlap_ms=self.crossfade_ms, sample_rate=self.codec.sample_rate)
                        yield blended
                        trailing = bytearray(trailing[overlap_bytes:])
                        prev_tail = None
                    # else: keep accumulating until we have enough for crossfade
                    continue

                # Stream the middle: yield everything except the last overlap_bytes
                if len(trailing) > overlap_bytes:
                    to_yield = bytes(trailing[:-overlap_bytes])
                    trailing = bytearray(trailing[-overlap_bytes:])
                    yield to_yield

            # After chunk finishes, handle any remaining crossfade that didn't have enough data
            if prev_tail is not None:
                # Chunk was very short, just crossfade what we have
                if trailing:
                    blended = crossfade_pcm(prev_tail, bytes(trailing),
                                            overlap_ms=self.crossfade_ms, sample_rate=self.codec.sample_rate)
                    yield blended
                else:
                    yield prev_tail
                prev_tail = None
                trailing = bytearray()

            # Hold back the tail for crossfading with the next chunk
            if not is_last_chunk and len(trailing) > overlap_bytes:
                yield bytes(trailing[:-overlap_bytes])
                prev_tail = bytes(trailing[-overlap_bytes:])
            elif not is_last_chunk:
                prev_tail = bytes(trailing) if trailing else None
            else:
                # Last chunk — yield everything
                if trailing:
                    yield bytes(trailing)

            is_first_chunk = False

    async def _astream_one(self,
                           text: str,
                           audio_reference: Optional[List[int]] = None,
                           reference_text: Optional[str] = None,
                           speaker_id: Optional[str] = None,
                           **gen_kwargs) -> AsyncIterator[bytes]:

        prompt = svara_text_to_tokens(
            text=text,
            speaker_id=speaker_id or self.speaker_id,
            audio_tokens=audio_reference,
            transcript=reference_text,
            tokenizer=self.tokenizer,
            return_decoded=True
        )

        logger.info(f"Final prompt before inference: {len(prompt)} chars")
        logger.debug(f"Full prompt: {prompt}")

        mapper = SvaraMapper(window_size=self.snac_window_size)
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
            async for token_text in self.transport.astream(prompt, **gen_kwargs):
                for n in extract_custom_token_numbers(token_text):
                    win = mapper.feed_raw(n)
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

            # Flush any prebuffered audio that never hit the threshold
            tail = audio_buf.flush()
            if tail:
                yield tail
        finally:
            if executor:
                executor.shutdown(wait=True)
