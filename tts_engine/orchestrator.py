
from __future__ import annotations
import os
from typing import Iterator, AsyncIterator, List, Optional, Literal
import concurrent.futures
import asyncio
import logging
import queue
import threading
import time
from .transports import VLLMAsyncEngineTransport
from .mapper import SvaraMapper, extract_custom_token_numbers
from .codec import SNACCodec, get_or_load_tokenizer
from .encoder import svara_text_to_tokens, svara_text_to_tokens_v05
from .utils import create_speaker_id
from .buffers import AudioBuffer
from .timing import track_time

# Route orchestrator logs to Uvicorn logger so INFO lines are visible in container logs.
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

class SvaraTTSOrchestrator:
    """
    TTS orchestrator:
    async vLLM engine -> mapper -> decoder -> PCM int16 chunks.
    
    Args:
        model: The model name.
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
                 model: str = "kenpath/svara-tts-v1",
                 speaker_id: Optional[str] = None,
                 lang_code: str = "en",
                 gender: Literal["male", "female"] = "male",
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
        
        self.transport_async = VLLMAsyncEngineTransport(model=self.model_name)
        self.codec      = SNACCodec(device)
        self.prebuffer_samples = int(self.codec.sample_rate * prebuffer_seconds)
        self.concurrent_decode = concurrent_decode
        self.max_workers    = max_workers

    def _use_v05_prompting(self) -> bool:
        m = (self.model_name or "").lower()
        return "v0.5" in m or "voice-svara-tts-v1-fft-v0.5" in m
        
    # ------------ SYNC path ------------
    def stream(self, 
               text: str, 
               audio_reference: Optional[List[int]] = None,
               reference_text: Optional[str] = None,
               speaker_id: Optional[str] = None,
               **gen_kwargs) -> Iterator[bytes]:
        """
        Sync wrapper around the async stream path.
        """
        out_q: queue.Queue[Optional[bytes]] = queue.Queue()
        err_q: queue.Queue[BaseException] = queue.Queue()

        async def _produce() -> None:
            async for chunk in self.astream(
                text=text,
                audio_reference=audio_reference,
                reference_text=reference_text,
                speaker_id=speaker_id,
                **gen_kwargs,
            ):
                out_q.put(chunk)

        def _runner() -> None:
            try:
                asyncio.run(_produce())
            except BaseException as exc:
                err_q.put(exc)
            finally:
                out_q.put(None)

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()

        while True:
            chunk = out_q.get()
            if chunk is None:
                break
            yield chunk

        thread.join()
        if not err_q.empty():
            raise err_q.get()

    @track_time("Orchestrator.stream_one")
    def _stream_one(self, 
                    text: str, 
                    audio_reference: Optional[List[int]] = None,
                    reference_text: Optional[str] = None,
                    speaker_id: Optional[str] = None,
                    **gen_kwargs) -> Iterator[bytes]:
        # Kept for backward compatibility; callers should use `astream`.
        yield from self.stream(
            text=text,
            audio_reference=audio_reference,
            reference_text=reference_text,
            speaker_id=speaker_id,
            **gen_kwargs,
        )

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
        async for b in self._astream_one(text, audio_reference=audio_reference, reference_text=reference_text, speaker_id=speaker_id, **gen_kwargs):
            yield b

    @track_time("Orchestrator.astream_one")
    async def _astream_one(self, 
                           text: str, 
                           audio_reference: Optional[List[int]] = None,
                           reference_text: Optional[str] = None,
                           speaker_id: Optional[str] = None,
                           **gen_kwargs) -> AsyncIterator[bytes]:
        req_id = str(gen_kwargs.get("request_id", "unknown"))
        voice_name = gen_kwargs.pop("voice_name", None)
        req_start = time.perf_counter()
        ttft_logged = False

        if self._use_v05_prompting():
            selected_voice = voice_name or os.getenv("SVARA_V05_DEFAULT_VOICE", "Prakash")
            prompt = svara_text_to_tokens_v05(
                text=text,
                voice_name=selected_voice,
                tokenizer=self.tokenizer,
                return_decoded=True,
            )
            # v0.5 raw/modal path uses this stop token by default.
            gen_kwargs.setdefault("stop_token_id", 49158)
        else:
            # Generate prompt using current v1 path.
            prompt = svara_text_to_tokens(
                text=text,
                speaker_id=speaker_id or self.speaker_id,
                audio_tokens=audio_reference,
                transcript=reference_text,
                tokenizer=self.tokenizer,
                return_decoded=True
            )
        
        audio_buf = AudioBuffer(self.prebuffer_samples)
        mapper = SvaraMapper()
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

        def _log_ttft_once() -> None:
            nonlocal ttft_logged
            if ttft_logged:
                return
            ttft_logged = True
            ttft_ms = (time.perf_counter() - req_start) * 1000.0
            logger.info("TTFT(model_level_orchestrator) req_id=%s %.2f ms", req_id, ttft_ms)

        try:
            async for token_text in self.transport_async.astream(prompt, **gen_kwargs):
                for n in extract_custom_token_numbers(token_text):
                    win = mapper.feed_raw(n)
                    if win is not None:
                        pending.append(asyncio.create_task(submit_async(win)))
                        
                    # Yield when we have enough pending
                    while len(pending) > 2:
                        result = audio_buf.process(await pending.pop(0))
                        if result:
                            _log_ttft_once()
                            yield result
            
            # Flush remaining
            for task in pending:
                result = audio_buf.process(await task)
                if result:
                    _log_ttft_once()
                    yield result
            final = audio_buf.flush()
            if final:
                _log_ttft_once()
                yield final
            if not ttft_logged:
                logger.info("TTFT(model_level_orchestrator) req_id=%s NA (no audio chunk emitted)", req_id)
        finally:
            if executor:
                executor.shutdown(wait=True)
