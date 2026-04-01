from __future__ import annotations
import asyncio
import uuid
import threading
import logging
import os
from pathlib import Path
from typing import Iterator, AsyncIterator, Optional, Any

from .constants import END_OF_SPEECH, TOKENISER_LENGTH, AUDIO_TOKENS_START

logger = logging.getLogger(__name__)


class VLLMEmbeddedTransport:
    """
    In-process vLLM transport using AsyncLLMEngine.

    The engine is a singleton -- GPU resources must not be double-allocated.
    Call `initialize_engine()` once during app startup (e.g. in FastAPI lifespan).
    """

    _engine = None  # class-level singleton

    @classmethod
    def initialize_engine(
        cls,
        model: str,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = True,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        enforce_eager: bool = False,
        attention_backend: Optional[str] = None,
        kv_cache_dtype: str = "auto",
    ):
        """Initialize the shared AsyncLLMEngine. Must be called once at startup."""
        if cls._engine is not None:
            logger.warning("Engine already initialized, skipping re-initialization")
            return

        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        engine_args = AsyncEngineArgs(
            model=model,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            quantization=quantization,
            enforce_eager=enforce_eager,
            attention_backend=attention_backend,
            kv_cache_dtype=kv_cache_dtype,
        )

        cls._engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info(f"vLLM engine initialized: model={model}, dtype={dtype}, "
                     f"quantization={quantization}, max_model_len={max_model_len}")

    def __init__(self, model: str):
        self.model = model

    @property
    def engine(self):
        if self._engine is None:
            raise RuntimeError(
                "VLLMEmbeddedTransport.initialize_engine() must be called before use"
            )
        return self._engine

    async def astream(self, prompt: str, **gen_kwargs) -> AsyncIterator[str]:
        """Stream text deltas from the embedded vLLM engine."""
        from vllm.sampling_params import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=gen_kwargs.get("max_tokens", 2048),
            temperature=gen_kwargs.get("temperature", 0.75),
            top_p=gen_kwargs.get("top_p", 0.9),
            top_k=gen_kwargs.get("top_k", 40),
            repetition_penalty=gen_kwargs.get("repetition_penalty", 1.1),
            stop_token_ids=[END_OF_SPEECH],
        )

        request_id = str(uuid.uuid4())
        prev_text = ""

        try:
            async for request_output in self.engine.generate(
                prompt, sampling_params, request_id
            ):
                current_text = request_output.outputs[0].text
                delta = current_text[len(prev_text):]
                prev_text = current_text
                if delta:
                    yield delta
        except asyncio.CancelledError:
            await self.engine.abort(request_id)
            raise
        except Exception:
            await self.engine.abort(request_id)
            raise

    def stream(self, prompt: str, **gen_kwargs) -> Iterator[str]:
        """Sync wrapper around astream() for the sync orchestrator path."""

        async def _collect():
            results = []
            async for delta in self.astream(prompt, **gen_kwargs):
                results.append(delta)
            return results

        # Run in a new event loop on a separate thread to avoid
        # blocking or conflicting with the main async loop
        results = []
        exception = None

        def _run():
            nonlocal results, exception
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    results = loop.run_until_complete(_collect())
                finally:
                    loop.close()
            except Exception as e:
                exception = e

        thread = threading.Thread(target=_run)
        thread.start()
        thread.join()

        if exception is not None:
            raise exception

        yield from results


class OpenVINOTransport:
    """
    In-process OpenVINO transport using openvino-genai LLMPipeline.

    This transport emits `<custom_token_N>` strings so the existing Svara mapper
    pipeline can decode audio without changing orchestrator internals.
    """

    _pipeline = None
    _tokenizer = None
    _resolved_model_path = None

    @classmethod
    def _resolve_model_path(cls, model: str) -> str:
        """
        Resolve OpenVINO model input to a local directory path.

        Accepts:
        - Local directory path containing OpenVINO IR artifacts
        - Hugging Face repo ID (downloads snapshot to cache)
        """
        p = Path(model)
        if p.exists() and p.is_dir():
            return str(p.resolve())

        # Treat as Hugging Face repo id when local directory is not found.
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise RuntimeError(
                "OPENVINO_MODEL looks like a Hugging Face repo ID, but "
                "huggingface_hub is not installed."
            ) from e

        hf_token = os.getenv("HF_TOKEN") or None
        logger.info("Downloading OpenVINO model snapshot from Hugging Face: %s", model)
        local_dir = snapshot_download(
            repo_id=model,
            token=hf_token,
            local_dir_use_symlinks=False,
        )
        return str(Path(local_dir).resolve())

    @classmethod
    def initialize_engine(
        cls,
        model: str,
        device: str = "CPU",
        trust_remote_code: bool = True,
    ):
        if cls._pipeline is not None and cls._tokenizer is not None:
            logger.warning("OpenVINO pipeline already initialized, skipping re-initialization")
            return

        try:
            import openvino_genai as ov_genai
            from transformers import AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "OpenVINO backend selected, but dependencies are missing. "
                "Install with: pip install -r requirements-openvino.txt"
            ) from e

        model_path = cls._resolve_model_path(model)
        cls._pipeline = ov_genai.LLMPipeline(model_path, device)
        cls._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
        cls._resolved_model_path = model_path
        logger.info(
            "OpenVINO pipeline initialized: model=%s (resolved=%s), device=%s, tokenizer=%s",
            model, model_path, device, model_path
        )

    def __init__(self, model: str):
        self.model = model

    @property
    def pipeline(self):
        if self._pipeline is None:
            raise RuntimeError("OpenVINOTransport.initialize_engine() must be called before use")
        return self._pipeline

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            raise RuntimeError("OpenVINOTransport.initialize_engine() must be called before use")
        return self._tokenizer

    async def astream(self, prompt: str, **gen_kwargs) -> AsyncIterator[str]:
        """
        Generate token ids with OpenVINO and convert audio token ids into
        `<custom_token_N>` text chunks consumed by the mapper.
        """

        max_new_tokens = gen_kwargs.get("max_tokens", 2048)
        temperature = gen_kwargs.get("temperature", 0.75)
        top_p = gen_kwargs.get("top_p", 0.9)
        top_k = gen_kwargs.get("top_k", 40)
        repetition_penalty = gen_kwargs.get("repetition_penalty", 1.1)
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[Any] = asyncio.Queue()
        sentinel = object()
        worker_error: dict[str, BaseException] = {}

        def put_threadsafe(item: Any):
            loop.call_soon_threadsafe(q.put_nowait, item)

        def _generate_streaming():
            try:
                import openvino_genai as ov_genai
                from openvino import Tensor

                class _TokenStreamer(ov_genai.StreamerBase):
                    def __init__(self):
                        super().__init__()

                    def write(self, token_id) -> "ov_genai.StreamingStatus":
                        tokens_list = list(token_id) if isinstance(token_id, (list, tuple)) else [int(token_id)]
                        for t in tokens_list:
                            if t == END_OF_SPEECH:
                                return ov_genai.StreamingStatus.STOP
                            if t >= AUDIO_TOKENS_START:
                                # Convert vocab token id (e.g. 128266) to custom token index (10).
                                custom_idx = t - TOKENISER_LENGTH
                                if custom_idx > 0:
                                    put_threadsafe(f"<custom_token_{custom_idx}>")
                        return ov_genai.StreamingStatus.RUNNING

                    def end(self):
                        # No-op: sentinel is pushed by worker finally block.
                        return None

                input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                input_tensor = Tensor([input_ids])
                streamer = _TokenStreamer()
                generate_kwargs = {
                    "inputs": input_tensor,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": temperature > 0,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "repetition_penalty": repetition_penalty,
                    "streamer": streamer,
                }
                unsupported: list[str] = []

                # Keep the same API knobs as vLLM; if local OpenVINO version does
                # not support one of them, drop it and continue.
                while True:
                    try:
                        self.pipeline.generate(**generate_kwargs)
                        break
                    except TypeError as e:
                        msg = str(e)
                        dropped = False
                        for key in list(generate_kwargs.keys()):
                            if key in ("inputs", "max_new_tokens", "do_sample", "streamer"):
                                continue
                            if key in msg:
                                unsupported.append(key)
                                generate_kwargs.pop(key, None)
                                dropped = True
                                break
                        if not dropped:
                            raise
                if unsupported:
                    logger.warning(
                        "OpenVINO runtime does not support generation args: %s",
                        ", ".join(sorted(set(unsupported))),
                    )
            except BaseException as e:
                worker_error["error"] = e
            finally:
                put_threadsafe(sentinel)

        worker = threading.Thread(target=_generate_streaming, daemon=True)
        worker.start()

        while True:
            item = await q.get()
            if item is sentinel:
                break
            yield item

        await asyncio.to_thread(worker.join)

        if "error" in worker_error:
            raise worker_error["error"]

    def stream(self, prompt: str, **gen_kwargs) -> Iterator[str]:
        """Sync wrapper around astream() for the sync orchestrator path."""

        async def _collect():
            results = []
            async for delta in self.astream(prompt, **gen_kwargs):
                results.append(delta)
            return results

        results = []
        exception = None

        def _run():
            nonlocal results, exception
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    results = loop.run_until_complete(_collect())
                finally:
                    loop.close()
            except Exception as e:
                exception = e

        thread = threading.Thread(target=_run)
        thread.start()
        thread.join()

        if exception is not None:
            raise exception

        yield from results
