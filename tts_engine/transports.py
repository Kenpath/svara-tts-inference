from __future__ import annotations
import os
import asyncio
import uuid
import threading
import logging
from typing import Iterator, AsyncIterator, Optional, Dict, Any

from .timing import track_time
from .constants import END_OF_SPEECH

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

    @track_time("vLLM.astream")
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

    @track_time("vLLM.stream")
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
