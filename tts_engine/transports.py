from __future__ import annotations
import os
from typing import Iterator, AsyncIterator, Optional
from uuid import uuid4
import queue
import threading

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

from .timing import track_time
from .constants import END_OF_SPEECH


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class VLLMAsyncEngineTransport:
    """
    Async transport backed by in-process vLLM AsyncLLMEngine.
    """

    _engine: Optional[AsyncLLMEngine] = None
    _engine_model: Optional[str] = None

    def __init__(self, model: str):
        self.model = model
        self._ensure_engine()

    def _ensure_engine(self) -> None:
        if self.__class__._engine is not None and self.__class__._engine_model == self.model:
            return

        args = AsyncEngineArgs(
            model=self.model,
            trust_remote_code=_env_bool("VLLM_TRUST_REMOTE_CODE", True),
            tensor_parallel_size=int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")),
            max_model_len=int(os.getenv("VLLM_MAX_MODEL_LEN", "8192")),
            gpu_memory_utilization=float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9")),
            dtype=os.getenv("VLLM_DTYPE", "auto"),
            quantization=os.getenv("VLLM_QUANTIZATION") or None,
            enforce_eager=_env_bool("VLLM_ENFORCE_EAGER", True),
        )
        self.__class__._engine = AsyncLLMEngine.from_engine_args(args)
        self.__class__._engine_model = self.model

    @property
    def engine(self) -> AsyncLLMEngine:
        if self.__class__._engine is None:
            raise RuntimeError("vLLM async engine is not initialized")
        return self.__class__._engine

    @track_time("vLLM.local_async_engine")
    async def astream(self, prompt: str, **gen_kwargs) -> AsyncIterator[str]:
        params = SamplingParams(
            max_tokens=gen_kwargs.get("max_tokens", 2048),
            temperature=gen_kwargs.get("temperature", 0.75),
            top_p=gen_kwargs.get("top_p", 0.9),
            top_k=gen_kwargs.get("top_k", 40),
            repetition_penalty=gen_kwargs.get("repetition_penalty", 1.1),
            stop_token_ids=[gen_kwargs.get("stop_token_id", END_OF_SPEECH)],
        )

        request_id = f"req-{uuid4().hex}"
        stream = self.engine.generate(
            prompt=prompt,
            sampling_params=params,
            request_id=request_id,
        )

        prev_text = ""
        async for out in stream:
            choices = out.outputs or []
            if not choices:
                continue
            text = choices[0].text or ""
            if not text:
                continue

            # vLLM returns accumulated text for the request; emit only the delta.
            if text.startswith(prev_text):
                delta = text[len(prev_text) :]
            else:
                delta = text
            prev_text = text

            if delta:
                yield delta

    def stream(self, prompt: str, **gen_kwargs) -> Iterator[str]:
        out_q: queue.Queue[Optional[str]] = queue.Queue()
        err_q: queue.Queue[BaseException] = queue.Queue()

        async def _produce() -> None:
            async for token in self.astream(prompt, **gen_kwargs):
                out_q.put(token)

        def _runner() -> None:
            try:
                import asyncio

                asyncio.run(_produce())
            except BaseException as exc:
                err_q.put(exc)
            finally:
                out_q.put(None)

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()

        while True:
            token = out_q.get()
            if token is None:
                break
            yield token

        thread.join()
        if not err_q.empty():
            raise err_q.get()
