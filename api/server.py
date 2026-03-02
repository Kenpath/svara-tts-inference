"""
FastAPI server for Svara TTS API.

Uses in-process async vLLM engine + SNAC decoder for streaming audio generation.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.models import TTSRequest, VoiceResponse, VoicesResponse
from tts_engine.orchestrator import SvaraTTSOrchestrator
from tts_engine.timing import get_timing_stats, reset_timing_stats
from tts_engine.utils import create_speaker_id
from tts_engine.voice_config import (
    get_all_voices,
    get_voice_by_name,
    get_voices_for_runtime_model,
    resolve_voice_model_id,
)

# Use Uvicorn's logger so INFO logs are visible with `--log-level info`.
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)


VLLM_MODEL = os.getenv("VLLM_MODEL", "kenpath/svara-tts-v1")
TOKENIZER_MODEL = os.getenv("TOKENIZER_MODEL", VLLM_MODEL)
TTS_DEVICE = os.getenv("TTS_DEVICE", None)

orchestrator: Optional[SvaraTTSOrchestrator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator

    print("🚀 Initializing Svara TTS API...")
    print(f"   Model: {VLLM_MODEL}")
    print(f"   Tokenizer Model: {TOKENIZER_MODEL}")
    print(f"   Device: {TTS_DEVICE or 'auto-detect'}")
    print(f"   HF_TOKEN: {'set' if os.getenv('HF_TOKEN') else 'not set'}")

    orchestrator = SvaraTTSOrchestrator(
        model=VLLM_MODEL,
        speaker_id="English (Male)",
        device=TTS_DEVICE,
        prebuffer_seconds=0.5,
        concurrent_decode=True,
        max_workers=2,
    )

    print("✓ Orchestrator initialized")
    print(f"✓ Loaded {len(get_voices_for_runtime_model(VLLM_MODEL))} voices")

    yield

    print("🛑 Shutting down Svara TTS API...")


app = FastAPI(
    title="Svara TTS API",
    description="Text-to-speech API for Indian languages with streaming support",
    version="1.0.0",
    lifespan=lifespan,
)


async def audio_stream_converter(
    pcm_stream: AsyncGenerator[bytes, None],
    format: str,
    sample_rate: int = 24000,
    channels: int = 1,
) -> AsyncGenerator[bytes, None]:
    if format == "pcm":
        async for chunk in pcm_stream:
            yield chunk
        return

    cmd = [
        "ffmpeg",
        "-f", "s16le",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-i", "pipe:0",
        "-loglevel", "error",
    ]

    if format == "mp3":
        cmd.extend(["-f", "mp3", "pipe:1"])
    elif format == "opus":
        cmd.extend(["-f", "opus", "pipe:1"])
    elif format == "aac":
        cmd.extend(["-f", "adts", "pipe:1"])
    elif format == "wav":
        cmd.extend(["-f", "wav", "pipe:1"])
    else:
        logger.warning("Unknown format '%s', falling back to PCM", format)
        async for chunk in pcm_stream:
            yield chunk
        return

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async def write_stdin() -> None:
        try:
            async for chunk in pcm_stream:
                if process.stdin:
                    process.stdin.write(chunk)
                    await process.stdin.drain()
            if process.stdin:
                process.stdin.close()
        except Exception:
            if process.returncode is None:
                process.kill()

    write_task = asyncio.create_task(write_stdin())

    try:
        if process.stdout:
            while True:
                chunk = await process.stdout.read(4096)
                if not chunk:
                    break
                yield chunk
    finally:
        if not write_task.done():
            write_task.cancel()
            try:
                await write_task
            except asyncio.CancelledError:
                pass

        if process.returncode is None:
            process.kill()
            await process.wait()


def _build_gen_kwargs(req: TTSRequest) -> dict:
    out = {}
    if req.temperature is not None:
        out["temperature"] = req.temperature
    if req.top_p is not None:
        out["top_p"] = req.top_p
    if req.top_k is not None:
        out["top_k"] = req.top_k
    if req.repetition_penalty is not None:
        out["repetition_penalty"] = req.repetition_penalty
    if req.max_tokens is not None:
        out["max_tokens"] = req.max_tokens
    return out


def _media_type(response_format: str) -> str:
    return {
        "mp3": "audio/mpeg",
        "opus": "audio/ogg",
        "aac": "audio/aac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }.get(response_format, "audio/pcm")


async def _instrument_stream(
    audio_stream: AsyncGenerator[bytes, None],
    req_id: str,
    req_start: float,
    mode: str,
) -> AsyncGenerator[bytes, None]:
    first_chunk_sent = False
    chunk_count = 0
    total_bytes = 0
    try:
        async for chunk in audio_stream:
            if not chunk:
                continue
            chunk_count += 1
            total_bytes += len(chunk)
            if not first_chunk_sent:
                first_chunk_sent = True
                ttft_ms = (time.perf_counter() - req_start) * 1000.0
                logger.info(
                    "TTFT(client_level_api_send) req_id=%s mode=%s %.2f ms",
                    req_id,
                    mode,
                    ttft_ms,
                )
            yield chunk
    finally:
        total_ms = (time.perf_counter() - req_start) * 1000.0
        logger.info(
            "Request done req_id=%s mode=%s chunks=%d bytes=%d total=%.2f ms",
            req_id,
            mode,
            chunk_count,
            total_bytes,
            total_ms,
        )


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": VLLM_MODEL,
        "engine": "async-llm-engine",
    }


@app.get("/v1/voices", response_model=VoicesResponse)
async def get_voices():
    voices = get_voices_for_runtime_model(VLLM_MODEL)
    return VoicesResponse(voices=[VoiceResponse(**voice.to_dict()) for voice in voices])


@app.post("/v1/text-to-speech")
async def text_to_speech_async(req: TTSRequest):
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    req_id = f"req-{uuid4().hex[:8]}"
    req_start = time.perf_counter()
    logger.info(
        "Request start req_id=%s stream=%s format=%s lang=%s gender=%s",
        req_id,
        req.stream,
        req.response_format,
        req.language,
        req.gender,
    )

    speaker_id = create_speaker_id(req.language, req.gender)
    gen_kwargs = _build_gen_kwargs(req)
    gen_kwargs["request_id"] = req_id

    runtime_voice_model = resolve_voice_model_id(VLLM_MODEL)
    runtime_voices = get_voices_for_runtime_model(VLLM_MODEL)
    if not runtime_voices:
        raise HTTPException(status_code=500, detail="No voices configured for the active model")

    # v0.5 / v2-style models require an explicit voice_name.
    if runtime_voice_model == "svara-tts-v2":
        if not req.voice_name:
            raise HTTPException(
                status_code=400,
                detail="voice_name is required for the active model. Use GET /v1/voices to list valid voices.",
            )
        voice = get_voice_by_name(req.voice_name, model_id=runtime_voice_model)
        if voice is None:
            valid = ", ".join(v.name for v in runtime_voices[:10])
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid voice_name '{req.voice_name}' for active model {runtime_voice_model}. "
                    f"Use GET /v1/voices. Example voices: {valid}"
                ),
            )
        # Keep canonical casing from YAML.
        req.voice_name = voice.name
    else:
        # v1-style models use language+gender voice selection.
        has_match = any(req.language in v.languages and v.gender == req.gender for v in runtime_voices)
        if not has_match:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"No voice found for language='{req.language}' gender='{req.gender}' "
                    f"for active model {runtime_voice_model}. Use GET /v1/voices."
                ),
            )

    pcm_generator = orchestrator.astream(
        text=req.transcript,
        speaker_id=speaker_id,
        voice_name=req.voice_name,
        **gen_kwargs,
    )
    audio_stream = audio_stream_converter(pcm_generator, format=req.response_format)

    media_type = _media_type(req.response_format)

    if req.stream:
        instrumented_stream = _instrument_stream(audio_stream, req_id=req_id, req_start=req_start, mode="stream")
        return StreamingResponse(
            instrumented_stream,
            media_type=media_type,
            headers={
                "Content-Type": media_type,
                "X-Sample-Rate": "24000",
                "X-Channels": "1",
                "X-Execution-Mode": "async",
            },
        )

    chunks = []
    first_chunk_seen = False
    async for chunk in audio_stream:
        if chunk and not first_chunk_seen:
            first_chunk_seen = True
            ttft_ms = (time.perf_counter() - req_start) * 1000.0
            logger.info(
                "TTFT(client_level_api_send) req_id=%s mode=%s %.2f ms",
                req_id,
                "non-stream",
                ttft_ms,
            )
        chunks.append(chunk)
    complete_audio = b"".join(chunks)
    total_ms = (time.perf_counter() - req_start) * 1000.0
    logger.info(
        "Request done req_id=%s mode=%s chunks=%d bytes=%d total=%.2f ms",
        req_id,
        "non-stream",
        len(chunks),
        len(complete_audio),
        total_ms,
    )

    return Response(
        content=complete_audio,
        media_type=media_type,
        headers={
            "Content-Type": media_type,
            "X-Sample-Rate": "24000",
            "X-Channels": "1",
            "X-Execution-Mode": "async",
            "Content-Length": str(len(complete_audio)),
        },
    )


@app.get("/debug/timing")
async def get_timing():
    stats = get_timing_stats()
    formatted_stats = {}
    for func_name, data in stats.items():
        if data["count"] == 0:
            continue
        avg_time = data["total_time"] / data["count"]
        min_time = data["min_time"] if data["min_time"] != float("inf") else 0
        max_time = data["max_time"] if data["max_time"] != float("-inf") else 0
        formatted_stats[func_name] = {
            "calls": data["count"],
            "total_ms": round(data["total_time"] * 1000, 2),
            "avg_ms": round(avg_time * 1000, 2),
            "min_ms": round(min_time * 1000, 2),
            "max_ms": round(max_time * 1000, 2),
        }
    return {"timing_stats": formatted_stats, "note": "All times in milliseconds"}


@app.post("/debug/timing/reset")
async def reset_timing():
    reset_timing_stats()
    return {"status": "success", "message": "Timing statistics have been reset"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("API_PORT", "8080"))
    host = os.getenv("API_HOST", "0.0.0.0")

    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
