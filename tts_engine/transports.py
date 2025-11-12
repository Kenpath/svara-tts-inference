
from __future__ import annotations
import json
from typing import Iterator, AsyncIterator, Optional, Dict, Any
import requests
import aiohttp
class VLLMCompletionsTransport:
    """
    Sync transport for OpenAI-compatible /v1/completions streaming (SSE).
    Yields incremental text from choices[0].text.
    """
    def __init__(self, base_url: str, model: str, headers: Optional[Dict[str, str]] = None):
        self.url = base_url.rstrip('/') + '/completions'
        self.model = model
        self.headers = {"Content-Type": "application/json"}
        if headers:
            self.headers.update(headers)

    def stream(self, prompt: str, **gen_kwargs) -> Iterator[str]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "max_tokens": gen_kwargs.get("max_tokens", 2048),
            "temperature": gen_kwargs.get("temperature", 0.75),
            "top_p": gen_kwargs.get("top_p", 0.9),
            "do_sample": gen_kwargs.get("do_sample", True),
            "repetition_penalty": gen_kwargs.get("repetition_penalty", 1.1),
            "stop_token_ids": [128258,128262]
        }

        with requests.post(self.url, headers=self.headers, json=payload, stream=True) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                text = choices[0].get("text") or ""
                if text:
                    yield text


class VLLMCompletionsTransportAsync:
    """
    Async transport for OpenAI-compatible /v1/completions streaming (SSE).
    Requires aiohttp.
    """
    def __init__(self, base_url: str, model: str, headers: Optional[Dict[str, str]] = None):
        if aiohttp is None:
            raise RuntimeError("aiohttp is not installed. pip install aiohttp")
        self.url = base_url.rstrip('/') + '/completions'
        self.model = model
        self.headers = {"Content-Type": "application/json"}
        if headers:
            self.headers.update(headers)

    async def astream(self, prompt: str, **gen_kwargs) -> AsyncIterator[str]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "max_tokens": gen_kwargs.get("max_tokens", 2048),
            "temperature": gen_kwargs.get("temperature", 0.75),
            "top_p": gen_kwargs.get("top_p", 0.9),
            "repetition_penalty": gen_kwargs.get("repetition_penalty", 1.1),
            "stop_token_ids": [128258,128262]
        }

        async with aiohttp.ClientSession() as sess:
            async with sess.post(self.url, headers=self.headers, json=payload) as resp:
                resp.raise_for_status()
                async for raw in resp.content:
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    text = choices[0].get("text") or ""
                    if text:
                        yield text
