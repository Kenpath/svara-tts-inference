
from __future__ import annotations
import json
from typing import Iterator, AsyncIterator, Optional, Dict, Any, Union, List
import requests
import aiohttp
from .timing import track_time
from .constants import END_OF_SPEECH
from .codec import get_or_load_tokenizer
import os
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

    @track_time("vLLM.stream")
    def stream(self, prompt: Union[str, List[int]], **gen_kwargs) -> Iterator[str]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "stream": True,
            "max_tokens": gen_kwargs.get("max_tokens", 2048),
            "temperature": gen_kwargs.get("temperature", 0.75),
            "top_p": gen_kwargs.get("top_p", 0.9),
            "top_k": gen_kwargs.get("top_k", 40),
            "repetition_penalty": gen_kwargs.get("repetition_penalty", 1.1),
            "stop_token_ids": [END_OF_SPEECH]  # Stop at end of speech generation
        }
        
        # Use prompt_token_ids if we have a list, otherwise use prompt string
        if isinstance(prompt, list):
            # Decode token IDs to string (includes custom tokens like <custom_token_X>)
            # vLLM will tokenize it back to the same IDs
            tokenizer_model = os.getenv("TOKENIZER_MODEL", os.getenv("VLLM_MODEL", "kenpath/svara-tts-v1"))
            tokenizer = get_or_load_tokenizer(tokenizer_model)
            prompt_str = tokenizer.decode(prompt, skip_special_tokens=False)
            payload["prompt"] = prompt_str
            print(f"[DEBUG] Decoded {len(prompt)} token IDs to prompt string ({len(prompt_str)} chars)")
            print(f"[DEBUG] Token ID range: min={min(prompt)}, max={max(prompt)}")
            print(f"[DEBUG] Prompt preview (first 200 chars): {prompt_str[:200]}")
        else:
            payload["prompt"] = prompt
            print(f"[DEBUG] Sending prompt string to vLLM: {len(prompt)} chars")
        
        print(f"[DEBUG] vLLM payload keys: {list(payload.keys())}")

        with requests.post(self.url, headers=self.headers, json=payload, stream=True) as resp:
            if resp.status_code != 200:
                error_text = resp.text
                print(f"[ERROR] vLLM returned {resp.status_code}: {error_text}")
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

    @track_time("vLLM.astream")
    async def astream(self, prompt: Union[str, List[int]], **gen_kwargs) -> AsyncIterator[str]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "stream": True,
            "max_tokens": gen_kwargs.get("max_tokens", 2048),
            "temperature": gen_kwargs.get("temperature", 0.75),
            "top_p": gen_kwargs.get("top_p", 0.9),
            "top_k": gen_kwargs.get("top_k", 40),
            "repetition_penalty": gen_kwargs.get("repetition_penalty", 1.1),
            "stop_token_ids": [END_OF_SPEECH]  # Stop at end of speech generation
        }
        
        # Use prompt_token_ids if we have a list, otherwise use prompt string
        if isinstance(prompt, list):
            # Decode token IDs to string (includes custom tokens like <custom_token_X>)
            # vLLM will tokenize it back to the same IDs
            tokenizer_model = os.getenv("TOKENIZER_MODEL", os.getenv("VLLM_MODEL", "kenpath/svara-tts-v1"))
            tokenizer = get_or_load_tokenizer(tokenizer_model)
            prompt_str = tokenizer.decode(prompt, skip_special_tokens=False)
            payload["prompt"] = prompt_str
            print(f"[DEBUG] Decoded {len(prompt)} token IDs to prompt string ({len(prompt_str)} chars)")
            print(f"[DEBUG] Token ID range: min={min(prompt)}, max={max(prompt)}")
            print(f"[DEBUG] Prompt preview (first 200 chars): {prompt_str[:200]}")
        else:
            payload["prompt"] = prompt
            print(f"[DEBUG] Sending prompt string to vLLM: {len(prompt)} chars")
        
        print(f"[DEBUG] vLLM payload keys: {list(payload.keys())}")

        async with aiohttp.ClientSession() as sess:
            async with sess.post(self.url, headers=self.headers, json=payload) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    print(f"[ERROR] vLLM returned {resp.status}: {error_text}")
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
