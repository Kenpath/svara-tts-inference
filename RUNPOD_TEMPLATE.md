# Svara-TTS v1 - vLLM RunPod Template

**Docker Image:** `vllm/vllm-openai:latest`

Multilingual TTS model supporting **19 languages** (18 Indic + English): Hindi, Bengali, Marathi, Telugu, Kannada, Bhojpuri, Magahi, Chhattisgarh, Maithili, Assamese, Bodo, Dogri, Gujarati, Malayalam, Punjabi, Tamil, Nepali, Sanskrit, English.

**Features:** Emotion control (`<happy>`, `<sad>`, `<anger>`, `<fear>`, `<clear>`), Speaker IDs (`Language (Gender)`), LoRA-friendly

**By Kenpath Technologies** | [GitHub](https://github.com/Kenpath/svara-tts-inference) | [Model](https://huggingface.co/kenpath/svara-tts-v1)

---

## Setup

**GPU:** Ampere/Hopper/Blackwell, ~6-8GB VRAM minimum

**Docker Command:**
```bash
--model kenpath/svara-tts-v1 --port 8000 --trust-remote-code
```

**Port:** 8000 (vLLM OpenAI-compatible API)

**Endpoint:** `https://<POD-ID>-8000.proxy.runpod.net`

---

## Usage

**Install dependencies:**
```bash
pip install requests snac torch soundfile numpy
```

**Inference script:**
```python
#!/usr/bin/env python3
import json, re, requests, numpy as np, torch, soundfile as sf
from snac import SNAC

RUNPOD_URL = "https://<POD-ID>-8000.proxy.runpod.net/v1/chat/completions"
MODEL_NAME = "kenpath/svara-tts-v1"
SPEAKER_ID = "Hindi (Female)"
PROMPT_TEXT = "नमस्ते! आज मौसम बहुत अच्छा है। <happy>"
OUT_FILE = "output.wav"

CODE_TOKEN_OFFSET = 128266
TOKENS_PER_GROUP = 7
N_LAYERS = 3

token_pat = re.compile(r"<custom_token_(\d+)>")

def request_stream():
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": f"You are {SPEAKER_ID}"},
            {"role": "user", "content": PROMPT_TEXT}
        ],
        "stream": True,
    }
    with requests.post(RUNPOD_URL, json=payload, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or raw.startswith("data: [DONE]"):
                break
            if raw.startswith("data: "):
                chunk = json.loads(raw[6:])
                txt = chunk["choices"][0].get("delta", {}).get("content", "")
                yield from (int(m) for m in token_pat.findall(txt))

def tokens_to_snac_codes(token_ids):
    ids = [tid - CODE_TOKEN_OFFSET for tid in token_ids if tid >= CODE_TOKEN_OFFSET]
    whole = len(ids) // (TOKENS_PER_GROUP * N_LAYERS) * (TOKENS_PER_GROUP * N_LAYERS)
    ids = np.asarray(ids[:whole], dtype=np.int16)
    codes = ids.reshape(-1, N_LAYERS, TOKENS_PER_GROUP).transpose(1, 0, 2).reshape(N_LAYERS, -1)
    return torch.from_numpy(codes)[None]

tokens = list(request_stream())
codes = tokens_to_snac_codes(tokens)

snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
device = "cuda" if torch.cuda.is_available() else "cpu"

with torch.inference_mode():
    audio = snac.decode(codes.to(device)).cpu().squeeze().numpy()

sf.write(OUT_FILE, audio, 24_000, subtype="PCM_16")
print(f"✓ Saved {OUT_FILE} ({audio.shape[-1]/24_000:.2f}s)")
```

---

## Prompting

**Speaker:** `Language (Gender)` — e.g., `Hindi (Female)`, `Bengali (Male)`

**Emotions (end of sentence):** `<happy>`, `<sad>`, `<anger>`, `<fear>`, `<clear>`

**Prosody:** Use `...` for pauses, `,` for breaks, `!` for emphasis

**Example:** `आज सच में अच्छी खबर है! <happy>`

---

## Troubleshooting

- **No tokens:** Check `RUNPOD_URL`, port 8000 exposed, pod running
- **Poor quality:** Try emotion tags, add punctuation, use `<clear>`
- **OOM:** Add `--max-model-len 2048` or `--max-num-seqs 4`

Built by [Kenpath Technologies](https://kenpath.ai) with ❤️