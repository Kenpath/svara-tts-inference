#!/usr/bin/env python3
"""
Benchmark Svara TTS API modes on the async endpoint.

Tests: stream/non-stream × wav/pcm/mp3
Reports TTFB, total latency, audio duration, and stream speed for each.
"""
import argparse
import os
import sys
import time
import urllib.request
import json
import wave
from io import BytesIO
from pathlib import Path

SAMPLE_RATE = 24000
BYTES_PER_SAMPLE = 2  # 16-bit PCM mono

DEFAULT_BASE_URL = "http://localhost:8080"
DEFAULT_TEXT_EN = "Hello from Svara benchmark. Testing latency and audio quality across all modes."
DEFAULT_TEXT_HI = (
    "नमस्ते, यह स्वरा बेंचमार्क है। हम सभी मोड में लेटेंसी और ऑडियो गुणवत्ता का परीक्षण कर रहे हैं।"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = str(PROJECT_ROOT / "audio_outputs" / "benchmark")

# All test combinations: (label, endpoint, stream, response_format)
TEST_CASES = [
    ("async_stream_wav",      "/v1/text-to-speech",      True,  "wav"),
    ("async_stream_pcm",      "/v1/text-to-speech",      True,  "pcm"),
    ("async_stream_mp3",      "/v1/text-to-speech",      True,  "mp3"),
    ("async_nonstream_wav",   "/v1/text-to-speech",      False, "wav"),
    ("async_nonstream_pcm",   "/v1/text-to-speech",      False, "pcm"),
]


def check_health(base_url: str) -> bool:
    """Check if server is healthy before running benchmarks."""
    try:
        url = f"{base_url}/health"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
            model = data.get("model", "unknown")
            print(f"✓ Server healthy | model={model}")
            return True
    except Exception as e:
        print(f"✗ Server not reachable: {e}")
        return False


def estimate_audio_seconds(data: bytes, fmt: str) -> float:
    """Estimate audio duration from raw bytes."""
    if fmt == "wav":
        try:
            with wave.open(BytesIO(data), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return frames / rate if rate > 0 else 0.0
        except Exception:
            return 0.0
    elif fmt == "pcm":
        return len(data) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
    else:
        # For mp3/opus/aac we can't easily compute duration without ffprobe
        return -1.0


def run_test(
    base_url: str,
    label: str,
    endpoint: str,
    stream: bool,
    response_format: str,
    text: str,
    language: str,
    gender: str,
    out_dir: str | None,
) -> dict:
    """Run a single test case and return metrics."""
    url = f"{base_url}{endpoint}"
    body = {
        "transcript": text,
        "language": language,
        "gender": gender,
        "stream": stream,
        "response_format": response_format,
    }
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.time()
    ttfb = None
    all_data = bytearray()

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                if ttfb is None:
                    ttfb = time.time() - start
                all_data.extend(chunk)
    except Exception as e:
        return {
            "label": label,
            "status": "FAIL",
            "error": str(e),
        }

    elapsed = time.time() - start
    data = bytes(all_data)
    audio_secs = estimate_audio_seconds(data, response_format)

    # Save output file
    saved_path = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        ext = response_format if response_format != "pcm" else "raw"
        out_path = os.path.join(out_dir, f"{label}.{ext}")
        with open(out_path, "wb") as f:
            f.write(data)
        saved_path = out_path

    result = {
        "label": label,
        "status": "OK",
        "endpoint": endpoint,
        "stream": stream,
        "format": response_format,
        "ttfb_ms": round(ttfb * 1000, 2) if ttfb is not None else None,
        "total_ms": round(elapsed * 1000, 2),
        "bytes": len(data),
        "audio_seconds": round(audio_secs, 2) if audio_secs >= 0 else "N/A",
        "realtime_factor": round(audio_secs / elapsed, 2) if audio_secs > 0 and elapsed > 0 else "N/A",
    }
    if saved_path:
        result["saved"] = saved_path

    return result


def print_results(results: list[dict]) -> None:
    """Print results as a formatted table."""
    # Header
    print()
    print(f"{'Label':<28} {'Status':<6} {'TTFB ms':>9} {'Total ms':>10} {'Audio s':>8} {'RTF':>6} {'Bytes':>10}")
    print("─" * 85)

    for r in results:
        if r["status"] != "OK":
            print(f"{r['label']:<28} {'FAIL':<6} {'—':>9} {'—':>10} {'—':>8} {'—':>6} {r.get('error', '')}")
            continue

        ttfb = f"{r['ttfb_ms']:.0f}" if r["ttfb_ms"] is not None else "—"
        audio = f"{r['audio_seconds']:.2f}" if isinstance(r["audio_seconds"], (int, float)) else "N/A"
        rtf = f"{r['realtime_factor']:.2f}" if isinstance(r["realtime_factor"], (int, float)) else "N/A"

        print(
            f"{r['label']:<28} {'OK':<6} {ttfb:>9} {r['total_ms']:>10.0f} {audio:>8} {rtf:>6} {r['bytes']:>10}"
        )

    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark all Svara TTS API combinations.")
    parser.add_argument("--url", default=DEFAULT_BASE_URL, help="Base URL (default: http://localhost:8080)")
    parser.add_argument("--text", default=None, help="Custom text to synthesize")
    parser.add_argument("--hindi", action="store_true", help="Use Hindi text instead of English")
    parser.add_argument("--language", default="en", help="Language code (default: en)")
    parser.add_argument("--gender", default="male", help="Gender (default: male)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory to save audio files")
    parser.add_argument("--no-save", action="store_true", help="Don't save audio files")
    parser.add_argument("--cases", nargs="+", help="Run only specific test cases by label (e.g. async_stream_wav)")
    args = parser.parse_args()

    if args.hindi:
        args.language = "hi"
        text = args.text or DEFAULT_TEXT_HI
    else:
        text = args.text or DEFAULT_TEXT_EN

    out_dir = None if args.no_save else args.out_dir

    print(f"🎯 Svara TTS API Benchmark")
    print(f"   URL: {args.url}")
    print(f"   Language: {args.language} | Gender: {args.gender}")
    print(f"   Text: {text[:80]}{'...' if len(text) > 80 else ''}")
    print()

    if not check_health(args.url):
        print("\n⚠️  Server is not responding. Is it running?")
        return 1

    # Filter test cases if specific ones requested
    cases = TEST_CASES
    if args.cases:
        cases = [tc for tc in TEST_CASES if tc[0] in args.cases]
        if not cases:
            print(f"No matching test cases found. Available: {[tc[0] for tc in TEST_CASES]}")
            return 1

    results = []
    total = len(cases)
    for i, (label, endpoint, stream, fmt) in enumerate(cases, 1):
        print(f"[{i}/{total}] {label}...", end=" ", flush=True)
        result = run_test(
            base_url=args.url,
            label=label,
            endpoint=endpoint,
            stream=stream,
            response_format=fmt,
            text=text,
            language=args.language,
            gender=args.gender,
            out_dir=out_dir,
        )
        status = result["status"]
        if status == "OK":
            print(f"✓ {result['total_ms']:.0f}ms")
        else:
            print(f"✗ {result.get('error', 'unknown error')}")
        results.append(result)

    print_results(results)

    if out_dir and any(r["status"] == "OK" for r in results):
        print(f"📁 Audio files saved to: {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
