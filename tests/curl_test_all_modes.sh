#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8080}"
OUT_DIR="${OUT_DIR:-/tmp/svara_tts_curl_tests}"
mkdir -p "$OUT_DIR"

echo "Base URL: $BASE_URL"
echo "Saving audio files to: $OUT_DIR"

# 1) Async + streaming
curl --fail --silent --show-error --no-buffer \
  -X POST "$BASE_URL/v1/text-to-speech" \
  -H "Content-Type: application/json" \
  -d '{"transcript":"Hello from Svara async stream","language":"en","gender":"male","stream":true,"response_format":"wav"}' \
  -o "$OUT_DIR/01_async_stream.wav"

# 2) Async + non-streaming
curl --fail --silent --show-error \
  -X POST "$BASE_URL/v1/text-to-speech" \
  -H "Content-Type: application/json" \
  -d '{"transcript":"Hello from Svara async non-stream","language":"en","gender":"male","stream":false,"response_format":"wav"}' \
  -o "$OUT_DIR/02_async_nonstream.wav"

echo "Saved files:"
ls -lh "$OUT_DIR"/0*_*.wav
