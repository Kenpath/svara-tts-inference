#!/usr/bin/env python3
"""
Supervisord event listener for health checking.
Monitors vLLM and FastAPI processes and logs their status.
"""
import sys
import os
import time
import requests
from datetime import datetime


def write_stdout(s):
    """Write to stdout and flush."""
    sys.stdout.write(s)
    sys.stdout.flush()


def write_stderr(s):
    """Write to stderr and flush."""
    sys.stderr.write(s)
    sys.stderr.flush()


def check_vllm_health():
    """Check if vLLM server is responding."""
    try:
        vllm_port = os.getenv("VLLM_PORT", "8000")
        response = requests.get(f"http://localhost:{vllm_port}/health", timeout=5)
        return response.status_code == 200
    except Exception as e:
        write_stderr(f"vLLM health check failed: {e}\n")
        return False


def check_fastapi_health():
    """Check if FastAPI server is responding."""
    try:
        api_port = os.getenv("API_PORT", "8080")
        response = requests.get(f"http://localhost:{api_port}/health", timeout=5)
        return response.status_code == 200
    except Exception as e:
        write_stderr(f"FastAPI health check failed: {e}\n")
        return False


def main():
    """Main event listener loop."""
    write_stdout('READY\n')
    
    while True:
        # Wait for an event
        line = sys.stdin.readline()
        write_stderr(f"Received event: {line}")
        
        # Acknowledge the event
        write_stdout('RESULT 2\nOK')
        write_stdout('READY\n')
        
        # Perform health checks
        timestamp = datetime.now().isoformat()
        vllm_ok = check_vllm_health()
        fastapi_ok = check_fastapi_health()
        
        status = "✓" if (vllm_ok and fastapi_ok) else "✗"
        write_stderr(
            f"[{timestamp}] {status} Health Check - "
            f"vLLM: {'✓' if vllm_ok else '✗'}, "
            f"FastAPI: {'✓' if fastapi_ok else '✗'}\n"
        )


if __name__ == '__main__':
    main()

