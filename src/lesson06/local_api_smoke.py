"""Lesson 6: smoke-test the local FastAPI app with stdlib only (no httpx)."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request


def main() -> None:
    base = "http://127.0.0.1:8765"
    try:
        with urllib.request.urlopen(f"{base}/api/health", timeout=5) as resp:
            print("GET /api/health ->", resp.status)
            print(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        print(
            f"GET /api/health failed: HTTP {exc.code}\n{body}",
            file=sys.stderr,
        )
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(f"Health check failed (is uvicorn running?): {exc}", file=sys.stderr)
        sys.exit(1)

    payload = {
        "prompt": "Reply with exactly the word: OK",
        "max_new_tokens": 8,
        "temperature": 0.2,
        "top_p": 0.9,
        "do_sample": False,
        "use_lora": False,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            print("POST /api/generate ->", resp.status)
            print(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        print(
            f"POST /api/generate failed: HTTP {exc.code}\n{body}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
