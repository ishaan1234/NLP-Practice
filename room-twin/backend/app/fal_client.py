"""fal.ai: hosted SAM 3 and SAM 3D Objects - no GPU, no gated downloads.

    fal-ai/sam-3/3d-objects   photo + mask -> 3D mesh (GLB)          $0.02 per object
    fal-ai/sam-3/image        photo + text prompt -> instance masks  $0.005 per prompt
(prices from fal.ai, Oct 2026). Set FAL_KEY (fal.ai/dashboard/keys) to turn them on.

Uses fal's queue API (submit, poll, fetch) over plain HTTP, so nothing extra to install and it
works the same locally and on Vercel. Images go up as data URIs.
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request

from PIL import Image

QUEUE = os.environ.get("FAL_QUEUE_URL", "https://queue.fal.run")  # override only for testing
KEY = os.environ.get("FAL_KEY", "")


def configured() -> bool:
    return bool(KEY)


def data_uri(img: Image.Image, fmt: str = "PNG") -> str:
    out = io.BytesIO()
    img.save(out, fmt, **({"quality": 92} if fmt == "JPEG" else {}))
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(out.getvalue()).decode()


def _request(url: str, body: dict | None = None, timeout: float = 60) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode() if body is not None else None,
        method="POST" if body is not None else "GET",
        headers={"Authorization": f"Key {KEY}", "Content-Type": "application/json", "User-Agent": "room-twin"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read()[:300].decode(errors="replace")
        if e.code in (401, 403):
            raise RuntimeError(f"fal.ai rejected the key (HTTP {e.code}): check FAL_KEY and your fal credit") from e
        raise RuntimeError(f"fal.ai HTTP {e.code}: {detail}") from e


def run(endpoint: str, payload: dict, timeout: float = 280) -> dict:
    """Submit to the queue, wait for the result. `timeout` stays under a serverless request limit."""
    sub = _request(f"{QUEUE}/{endpoint}", payload)
    status_url, response_url = sub["status_url"], sub["response_url"]
    deadline = time.time() + timeout
    delay = 0.5
    while time.time() < deadline:
        st = _request(status_url)
        if st.get("status") == "COMPLETED":
            return _request(response_url)
        if st.get("status") not in ("IN_QUEUE", "IN_PROGRESS"):
            raise RuntimeError(f"fal.ai {endpoint}: {st}")
        time.sleep(delay)
        delay = min(delay * 1.5, 3)
    raise TimeoutError(f"fal.ai {endpoint} took longer than {timeout:.0f} s")


def download(url: str, limit: int = 80 * 1024 * 1024) -> bytes:
    if url.startswith("data:"):
        return base64.b64decode(url.split(",", 1)[1])
    req = urllib.request.Request(url, headers={"User-Agent": "room-twin"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read(limit + 1)
    if len(data) > limit:
        raise ValueError("fal.ai result too large")
    return data
