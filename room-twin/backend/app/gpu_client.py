"""Client for a remote Room Twin GPU worker (gpu-worker/worker.py, e.g. on a Colab A100).

Set ROOM_TWIN_GPU_URL and ROOM_TWIN_GPU_TOKEN (printed by gpu-worker/start.sh) and the app
sends segmentation and 3D reconstruction there - from a laptop or from Vercel.
"""
from __future__ import annotations

import io
import json
import os
import secrets
import urllib.error
import urllib.request

from PIL import Image

URL = os.environ.get("ROOM_TWIN_GPU_URL", "").rstrip("/")
TOKEN = os.environ.get("ROOM_TWIN_GPU_TOKEN", "")


def configured() -> bool:
    return bool(URL and TOKEN)


def _png(img: Image.Image) -> bytes:
    out = io.BytesIO()
    img.save(out, "PNG")
    return out.getvalue()


def call(path: str, fields: dict[str, str] | None = None, files: dict[str, bytes] | None = None,
         timeout: float = 600) -> tuple[dict, bytes]:
    """POST multipart (or GET when nothing to send); returns (headers, body)."""
    boundary = secrets.token_hex(16)
    body = io.BytesIO()
    for name, value in (fields or {}).items():
        body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    for name, data in (files or {}).items():
        body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{name}.png"\r\n'
                   f'Content-Type: image/png\r\n\r\n'.encode())
        body.write(data)
        body.write(b"\r\n")
    has_body = bool(fields or files)
    if has_body:
        body.write(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        URL + path, data=body.getvalue() if has_body else None, method="POST" if has_body else "GET",
        headers={"X-Worker-Token": TOKEN, "User-Agent": "room-twin",
                 **({"Content-Type": f"multipart/form-data; boundary={boundary}"} if has_body else {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"GPU worker {path}: HTTP {e.code} {e.read()[:200]!r}") from e


def health() -> dict:
    return json.loads(call("/health", timeout=15)[1])


def segment(image: Image.Image, prompts: list[str]) -> list[dict]:
    _, body = call("/segment", {"prompts": ",".join(prompts)}, {"image": _png(image)}, timeout=300)
    return json.loads(body)["detections"]


def reconstruct(image: Image.Image, mask_png: bytes) -> tuple[str, bytes]:
    headers, body = call("/reconstruct", files={"image": _png(image), "mask": mask_png}, timeout=900)
    return headers.get("X-Kind", "mesh"), body
