"""Room Twin GPU worker: SAM 3 segmentation + SAM 3D Objects reconstruction over HTTP.

Runs on a GPU machine (e.g. a Colab A100 - see README.md here) and is called by the Room Twin
backend when ROOM_TWIN_GPU_URL / ROOM_TWIN_GPU_TOKEN are set, so a laptop or a Vercel
deployment gets real masks and real 3D meshes without a local GPU.

SAM 3 and SAM 3D Objects need different Python environments, so this runs as two processes:
    python worker.py main  --port 8000   # SAM 3 in-process; forwards /reconstruct to :8002
    python worker.py sam3d --port 8002   # inside the sam3d-objects environment
Every request must carry X-Worker-Token equal to $ROOM_TWIN_GPU_TOKEN.
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import secrets
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "backend"))  # reuse the app's own model adapters

TOKEN = os.environ.get("ROOM_TWIN_GPU_TOKEN", "")
MAX_BYTES = 20 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 40_000_000
app = FastAPI(docs_url=None, redoc_url=None)
gpu_lock = threading.Lock()  # one model call at a time on the GPU
_models: dict = {}


def check(token: str | None):
    if len(TOKEN) < 16:
        raise HTTPException(503, "worker token not configured")
    if not token or not secrets.compare_digest(token, TOKEN):
        raise HTTPException(401, "bad worker token")


async def read_image(f: UploadFile, mode: str = "RGB") -> Image.Image:
    data = await f.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "image too large")
    try:
        return Image.open(io.BytesIO(data)).convert(mode)
    except Exception:
        raise HTTPException(415, "unreadable image")


def png_b64(mask: np.ndarray) -> str:
    out = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255).convert("1").save(out, "PNG", optimize=True)
    return base64.b64encode(out.getvalue()).decode()


# ---------------------------------------------------------------- main process (SAM 3)
def segmenter():
    if "sam3" not in _models:
        from app.segmenters import Sam3Segmenter
        _models["sam3"] = Sam3Segmenter()
    return _models["sam3"]


@app.get("/health")
def health(x_worker_token: str | None = Header(None)):
    check(x_worker_token)
    info = {"role": ROLE}
    if ROLE == "main":
        import requests
        try:
            info["sam3d"] = requests.get(f"{SAM3D_URL}/health", headers={"X-Worker-Token": TOKEN}, timeout=5).ok
        except Exception:
            info["sam3d"] = False
    return info


@app.post("/segment")
async def segment(image: UploadFile = File(...), prompts: str = Form(...),
                  x_worker_token: str | None = Header(None)):
    """-> {detections: [{label, score, mask: base64 1-bit PNG}]}"""
    check(x_worker_token)
    img = await read_image(image)
    wanted = [p.strip() for p in prompts.split(",") if p.strip()][:64]
    with gpu_lock:
        dets = segmenter().segment(img, wanted)
    return {"detections": [{"label": d.label, "score": d.score, "mask": png_b64(d.mask)} for d in dets]}


@app.post("/reconstruct")
async def reconstruct(image: UploadFile = File(...), mask: UploadFile = File(...),
                      x_worker_token: str | None = Header(None)):
    """-> GLB mesh (X-Kind: mesh) or coloured point-cloud PLY (X-Kind: pointcloud)."""
    check(x_worker_token)
    if ROLE == "main":  # SAM 3D lives in its own environment: forward
        import requests
        files = {"image": ("image.png", await image.read(), "image/png"),
                 "mask": ("mask.png", await mask.read(), "image/png")}
        r = requests.post(f"{SAM3D_URL}/reconstruct", files=files, headers={"X-Worker-Token": TOKEN}, timeout=600)
        if not r.ok:
            raise HTTPException(r.status_code, r.text[:300])
        return Response(r.content, media_type=r.headers.get("content-type"), headers={"X-Kind": r.headers.get("X-Kind", "mesh")})

    img = await read_image(image)
    m = np.asarray(await read_image(mask, "L")) > 127
    if m.shape != (img.height, img.width) or not m.any():
        raise HTTPException(422, "mask must match the image and select something")
    if "sam3d" not in _models:
        from app.reconstructors import Sam3DReconstructor
        _models["sam3d"] = Sam3DReconstructor()
    with gpu_lock, tempfile.TemporaryDirectory() as tmp:
        res = _models["sam3d"].reconstruct(img, m, Path(tmp) / "out")
        if res is None:
            raise HTTPException(502, "SAM 3D returned nothing usable")
        kind, path = res
        data = Path(path).read_bytes()
    return Response(data, media_type="model/gltf-binary" if kind == "mesh" else "application/octet-stream",
                    headers={"X-Kind": kind})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=["main", "sam3d"])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--sam3d-url", default="http://127.0.0.1:8002")
    args = ap.parse_args()
    ROLE, SAM3D_URL = args.role, args.sam3d_url
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")
