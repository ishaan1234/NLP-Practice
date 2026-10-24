"""Last resort for "anything": text -> picture -> the item-from-photo pipeline.

When neither catalog has the thing ("a cat-shaped bookend"), a text-to-image model draws it
on a plain background; the normal item pipeline then cuts it out and, with SAM 3D Objects
installed, lifts it to a 3D mesh (otherwise it becomes a photo cut-out you can still move).

Optional: needs `pip install diffusers` (+ torch). Model via ROOM_TWIN_TEXT2IMG
(default stabilityai/sd-turbo: fast, 1-4 steps, check its license for commercial use).
"""
from __future__ import annotations

import logging
import os
import threading
import uuid
from pathlib import Path

log = logging.getLogger(__name__)

MODEL = os.environ.get("ROOM_TWIN_TEXT2IMG", "stabilityai/sd-turbo")
_pipe = None
_failed: str | None = None
_lock = threading.Lock()


def _load():
    global _pipe, _failed
    if _pipe is not None or _failed:
        return _pipe
    try:
        import torch
        from diffusers import AutoPipelineForText2Image

        cuda = torch.cuda.is_available()
        _pipe = AutoPipelineForText2Image.from_pretrained(
            MODEL, torch_dtype=torch.float16 if cuda else torch.float32)
        _pipe = _pipe.to("cuda" if cuda else "cpu")
        _pipe.set_progress_bar_config(disable=True)
    except Exception as e:  # not installed, offline, no disk...
        _failed = str(e)
        log.warning("text-to-image unavailable: %s", e)
    return _pipe


def available() -> bool:
    try:
        import diffusers  # noqa: F401
        return not _failed
    except ImportError:
        return False


def picture(description: str, out_dir: Path) -> Path:
    """Draw one isolated object on white; returns the image path."""
    with _lock:  # one generation at a time; the model is large
        pipe = _load()
        if pipe is None:
            raise RuntimeError(f"text-to-image unavailable: {_failed}")
        turbo = "turbo" in MODEL
        image = pipe(
            prompt=f"a single {description}, whole object fully in frame, three-quarter view, "
                   "product photo, plain white background, soft studio lighting",
            negative_prompt=None if turbo else "cropped, multiple objects, text, watermark",
            num_inference_steps=4 if turbo else 25, guidance_scale=0.0 if turbo else 7.0,
            height=512, width=512,
        ).images[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"gen_{uuid.uuid4().hex[:8]}.png"
    image.save(path)
    return path
