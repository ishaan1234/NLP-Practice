"""2D object segmentation.

`Sam3Segmenter` uses Meta's SAM 3 (promptable concept segmentation): every
text prompt ("chair", "lamp", ...) returns all matching instance masks.
`FallbackSegmenter` is a dependency-free colour-region heuristic so the app
still works on a laptop without the model; it is a demo, not a substitute.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

DEFAULT_PROMPTS = [
    "chair", "armchair", "sofa", "table", "desk", "bed", "nightstand", "cabinet",
    "shelf", "wardrobe", "lamp", "plant", "tv", "monitor", "stool", "bench",
    "ottoman", "rug", "mirror", "picture frame",
]


@dataclass
class Detection:
    label: str
    mask: np.ndarray  # HxW bool, same size as the input image
    score: float


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    return float(inter / np.logical_or(a, b).sum())


def nms(dets: list[Detection], iou_thr: float = 0.5) -> list[Detection]:
    keep: list[Detection] = []
    for d in sorted(dets, key=lambda d: d.score, reverse=True):
        if all(mask_iou(d.mask, k.mask) < iou_thr for k in keep):
            keep.append(d)
    return keep


class Sam3Segmenter:
    name = "sam3"

    def __init__(self, score_threshold: float = 0.5):
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.processor = Sam3Processor(build_sam3_image_model())
        self.score_threshold = score_threshold

    def segment(self, image: Image.Image, prompts: list[str]) -> list[Detection]:
        state = self.processor.set_image(image)
        dets: list[Detection] = []
        for prompt in prompts:
            out = self.processor.set_text_prompt(state=state, prompt=prompt)
            masks, scores = _to_numpy(out["masks"]), _to_numpy(out["scores"]).reshape(-1)
            masks = masks.reshape(-1, *masks.shape[-2:]) if masks.size else masks
            for m, s in zip(masks, scores):
                if s >= self.score_threshold:
                    dets.append(Detection(prompt, m > 0.5, float(s)))
        return nms(dets)


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x)


class FallbackSegmenter:
    """K-means colour regions + connected components on a downsampled image.

    Large regions touching the image border are treated as walls / floor /
    ceiling; compact regions in between become "object" detections.
    """
    name = "fallback"

    def __init__(self, work_width: int = 160, k: int = 7, seed: int = 0):
        self.work_width = work_width
        self.k = k
        self.rng = np.random.default_rng(seed)

    def segment(self, image: Image.Image, prompts: list[str]) -> list[Detection]:
        W, H = image.size
        w = self.work_width
        h = max(8, round(H * w / W))
        small = np.asarray(image.convert("RGB").resize((w, h), Image.BILINEAR), dtype=np.float32)
        labels = self._kmeans(small.reshape(-1, 3)).reshape(h, w)
        area = w * h
        dets = []
        for comp in self._components(labels):
            n = comp.sum()
            if not (0.004 * area <= n <= 0.22 * area):
                continue
            ys, xs = np.nonzero(comp)
            if ys.min() == 0 or (xs.min() == 0 and xs.max() == w - 1):
                continue  # ceiling / wall bands
            bh, bw = np.ptp(ys) + 1, np.ptp(xs) + 1
            fill = n / (bh * bw)
            if fill < 0.35 or bw > 0.7 * w:
                continue
            mask = np.asarray(Image.fromarray(comp.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127
            dets.append(Detection("object", mask, float(min(0.99, 0.4 + fill * 0.5))))
        return nms(dets, 0.3)[:12]

    def _kmeans(self, px: np.ndarray, iters: int = 10) -> np.ndarray:
        centers = px[self.rng.choice(len(px), self.k, replace=False)]
        for _ in range(iters):
            d = ((px[:, None, :] - centers[None]) ** 2).sum(-1)
            lab = d.argmin(1)
            for i in range(self.k):
                sel = px[lab == i]
                if len(sel):
                    centers[i] = sel.mean(0)
        return lab

    @staticmethod
    def _components(labels: np.ndarray):
        h, w = labels.shape
        seen = np.zeros_like(labels, dtype=bool)
        for sy in range(h):
            for sx in range(w):
                if seen[sy, sx]:
                    continue
                lab = labels[sy, sx]
                comp = np.zeros_like(seen)
                q = deque([(sy, sx)])
                seen[sy, sx] = True
                while q:
                    y, x = q.popleft()
                    comp[y, x] = True
                    for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                        if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and labels[ny, nx] == lab:
                            seen[ny, nx] = True
                            q.append((ny, nx))
                yield comp


def load_segmenter():
    try:
        seg = Sam3Segmenter()
        log.info("SAM 3 loaded")
        return seg
    except Exception as e:  # ImportError, missing checkpoint, no GPU, ...
        log.warning("SAM 3 unavailable (%s); using fallback segmenter", e)
        return FallbackSegmenter()
