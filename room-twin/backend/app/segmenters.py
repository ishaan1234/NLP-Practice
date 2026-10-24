"""2D object segmentation.

`Sam3Segmenter` uses Meta's SAM 3 (promptable concept segmentation): every
text prompt ("chair", "lamp", ...) returns all matching instance masks.
`FallbackSegmenter` is a dependency-free colour-region heuristic so the app
still works on a laptop without the model; it is a demo, not a substitute.
"""
from __future__ import annotations

import logging
import os
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


STRUCTURE_LABELS = {"floor", "rug", "door", "window"}


def nms(dets: list[Detection], iou_thr: float = 0.5) -> list[Detection]:
    """Drop overlapping duplicates. Room structure (floor, doors, ...) never competes with
    furniture: the pipeline needs both a door's mask and the wardrobe it was mistaken for."""
    structure = [d for d in dets if d.label in STRUCTURE_LABELS]
    keep: list[Detection] = []
    for d in sorted((d for d in dets if d.label not in STRUCTURE_LABELS), key=lambda d: d.score, reverse=True):
        if all(mask_iou(d.mask, k.mask) < iou_thr for k in keep):
            keep.append(d)
    return keep + structure


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


class GroundedSamSegmenter:
    """Open-vocabulary detection (Grounding DINO) + box-prompted masks (SAM).

    Both checkpoints are ungated on Hugging Face and run on CPU (a few seconds
    per image), so this is the middle ground when SAM 3 isn't available.
    """
    name = "grounding-dino+sam"

    def __init__(self, det_id: str = "IDEA-Research/grounding-dino-tiny",
                 sam_id: str = "facebook/sam-vit-base", box_threshold: float = 0.35):
        import torch
        from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel,
                                  SamProcessor)

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.det_proc = AutoProcessor.from_pretrained(det_id)
        self.det = AutoModelForZeroShotObjectDetection.from_pretrained(det_id).to(self.device).eval()
        self.sam_proc = SamProcessor.from_pretrained(sam_id)
        self.sam = SamModel.from_pretrained(sam_id).to(self.device).eval()
        self.box_threshold = box_threshold

    def _detect(self, image: Image.Image, prompts: list[str], size: dict | None = None):
        """Grounding DINO boxes for the prompts; `size` lets big targets run at lower resolution."""
        torch = self.torch
        W, H = image.size
        boxes, scores, labels = [], [], []
        # Grounding DINO's text window is 256 tokens: ~24 short nouns fit in one pass
        for i in range(0, len(prompts), 24):
            group = prompts[i:i + 24]
            text = " ".join(f"{p.lower()}." for p in group)
            kw = {"size": size} if size else {}
            inputs = self.det_proc(images=image, text=text, return_tensors="pt", **kw).to(self.device)
            with torch.no_grad():
                out = self.det(**inputs)
            res = self.det_proc.post_process_grounded_object_detection(
                out, inputs["input_ids"], threshold=self.box_threshold, text_threshold=0.25,
                target_sizes=[(H, W)])[0]
            names = res.get("text_labels", res.get("labels"))
            for box, score, name in zip(res["boxes"].tolist(), res["scores"].tolist(), names):
                label = next((p for p in group if p.lower() in str(name)), None)
                if label is None:
                    continue  # merged / partial phrase we can't map back to a prompt
                x0, y0, x1, y1 = box
                if label not in STRUCTURE_LABELS and (x1 - x0) * (y1 - y0) > 0.6 * W * H:
                    continue  # "the whole room" boxes
                boxes.append([x0, y0, x1, y1])
                scores.append(score)
                labels.append(label)
        return boxes, scores, labels

    def _masks(self, image: Image.Image, boxes, scores, labels) -> list[Detection]:
        """One SAM image encoding (the slow part) shared by every box."""
        if not boxes:
            return []
        torch = self.torch
        W, H = image.size
        inputs = self.sam_proc(image, input_boxes=[boxes], return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.sam(**inputs, multimask_output=False)
        masks = self.sam_proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0]
        masks = masks.reshape(len(boxes), H, W).numpy()
        return nms([Detection(l, m, float(s)) for l, m, s in zip(labels, masks, scores) if m.sum() > 50])

    def segment(self, image: Image.Image, prompts: list[str]) -> list[Detection]:
        return self._masks(image, *self._detect(image, prompts))

    def segment_with_structure(self, image: Image.Image, prompts: list[str],
                               structure: list[str]) -> list[Detection]:
        """Furniture and room structure in separate detector passes (in a joint pass a door
        comes back as "wardrobe"), both at full resolution, one shared SAM encoding."""
        b1, s1, l1 = self._detect(image, prompts)
        b2, s2, l2 = self._detect(image, structure)
        return self._masks(image, b1 + b2, s1 + s2, l1 + l2)


def load_segmenter():
    """SAM 3 if installed, else Grounding DINO + SAM, else the colour-region fallback.

    ROOM_TWIN_SEGMENTER=sam3|grounded-sam|fallback forces one.
    """
    choice = os.environ.get("ROOM_TWIN_SEGMENTER", "auto")
    order = {"sam3": [Sam3Segmenter], "grounded-sam": [GroundedSamSegmenter],
             "fallback": []}.get(choice, [Sam3Segmenter, GroundedSamSegmenter])
    for cls in order:
        try:
            seg = cls()
            log.info("segmenter: %s", seg.name)
            return seg
        except Exception as e:  # ImportError, missing / gated checkpoint, ...
            log.warning("%s unavailable (%s)", cls.__name__, e)
    log.warning("using fallback segmenter")
    return FallbackSegmenter()
