"""Photos / 360 panorama -> editable room scene (scene.json)."""
from __future__ import annotations

import json
import logging
import math
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .geometry import (View, equirect_to_perspective, estimate_pitch, hfov_from_exif,
                       is_equirectangular, place_from_mask)
from .layout import Bounds, fit_bounds, plausibility, prior_depth, room_rotation, wall_base_points
from .segmenters import DEFAULT_PROMPTS, Detection

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "projects"
PANO_CROP_YAWS = [i * 45 for i in range(8)]
PANO_CROP_PITCH = -15  # look slightly down: furniture lives below the horizon
PANO_CROP_FOV = 90


class Models:
    """Lazily-loaded singletons so the server starts instantly."""
    segmenter = None
    reconstructor = None
    _loaded = False

    @classmethod
    def load(cls):
        if not cls._loaded:
            from .reconstructors import load_reconstructor
            from .segmenters import load_segmenter
            cls.segmenter = load_segmenter()
            cls.reconstructor = load_reconstructor()
            cls._loaded = True
        return cls

    @classmethod
    def info(cls) -> dict:
        return {
            "loaded": cls._loaded,
            "segmenter": getattr(cls.segmenter, "name", None),
            "reconstructor": getattr(cls.reconstructor, "name", None),
        }


def project_dir(pid: str) -> Path:
    d = DATA_DIR / pid
    if not d.resolve().is_relative_to(DATA_DIR.resolve()):
        raise ValueError("bad project id")
    return d


def write_status(pid: str, stage: str, progress: float, **extra):
    status = {"stage": stage, "progress": round(progress, 3), "time": time.time(), **extra}
    (project_dir(pid) / "status.json").write_text(json.dumps(status))


def rel(pid: str, path: Path) -> str:
    return f"files/{pid}/{path.relative_to(project_dir(pid)).as_posix()}"


def build_views(pid: str, inputs: list[Path], mode: str, yaws_deg: list[float] | None,
                hfov_deg: float | None, pitches_deg: list[float | None] | None = None
                ) -> tuple[list[View], str | None]:
    """Turn uploads into oriented perspective views (+ the panorama path if any)."""
    imgs = [ImageOps.exif_transpose(Image.open(p)).convert("RGB") for p in inputs]
    pano_idx = None
    if mode == "360" or (mode == "auto" and len(imgs) >= 1 and is_equirectangular(imgs[0])):
        pano_idx = 0
    views: list[View] = []
    if pano_idx is not None:
        pano = imgs[pano_idx]
        if pano.width > 4096:
            pano = pano.resize((4096, 2048), Image.LANCZOS)
        for yaw in PANO_CROP_YAWS:
            crop = equirect_to_perspective(pano, math.radians(yaw), math.radians(PANO_CROP_PITCH),
                                           math.radians(PANO_CROP_FOV))
            views.append(View(f"pano_{yaw:03d}", crop, math.radians(yaw), math.radians(PANO_CROP_PITCH),
                              math.radians(PANO_CROP_FOV), "pano-crop", rel(pid, inputs[pano_idx])))
        return views, rel(pid, inputs[pano_idx])

    n = len(imgs)
    for i, (img, path) in enumerate(zip(imgs, inputs)):
        yaw = yaws_deg[i] if yaws_deg and i < len(yaws_deg) else i * 360.0 / n
        hfov = math.radians(hfov_deg) if hfov_deg else hfov_from_exif(Image.open(path))
        if img.width > 1600:
            img = img.resize((1600, round(img.height * 1600 / img.width)), Image.LANCZOS)
        manual = pitches_deg[i] if pitches_deg and i < len(pitches_deg) else None
        if manual is not None:
            pitch, source = math.radians(manual), "manual"
        else:  # people rarely hold the phone level: read the tilt off vertical edges
            est, conf = estimate_pitch(img, hfov, return_confidence=True)
            pitch, source = (est, "estimated") if est is not None and conf >= 0.12 else (0.0, "assumed level")
        views.append(View(f"view_{i}", img, math.radians(yaw), pitch, hfov, "photo", rel(pid, path),
                          meta={"pitchSource": source}))
    return views, None


def save_cutout(view: View, det: Detection, path: Path) -> list[int]:
    ys, xs = np.nonzero(det.mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    rgba = np.dstack([np.asarray(view.image), det.mask.astype(np.uint8) * 255])[y0:y1, x0:x1]
    Image.fromarray(rgba, "RGBA").save(path)
    return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]


def dedupe(objs: list[dict]) -> list[dict]:
    """Same object seen from neighbouring views / overlapping panorama crops.

    Two detections merge when their footprints overlap: centre distance below
    half the summed widths (min 0.5 m). The higher-scoring one wins.
    """
    keep: list[dict] = []
    for o in sorted(objs, key=lambda o: o["score"], reverse=True):
        p = np.array(o["position"])
        dup = False
        for k in keep:
            same_kind = k["label"] == o["label"] or "object" in (k["label"], o["label"])
            dist = max(0.5, 0.5 * (k["size"][0] + o["size"][0]))
            if same_kind and np.linalg.norm(np.array(k["position"]) - p) < dist:
                dup = True
                break
        if not dup:
            keep.append(o)
    return keep


def estimate_colors(views: list[View]) -> tuple[str, str]:
    floor, wall = [], []
    for v in views:
        a = np.asarray(v.image)
        h = a.shape[0]
        floor.append(a[int(h * 0.88):].reshape(-1, 3))
        wall.append(a[int(h * 0.1):int(h * 0.3)].reshape(-1, 3))
    hexc = lambda arr: "#%02x%02x%02x" % tuple(int(c) for c in np.median(np.concatenate(arr), 0))
    return hexc(floor), hexc(wall)


def _footprint(o: dict) -> tuple[float, float, float, float]:
    x, _, z = o["position"]
    w, _, d = o["size"]
    c, s = abs(math.cos(o["rotationY"])), abs(math.sin(o["rotationY"]))
    hx, hz = (w * c + d * s) / 2, (w * s + d * c) / 2
    return x - hx, x + hx, z - hz, z + hz


def _snap_to_walls(rot: float) -> float:
    """Furniture almost always lines up with the walls: round to the nearest 90 deg."""
    return round(rot / (math.pi / 2)) * (math.pi / 2)


def run_pipeline(pid: str, inputs: list[Path], mode: str = "auto", yaws: list[float] | None = None,
                 hfov: float | None = None, camera_height: float = 1.5,
                 prompts: list[str] | None = None, room: tuple[float, float, float] = (5.0, 4.0, 2.7),
                 reconstruct: bool = True, name: str = "My room",
                 pitches: list[float | None] | None = None):
    pdir = project_dir(pid)
    (pdir / "cutouts").mkdir(exist_ok=True)
    (pdir / "meshes").mkdir(exist_ok=True)
    try:
        write_status(pid, "loading models", 0.02)
        models = Models.load()
        write_status(pid, "preparing views", 0.08)
        views, pano = build_views(pid, inputs, mode, yaws, hfov, pitches)
        prompts = [p for p in (prompts or DEFAULT_PROMPTS) if p != "floor"]
        height = room[2]

        # 1. segment furniture + floor in every view
        raw: list[tuple[View, Detection]] = []
        floors: dict[str, np.ndarray] = {}
        for vi, view in enumerate(views):
            write_status(pid, f"segmenting {view.id} ({models.segmenter.name})", 0.1 + 0.5 * vi / len(views))
            for det in models.segmenter.segment(view.image, prompts + ["floor"]):
                if det.label == "floor":
                    floors[view.id] = floors.get(view.id, np.zeros_like(det.mask)) | det.mask
                elif det.label != "rug":  # flat; better as floor texture than a box
                    raw.append((view, det))

        # 2. room orientation + walls from the floor outline; from here on work in the room's frame
        def outline():
            pts = [wall_base_points(v, floors[v.id], camera_height) for v in views if v.id in floors]
            return np.concatenate(pts) if pts else np.zeros((0, 2))
        turn = room_rotation(outline())
        for v in views:
            v.yaw += turn
        pts = outline()
        layout_ok = len(pts) >= 30
        bounds = fit_bounds(pts) if layout_ok else Bounds(-room[0] / 2, room[0] / 2, -room[1] / 2, room[1] / 2)
        seen = bounds.seen if layout_ok else {}

        # 3. floor-standing objects pin themselves by where they touch the floor
        objects, wall_dets = [], []
        for view, det in raw:
            depth_fn = lambda w, m, label=det.label: prior_depth(label, w, m)
            pl = place_from_mask(view, det.mask, camera_height, None, depth_fn=depth_fn)
            if pl is None:
                wall_dets.append((view, det, depth_fn))
                continue
            objects.append(_make_object(pid, pdir, view, det, pl))

        # drop things seen through doorways (well past a wall we actually saw); grow for the rest
        kept = []
        for o in objects:
            x0, x1, z0, z1 = _footprint(o)
            outside = ((seen.get("min_x") and x1 < bounds.min_x - 0.3) or (seen.get("max_x") and x0 > bounds.max_x + 0.3)
                       or (seen.get("min_z") and z1 < bounds.min_z - 0.3) or (seen.get("max_z") and z0 > bounds.max_z + 0.3))
            if not outside:
                kept.append(o)
                bounds.include(x0, x1, z0, z1)
        objects = kept
        for side, grow in (("x", 2.5), ("z", 2.5)):  # a room is at least ~2.5 m across
            lo, hi = getattr(bounds, f"min_{side}"), getattr(bounds, f"max_{side}")
            if hi - lo < grow:
                pad = (grow - (hi - lo)) / 2
                setattr(bounds, f"min_{side}", lo - pad)
                setattr(bounds, f"max_{side}", hi + pad)

        # 4. wall-mounted objects go onto the walls we now know
        box = (bounds.min_x, bounds.max_x, bounds.min_z, bounds.max_z)
        for view, det, depth_fn in wall_dets:
            pl = place_from_mask(view, det.mask, camera_height, box, depth_fn=depth_fn)
            if pl is not None:
                objects.append(_make_object(pid, pdir, view, det, pl))
        objects = dedupe(objects)

        if reconstruct and models.reconstructor is not None:
            for i, o in enumerate(objects):
                write_status(pid, f"reconstructing {o['label']} ({i + 1}/{len(objects)})",
                             0.6 + 0.38 * i / len(objects))
                view, det = o["_det"]
                try:
                    res = models.reconstructor.reconstruct(view.image, det.mask, pdir / "meshes" / o["id"])
                    if res:
                        o["kind"], o["asset"] = res[0], rel(pid, res[1])
                except Exception as e:
                    log.exception("reconstruction failed for %s: %s", o["id"], e)
        for o in objects:
            o.pop("_det", None)

        view_meta = []
        for view in views:
            vpath = pdir / "views" / f"{view.id}.jpg"
            vpath.parent.mkdir(exist_ok=True)
            view.image.save(vpath, quality=90)
            view_meta.append({"id": view.id, "image": rel(pid, vpath), "source": view.source,
                              "yaw": round(math.degrees(view.yaw), 1), "pitch": round(math.degrees(view.pitch), 1),
                              "hfov": round(math.degrees(view.hfov), 1), "kind": view.kind, **view.meta})

        floor_c, wall_c = estimate_colors(views)
        w, d = bounds.size
        scene = {
            "version": 1, "id": pid, "name": name, "created": time.time(),
            "room": {"width": round(w, 2), "depth": round(d, 2), "height": height,
                     "center": [round(c, 3) for c in bounds.center],
                     "floorColor": floor_c, "wallColor": wall_c, "cameraHeight": camera_height,
                     "panoYaw": round(turn, 4), "layout": "measured" if layout_ok else "guessed"},
            "panorama": pano, "views": view_meta, "objects": objects,
            "models": Models.info(),
        }
        (pdir / "scene.json").write_text(json.dumps(scene, indent=1))
        write_status(pid, "done", 1.0, objects=len(objects))
    except Exception as e:
        log.exception("pipeline failed")
        write_status(pid, "error", 1.0, error=str(e))


def _make_object(pid: str, pdir: Path, view: View, det: Detection, pl) -> dict:
    w, h, _ = pl.size
    score = det.score * plausibility(det.label, w, h)
    if det.mask[:, 0].any() or det.mask[:, -1].any():
        score *= 0.6  # cut off by the frame: prefer a view that sees all of it
    oid = uuid.uuid4().hex[:8]
    cpath = pdir / "cutouts" / f"{oid}.png"
    bbox = save_cutout(view, det, cpath)
    return {
        "id": oid, "label": det.label, "score": round(score, 3),
        "kind": "cutout", "cutout": rel(pid, cpath), "asset": None,
        "position": [round(v, 3) for v in pl.position], "rotationY": _snap_to_walls(pl.rotation_y),
        "scale": [1, 1, 1], "size": [round(v, 3) for v in pl.size], "mount": pl.mount,
        "source": {"view": view.id, "bbox": bbox}, "_det": (view, det),
    }


def make_item_from_photo(pid: str, image_path: Path, prompt: str | None) -> dict:
    """Drop-in item: segment the main object in a product photo and lift it to 3D."""
    models = Models.load()
    pdir = project_dir(pid)
    img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    if img.width > 1600:
        img = img.resize((1600, round(img.height * 1600 / img.width)), Image.LANCZOS)
    dets = models.segmenter.segment(img, [prompt] if prompt else DEFAULT_PROMPTS)
    if dets:
        det = max(dets, key=lambda d: d.mask.sum() * d.score)
    else:  # product shot on plain background: take everything that differs from the corners
        a = np.asarray(img, dtype=np.float32)
        bg = np.median(np.concatenate([a[:8, :8], a[:8, -8:], a[-8:, :8], a[-8:, -8:]]).reshape(-1, 3), 0)
        det = Detection(prompt or "item", np.abs(a - bg).sum(-1) > 60, 0.5)
        if det.mask.sum() < 100:
            det.mask[:] = True
    oid = uuid.uuid4().hex[:8]
    (pdir / "cutouts").mkdir(exist_ok=True)
    (pdir / "meshes").mkdir(exist_ok=True)
    view = View("item", img, 0.0)
    cpath = pdir / "cutouts" / f"{oid}.png"
    bbox = save_cutout(view, det, cpath)
    aspect = bbox[2] / max(1, bbox[3])
    h = 0.9
    item = {"id": oid, "label": prompt or det.label, "score": round(det.score, 3), "kind": "cutout",
            "cutout": rel(pid, cpath), "asset": None, "position": [0, 0, 0], "rotationY": 0,
            "scale": [1, 1, 1], "size": [round(h * aspect, 2), h, round(min(h * aspect, 0.6), 2)],
            "mount": "floor", "source": {"view": "upload", "bbox": bbox}}
    if models.reconstructor is not None:
        try:
            res = models.reconstructor.reconstruct(img, det.mask, pdir / "meshes" / oid)
            if res:
                item["kind"], item["asset"] = res[0], rel(pid, res[1])
        except Exception:
            log.exception("item reconstruction failed")
    return item
