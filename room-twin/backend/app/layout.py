"""Room layout from floor masks + furniture size priors.

The top edge of the floor mask in each image column is where the floor meets
a wall (or the bottom of furniture standing in front of one). Casting those
pixels onto the floor plane gives a 2D point cloud of the room outline; the
room's orientation is the rotation that makes the points line up with the
axes best, and each wall sits where its points pile up.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import View

# Typical (width, height, depth) in metres; used for the unseen depth of an
# object and to flag detections whose measured size is implausible.
SIZE_PRIORS: dict[str, tuple[float, float, float]] = {
    "chair": (0.5, 0.9, 0.55), "armchair": (0.85, 0.9, 0.85), "sofa": (2.0, 0.85, 0.9),
    "table": (1.4, 0.75, 0.8), "desk": (1.3, 0.75, 0.65), "bed": (1.6, 0.9, 2.0),
    "nightstand": (0.5, 0.55, 0.4), "cabinet": (0.9, 0.9, 0.45), "shelf": (0.8, 1.8, 0.35),
    "wardrobe": (1.2, 2.0, 0.6), "lamp": (0.4, 1.5, 0.4), "plant": (0.5, 1.0, 0.5),
    "tv": (1.2, 0.7, 0.08), "monitor": (0.6, 0.45, 0.2), "stool": (0.4, 0.6, 0.4),
    "bench": (1.2, 0.45, 0.4), "ottoman": (0.6, 0.45, 0.6), "mirror": (0.6, 0.9, 0.03),
    "picture frame": (0.6, 0.5, 0.03),
}
FLAT = {"tv", "mirror", "picture frame"}


def prior_depth(label: str, width: float, mount: str) -> float:
    if mount == "wall" or label in FLAT:
        return SIZE_PRIORS.get(label, (0, 0, 0.05))[2] or 0.05
    pw, _, pd = SIZE_PRIORS.get(label, (width, 0, 0.6))
    # scale the typical depth with the measured width, within reason
    return float(np.clip(pd * width / max(pw, 1e-3), 0.5 * pd, 1.5 * pd)) if label in SIZE_PRIORS else min(width, 0.6)


def plausibility(label: str, w: float, h: float) -> float:
    """1.0 when the measured box matches the prior, falling toward 0.3 when it is way off."""
    if label not in SIZE_PRIORS:
        return 1.0
    pw, ph, _ = SIZE_PRIORS[label]
    off = max(abs(math.log(max(w, 0.05) / pw)), abs(math.log(max(h, 0.05) / ph)))
    return float(np.clip(1.0 - 0.5 * max(0.0, off - math.log(1.8)), 0.3, 1.0))


def wall_base_points(view: View, floor: np.ndarray, camera_height: float,
                     max_dist: float = 12.0, step: int = 4) -> np.ndarray:
    """Floor/wall contact points (x, z) seen in one view."""
    h, w = floor.shape
    pts = []
    for x in range(0, w, step):
        col = np.nonzero(floor[:, x])[0]
        if len(col) < 3:
            continue
        y = col.min()
        if y <= 1 or y >= h - 2:
            continue  # floor runs out of the frame: no wall in sight here
        ray = view.pixel_ray(x, y)
        if ray[1] > -0.01:
            continue
        p = np.array([0.0, camera_height, 0.0]) + ray * (camera_height / -ray[1])
        if 0.3 < math.hypot(p[0], p[2]) < max_dist:
            pts.append((p[0], p[2]))
    return np.array(pts).reshape(-1, 2)


def rotate(pts: np.ndarray, a: float) -> np.ndarray:
    """Rotate (x, z) points by yaw a (same convention as View.yaw)."""
    c, s = math.cos(a), math.sin(a)
    return np.stack([pts[:, 0] * c - pts[:, 1] * s, pts[:, 0] * s + pts[:, 1] * c], 1)


def room_rotation(pts: np.ndarray) -> float:
    """Yaw that best aligns the outline points with the x/z axes (in [-45, 45) deg)."""
    if len(pts) < 30:
        return 0.0
    best, best_a = -1.0, 0.0
    for deg in np.arange(-45, 45, 1.0):
        r = rotate(pts, math.radians(deg))
        score = 0.0
        for k in (0, 1):  # walls make sharp peaks in the coordinate histogram
            hist, _ = np.histogram(r[:, k], bins=np.arange(-12, 12.05, 0.1))
            score += float((hist.astype(float) ** 2).sum())
        if score > best:
            best, best_a = score, math.radians(deg)
    return best_a


@dataclass
class Bounds:
    min_x: float
    max_x: float
    min_z: float
    max_z: float

    @property
    def size(self):
        return self.max_x - self.min_x, self.max_z - self.min_z

    @property
    def center(self):
        return (self.min_x + self.max_x) / 2, (self.min_z + self.max_z) / 2

    seen: dict = None  # which walls were actually observed

    def include(self, x0, x1, z0, z1, pad=0.05):
        self.min_x, self.max_x = min(self.min_x, x0 - pad), max(self.max_x, x1 + pad)
        self.min_z, self.max_z = min(self.min_z, z0 - pad), max(self.max_z, z1 + pad)


def fit_bounds(pts: np.ndarray, min_points: int = 12) -> Bounds:
    """Each wall where its points pile up; sides we never saw stay None-ish (camera +/- 0.6 m)."""
    def side(vals: np.ndarray, sign: int) -> float | None:
        v = vals[vals * sign > 0.3] * sign
        if len(v) < min_points:
            return None
        # the wall is the far, dense end: a high percentile, then snap to the densest
        # 10 cm bin near it (furniture in front of the wall pulls points inward)
        guess = np.percentile(v, 90)
        near = v[np.abs(v - guess) < 0.5]
        hist, edges = np.histogram(near, bins=np.arange(guess - 0.5, guess + 0.55, 0.1))
        i = int(np.argmax(hist))
        return sign * float((edges[i] + edges[i + 1]) / 2)

    x0, x1 = side(pts[:, 0], -1) if len(pts) else None, side(pts[:, 0], 1) if len(pts) else None
    z0, z1 = side(pts[:, 1], -1) if len(pts) else None, side(pts[:, 1], 1) if len(pts) else None
    return Bounds(x0 if x0 is not None else -0.6, x1 if x1 is not None else 0.6,
                  z0 if z0 is not None else -0.6, z1 if z1 is not None else 0.6,
                  seen={"min_x": x0 is not None, "max_x": x1 is not None,
                        "min_z": z0 is not None, "max_z": z1 is not None})
