"""Camera / room geometry.

World frame (matches Three.js): +Y up, floor at y=0, the capture camera
stands at (0, camera_height, 0). Yaw 0 looks down -Z, positive yaw turns
toward +X. Pitch is positive looking up.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image


@dataclass
class View:
    """A perspective image with a known (assumed) orientation at the capture point."""
    id: str
    image: Image.Image
    yaw: float  # radians
    pitch: float = 0.0  # radians
    hfov: float = math.radians(65)
    kind: str = "photo"  # "photo" | "pano-crop"
    source: str = ""  # relative path of the original upload
    meta: dict = field(default_factory=dict)

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size

    @property
    def focal(self) -> float:
        w, _ = self.size
        return (w / 2) / math.tan(self.hfov / 2)

    def pixel_ray(self, x: float, y: float) -> np.ndarray:
        """Unit world-space direction through pixel (x, y)."""
        w, h = self.size
        f = self.focal
        d = np.array([(x - w / 2) / f, -(y - h / 2) / f, -1.0])
        # pitch: rotate about +X, then yaw: rotate about +Y (clockwise seen from above -> toward +X)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        d = np.array([d[0], cp * d[1] - sp * d[2], sp * d[1] + cp * d[2]])
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        d = np.array([cy * d[0] - sy * d[2], d[1], sy * d[0] + cy * d[2]])
        return d / np.linalg.norm(d)


def yaw_pitch_dir(yaw: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    cp = np.cos(pitch)
    return np.stack([np.sin(yaw) * cp, np.sin(pitch), -np.cos(yaw) * cp], axis=-1)


def is_equirectangular(img: Image.Image) -> bool:
    w, h = img.size
    return 1.9 <= w / h <= 2.1 and w >= 1024


def hfov_from_exif(img: Image.Image, default_deg: float = 65.0) -> float:
    """Horizontal FOV from the 35mm-equivalent focal length when the photo has it."""
    try:
        exif = img.getexif().get_ifd(0x8769)  # Exif IFD
        f35 = exif.get(0xA405)  # FocalLengthIn35mmFilm
        if f35:
            w, h = img.size
            sensor = 36.0 if w >= h else 24.0
            return 2 * math.atan(sensor / (2 * float(f35)))
    except Exception:
        pass
    return math.radians(default_deg)


def equirect_to_perspective(pano: Image.Image, yaw: float, pitch: float, hfov: float,
                            out_w: int = 1024, out_h: int = 768) -> Image.Image:
    """Resample a perspective crop out of an equirectangular panorama (bilinear)."""
    src = np.asarray(pano.convert("RGB"), dtype=np.float32)
    ph, pw, _ = src.shape
    view = View("tmp", Image.new("RGB", (out_w, out_h)), yaw, pitch, hfov)
    f = view.focal
    xs, ys = np.meshgrid(np.arange(out_w) + 0.5, np.arange(out_h) + 0.5)
    d = np.stack([(xs - out_w / 2) / f, -(ys - out_h / 2) / f, -np.ones_like(xs)], -1)
    cp, sp = math.cos(pitch), math.sin(pitch)
    d = np.stack([d[..., 0], cp * d[..., 1] - sp * d[..., 2], sp * d[..., 1] + cp * d[..., 2]], -1)
    cy, sy = math.cos(yaw), math.sin(yaw)
    d = np.stack([cy * d[..., 0] - sy * d[..., 2], d[..., 1], sy * d[..., 0] + cy * d[..., 2]], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    lon = np.arctan2(d[..., 0], -d[..., 2])  # 0 at image centre
    lat = np.arcsin(np.clip(d[..., 1], -1, 1))
    u = (lon / (2 * math.pi) + 0.5) * pw - 0.5
    v = (0.5 - lat / math.pi) * ph - 0.5
    u0 = np.floor(u).astype(int)
    v0 = np.clip(np.floor(v).astype(int), 0, ph - 1)
    v1 = np.clip(v0 + 1, 0, ph - 1)
    du, dv = (u - u0)[..., None], np.clip(v - v0, 0, 1)[..., None]
    u0m, u1m = u0 % pw, (u0 + 1) % pw
    out = (src[v0, u0m] * (1 - du) * (1 - dv) + src[v0, u1m] * du * (1 - dv)
           + src[v1, u0m] * (1 - du) * dv + src[v1, u1m] * du * dv)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


@dataclass
class Placement:
    position: tuple[float, float, float]  # base centre of the object
    size: tuple[float, float, float]  # width, height, depth (metres)
    rotation_y: float  # object faces the camera
    mount: str  # "floor" | "wall"


def place_from_mask(view: View, mask: np.ndarray, camera_height: float,
                    bounds: tuple[float, float, float, float] | None = None,
                    max_dist: float = 10.0, depth_fn=None) -> Placement | None:
    """Back-project a 2D mask into a box in the room.

    The lowest mask pixels are assumed to touch the floor, which pins the
    distance; width and height then follow from the mask extent at that
    distance. Objects whose bottom is above the horizon are wall-mounted and
    are placed on the room box `bounds` = (min_x, max_x, min_z, max_z); without
    bounds they are skipped (call again once the room is known).
    `depth_fn(width, mount)` supplies the unseen front-to-back depth.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return None
    y_bot = np.percentile(ys, 99)
    y_top = np.percentile(ys, 1)
    bottom_band = xs[ys >= y_bot - 2]
    x_mid = float(np.median(bottom_band))
    x_l, x_r = np.percentile(xs, 2), np.percentile(xs, 98)
    cam = np.array([0.0, camera_height, 0.0])

    ray = view.pixel_ray(x_mid, y_bot)
    mount = "floor"
    if ray[1] < -0.02:
        base = cam + ray * (camera_height / -ray[1])
    else:
        if bounds is None:
            return None
        mount = "wall"
        x0, x1, z0, z1 = bounds
        ts = []
        if ray[0] > 1e-6: ts.append(x1 / ray[0])
        if ray[0] < -1e-6: ts.append(x0 / ray[0])
        if ray[2] > 1e-6: ts.append(z1 / ray[2])
        if ray[2] < -1e-6: ts.append(z0 / ray[2])
        ts = [t for t in ts if t > 0]
        if not ts:
            return None
        base = cam + min(ts) * ray

    horiz = math.hypot(base[0], base[2])
    if horiz < 0.3 or horiz > max_dist:
        return None  # under the camera, or a sliver at the horizon (unreliable distance)

    def at_dist(px, py):  # point on the ray through (px, py) at the object's horizontal distance
        r = view.pixel_ray(px, py)
        return cam + r * (horiz / max(1e-6, math.hypot(r[0], r[2])))

    y_mid = (y_top + y_bot) / 2
    width = float(np.linalg.norm(at_dist(x_r, y_mid) - at_dist(x_l, y_mid)))
    height = float(at_dist(x_mid, y_top)[1] - base[1])
    if height < 0.1 or width < 0.05:
        return None
    width, height = min(width, 5.0), min(height, 3.0)
    obj_depth = depth_fn(width, mount) if depth_fn else min(width, 0.6 if mount == "floor" else 0.05)

    # push the base back by half the depth so the visible front face sits where we saw it
    dir_h = np.array([base[0], 0, base[2]]) / horiz
    center = base + dir_h * obj_depth / 2 if mount == "floor" else base - dir_h * obj_depth / 2
    rot = math.atan2(-center[0], -center[2])  # face toward the camera
    return Placement((float(center[0]), float(max(0.0, base[1])), float(center[2])),
                     (width, height, float(obj_depth)), rot, mount)


def estimate_pitch(img: Image.Image, hfov: float, max_deg: float = 35.0,
                   return_confidence: bool = False):
    """Camera tilt (radians, + = looking up) from the lean of vertical edges.

    With a level camera, vertical structure (door frames, wall corners, legs)
    stays vertical in the image. Tilting the camera makes those edges converge
    toward a vanishing point at y = cy - f / tan(pitch). Each near-vertical edge
    pixel at (x, y) with lean k = dx/dy gives  k * f = tan(pitch) * a,
    a = (x - cx) - (y - cy) * k, solved robustly (weighted median).
    Returns None when the image has too little vertical structure to tell.
    With return_confidence, returns (pitch, share of votes within 2 deg of the peak).
    """
    none = (None, 0.0) if return_confidence else None
    w0 = 640
    g = img.convert("L")
    if g.width > w0:
        g = g.resize((w0, round(g.height * w0 / g.width)), Image.BILINEAR)
    a = np.asarray(g, dtype=np.float32)
    h, w = a.shape
    f = (w / 2) / math.tan(hfov / 2)
    gx = np.zeros_like(a)
    gy = np.zeros_like(a)
    gx[1:-1, 1:-1] = (a[1:-1, 2:] - a[1:-1, :-2]) * 2 + (a[:-2, 2:] - a[:-2, :-2]) + (a[2:, 2:] - a[2:, :-2])
    gy[1:-1, 1:-1] = (a[2:, 1:-1] - a[:-2, 1:-1]) * 2 + (a[2:, :-2] - a[:-2, :-2]) + (a[2:, 2:] - a[:-2, 2:])
    # structure tensor: average gradient products over a 5x5 window so the edge
    # direction is continuous instead of snapping to the pixel grid
    jxx, jxy, jyy = _box(gx * gx, 2), _box(gx * gy, 2), _box(gy * gy, 2)
    mag = np.sqrt(jxx + jyy)
    coherent = (jxx - jyy) ** 2 + 4 * jxy ** 2 > (0.5 * (jxx + jyy)) ** 2
    sel = (mag > np.percentile(mag, 90)) & (jxx > 4 * jyy) & coherent
    # keep only pixels that sit on a vertical run (a real line, not texture)
    run = sel.copy()
    for s in range(1, 9):
        run[s:] &= sel[:-s] | np.roll(sel, 1, 1)[:-s] | np.roll(sel, -1, 1)[:-s]
    ys, xs = np.nonzero(run)
    if len(xs) < 200:
        return none
    k = -jxy[ys, xs] / jxx[ys, xs]  # edge runs perpendicular to the gradient: dx/dy
    xc, yc = xs - w / 2, ys - h / 2
    aa = xc - yc * k
    keep = (np.abs(aa) > 0.08 * w) & (np.abs(k) < 0.5)
    if keep.sum() < 150:
        return none
    t = k[keep] * f / aa[keep]
    wts = aa[keep] ** 2
    lim = math.tan(math.radians(max_deg))
    m = np.abs(t) < lim
    if m.sum() < 100:
        return none
    # vote in angle space and take the peak: robust to the scattered leans of
    # rug patterns / beams that aren't truly vertical
    ang = np.degrees(np.arctan(t[m]))
    bins = np.arange(-max_deg, max_deg + 0.25, 0.25)
    hist, _ = np.histogram(ang, bins=bins, weights=np.sqrt(wts[m]))
    total = hist.sum()
    kern = np.exp(-0.5 * (np.arange(-8, 9) / 3.0) ** 2)  # sigma = 0.75 deg
    hist = np.convolve(hist, kern, mode="same")
    peak = int(np.argmax(hist))
    pitch = math.radians((bins[peak] + bins[peak + 1]) / 2)
    if not return_confidence:
        return pitch
    near = np.abs(ang - math.degrees(pitch)) < 2
    return pitch, float(np.sqrt(wts[m])[near].sum() / max(total, 1e-6))


def _box(a: np.ndarray, r: int) -> np.ndarray:
    """Mean filter with a (2r+1)^2 window via cumulative sums."""
    p = np.pad(a.astype(np.float64), r + 1, mode="edge").cumsum(0).cumsum(1)
    n = 2 * r + 1
    return (p[n:, n:] - p[:-n, n:] - p[n:, :-n] + p[:-n, :-n])[: a.shape[0], : a.shape[1]] / (n * n)
