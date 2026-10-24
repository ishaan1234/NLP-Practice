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
                    room_half: tuple[float, float], max_dist: float = 8.0) -> Placement | None:
    """Back-project a 2D mask into a box in the room.

    The lowest mask pixels are assumed to touch the floor, which pins the
    depth; width and height then follow from the mask extent at that depth.
    Objects whose bottom is above the horizon are treated as wall-mounted.
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
        t = camera_height / -ray[1]
        base = cam + t * ray
    else:
        # hits a wall: intersect with the room box
        mount = "wall"
        hx, hz = room_half
        ts = [t for t in ((hx / ray[0]) if abs(ray[0]) > 1e-6 else math.inf,
                          (-hx / ray[0]) if abs(ray[0]) > 1e-6 else math.inf,
                          (hz / ray[2]) if abs(ray[2]) > 1e-6 else math.inf,
                          (-hz / ray[2]) if abs(ray[2]) > 1e-6 else math.inf) if t > 0]
        t = min(ts) if ts else 3.0
        base = cam + t * ray

    horiz = math.hypot(base[0], base[2])
    if horiz < 0.3 or horiz > max_dist:
        return None  # under the camera, or a sliver at the horizon (unreliable depth)
    # depth along the viewing direction, used to size width / height
    fwd = view.pixel_ray(view.size[0] / 2, view.size[1] / 2)
    fwd_h = np.array([fwd[0], 0, fwd[2]]) / max(1e-6, math.hypot(fwd[0], fwd[2]))
    depth = max(0.3, float(np.dot(base - cam, fwd_h)) if mount == "floor" else horiz)

    width = abs(x_r - x_l) / view.focal * depth
    top_ray = view.pixel_ray(x_mid, y_top)
    t_top = horiz / max(1e-6, math.hypot(top_ray[0], top_ray[2]))
    top_y = camera_height + t_top * top_ray[1]
    height = max(0.05, top_y - base[1])
    width = float(np.clip(width, 0.1, 4.0))
    if height < 0.1:
        return None
    height = float(min(height, 3.0))
    obj_depth = min(width, 0.6 if mount == "floor" else 0.05)

    # push the base back by half the depth so the visible front face sits where we saw it
    dir_h = np.array([base[0], 0, base[2]]) / horiz
    center = base + dir_h * obj_depth / 2 if mount == "floor" else base - dir_h * obj_depth / 2
    rot = math.atan2(-center[0], -center[2])  # face toward the camera
    return Placement((float(center[0]), float(max(0.0, base[1])), float(center[2])),
                     (width, height, obj_depth), rot, mount)
