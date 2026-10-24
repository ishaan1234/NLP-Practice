"""Phone capture: scan a QR code on the desktop, shoot photos on the phone, they appear on the desktop.

Design (adapted from RoomShift's pairing flow, simplified for home-network use):
- The desktop creates a session and keeps an owner secret; the phone gets a separate
  upload secret inside the URL *fragment* (`/capture.html#<id>.<secret>`), so it is
  never sent in request URLs, server logs or referrers.
- Sessions expire after an hour; photos are re-encoded (EXIF orientation applied,
  metadata such as GPS stripped, only the lens focal length kept) and size-limited.
- The desktop polls for new photos; no WebSocket needed. Sessions and photos live in the
  project store (storage.py), so this also works across Vercel's short-lived instances.
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import secrets
import socket
import threading
import time
import warnings

from fastapi import APIRouter, File, Form, Header, HTTPException, Request, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from .storage import DATA_DIR, store

router = APIRouter()

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_PIXELS = 40_000_000
MAX_SIDE = 2560
MAX_PANO_SIDE = 6144
MAX_PHOTOS = 40
SESSION_SECONDS = 3600

Image.MAX_IMAGE_PIXELS = MAX_PIXELS
warnings.simplefilter("error", Image.DecompressionBombWarning)

_lock = threading.Lock()


def sanitize_image(data: bytes) -> bytes:
    """Decode, upright, downscale and re-encode an upload as JPEG.

    Strips all metadata (GPS, device ids) except FocalLengthIn35mmFilm, which the
    pipeline uses to work out the lens field of view.
    """
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Photo is larger than 15 MB")
    try:
        with Image.open(io.BytesIO(data)) as src:
            if src.format not in ("JPEG", "PNG", "WEBP", "MPO", "BMP"):
                raise HTTPException(415, "Use a JPEG, PNG or WebP photo (convert HEIC to JPEG first)")
            focal = src.getexif().get_ifd(0x8769).get(0xA405)
            img = ImageOps.exif_transpose(src).convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise HTTPException(415, "Could not read this image")
    side = MAX_PANO_SIDE if 1.9 <= img.width / img.height <= 2.1 else MAX_SIDE  # keep 360s sharp
    img.thumbnail((side, side), Image.LANCZOS)
    exif = Image.Exif()
    if focal:
        exif[0x8769] = {0xA405: int(focal)}
    out = io.BytesIO()
    img.save(out, "JPEG", quality=90, exif=exif.tobytes())
    return out.getvalue()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


# Sessions and their photos go through the project store, so they work the same on one local
# server and on Vercel, where the phone's upload and the desktop's poll can hit different
# short-lived instances. A session is stored like a project named "cap<id>" with one document.
def _key(sid: str) -> str:
    return f"cap{sid}"


def _get(sid: str) -> dict:
    s = store.read(_key(sid), "capture") if re.fullmatch(r"[0-9a-f]{16}", sid) else None
    if not s or s["expires"] < time.time():
        if s:
            store.delete(_key(sid))
        raise HTTPException(410, "This capture session has ended. Create a new QR code on the desktop.")
    return s


def _save(sid: str, s: dict):
    (DATA_DIR / _key(sid)).mkdir(parents=True, exist_ok=True)
    store.write(_key(sid), "capture", s)


def _check(s: dict, secret: str | None, key: str):
    if not secret or not secrets.compare_digest(_digest(secret), s[key]):
        raise HTTPException(403, "Not allowed")


def _lan_ip() -> str | None:
    """This machine's LAN address (no packets are sent; UDP connect just picks a route)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("10.255.255.255", 1))
            return sock.getsockname()[0]
    except OSError:
        return None


def _phone_base(request: Request) -> tuple[str, str | None]:
    """URL the phone should open, and a warning if it probably can't reach it."""
    public = os.environ.get("PUBLIC_URL", "").rstrip("/")
    if public:
        return public, None
    base = str(request.base_url).rstrip("/")
    host = request.url.hostname or ""
    if host in ("localhost", "127.0.0.1", "::1"):
        ip = _lan_ip()
        if ip:
            base = base.replace(host, ip, 1)
            return base, ("Phone can't open it? Use the same Wi-Fi, and start the server with "
                          "--host 0.0.0.0 so other devices can reach it.")
        return base, "Couldn't find this computer's network address; set PUBLIC_URL."
    return base, None


@router.post("/api/capture")
def create_session(request: Request):
    sid = secrets.token_hex(8)
    owner, phone = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
    _save(sid, {"owner": _digest(owner), "phone": _digest(phone), "photos": [],
                "expires": time.time() + SESSION_SECONDS, "phone_seen": False})
    base, warning = _phone_base(request)
    return {"id": sid, "owner": owner, "url": f"{base}/capture.html#{sid}.{phone}",
            "warning": warning, "expires_in": SESSION_SECONDS}


@router.get("/api/capture/{sid}")
def session_state(sid: str, x_capture_owner: str | None = Header(None), x_capture_phone: str | None = Header(None)):
    s = _get(sid)
    if x_capture_phone:  # the phone checks in when it opens the page
        _check(s, x_capture_phone, "phone")
        if not s["phone_seen"]:
            s["phone_seen"] = True
            _save(sid, s)
    else:
        _check(s, x_capture_owner, "owner")
    return {"phone_connected": s["phone_seen"],
            "photos": [{"id": p["id"], "mode": p["mode"], "time": p["time"]} for p in s["photos"]]}


@router.post("/api/capture/{sid}/photos")
async def upload(sid: str, photo: UploadFile = File(...), mode: str = Form("room"),
                 x_capture_phone: str | None = Header(None)):
    s = _get(sid)
    _check(s, x_capture_phone, "phone")
    if mode not in ("room", "item"):
        raise HTTPException(400, "mode must be room or item")
    if len(s["photos"]) >= MAX_PHOTOS:
        raise HTTPException(413, f"Up to {MAX_PHOTOS} photos per session")
    data = sanitize_image(await photo.read(MAX_UPLOAD_BYTES + 1))
    pid = secrets.token_hex(6)
    path = DATA_DIR / _key(sid) / "photos" / f"{pid}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    store.publish(_key(sid), [path])
    with _lock:  # the phone sends one photo at a time; this guards a single server's threads
        s = _get(sid)
        s["photos"].append({"id": pid, "mode": mode, "time": time.time()})
        s["phone_seen"] = True
        _save(sid, s)
    return {"id": pid}


@router.get("/api/capture/{sid}/photos/{pid}")
def get_photo(sid: str, pid: str, x_capture_owner: str | None = Header(None)):
    s = _get(sid)
    _check(s, x_capture_owner, "owner")
    if not any(p["id"] == pid for p in s["photos"]):
        raise HTTPException(404, "Photo not found")
    res = store.file(_key(sid), f"photos/{pid}.jpg")
    res.headers["Cache-Control"] = "no-store"
    return res


@router.delete("/api/capture/{sid}")
def end_session(sid: str, x_capture_owner: str | None = Header(None)):
    s = _get(sid)
    _check(s, x_capture_owner, "owner")
    store.delete(_key(sid))
    return {"ok": True}
