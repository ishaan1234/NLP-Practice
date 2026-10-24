"""Breadth: Objaverse-LVIS (46k models in 1,156 everyday categories - cat, vase, kennel, piano...).

Models come from Sketchfab creators, mostly under CC-BY, so each one carries its author and
license for attribution. Their scale is arbitrary - the caller gives the real-world size.
Index files are downloaded once (~20 MB) and trimmed; a model's metadata chunk is fetched the
first time one of its models is considered.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import threading
import urllib.request

from .storage import DATA_DIR, store

log = logging.getLogger(__name__)

HF = "https://huggingface.co/datasets/allenai/objaverse/resolve/main"
DIR = DATA_DIR.parent / "objaverse"
LIB = DATA_DIR / "lib" / "objaverse"  # store key "lib", served at /library/
MAX_GLB = 30 * 1024 * 1024
MAX_FACES = 200_000
LICENSES = {"by": "CC-BY 4.0", "by-sa": "CC-BY-SA 4.0", "cc0": "CC0", "by-nd": "CC-BY-ND 4.0",
            "by-nc": "CC-BY-NC 4.0 (non-commercial)", "by-nc-sa": "CC-BY-NC-SA 4.0 (non-commercial)"}

_lock = threading.Lock()
_index: dict | None = None


def _get(url: str, limit: int = MAX_GLB) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "room-twin"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read(limit + 1)
    if len(data) > limit:
        raise ValueError("download too large")
    return data


def index() -> dict:
    """{"categories": {name: [uid, ...]}, "paths": {uid: "glbs/000-xxx/uid.glb"}}"""
    global _index
    with _lock:
        if _index is not None:
            return _index
        f = DIR / "lvis-index.json"
        if f.exists():
            _index = json.loads(f.read_text())
            return _index
        try:
            cats = json.loads(gzip.decompress(_get(f"{HF}/lvis-annotations.json.gz", 50 << 20)))
            paths = json.loads(gzip.decompress(_get(f"{HF}/object-paths.json.gz", 100 << 20)))
            wanted = {u for uids in cats.values() for u in uids}
            _index = {"categories": cats, "paths": {u: p for u, p in paths.items() if u in wanted}}
            DIR.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(_index))
        except Exception as e:
            log.warning("Objaverse index unavailable: %s", e)
            _index = {"categories": {}, "paths": {}}
        return _index


def categories() -> list[str]:
    return sorted(index()["categories"])


def _meta(uid: str) -> dict | None:
    chunk = index()["paths"][uid].split("/")[1]
    f = DIR / "meta" / f"{chunk}.json"
    if not f.exists():
        raw = json.loads(gzip.decompress(_get(f"{HF}/metadata/{chunk}.json.gz", 100 << 20)))
        slim = {u: {"name": m.get("name"), "license": m.get("license"), "faces": m.get("faceCount") or 0,
                    "author": (m.get("user") or {}).get("username"), "url": m.get("viewerUrl")}
                for u, m in raw.items()}
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(slim))
    return json.loads(f.read_text()).get(uid)


def fetch(category: str, variant: int = 0) -> dict:
    """A light, openly licensed model of `category`; `variant` steps to another one."""
    uids = index()["categories"].get(category)
    if not uids:
        raise KeyError(f"no Objaverse category {category!r}")
    # stable shuffle so variant 0 is the same model every time, but not just the first upload
    order = sorted(uids, key=lambda u: hashlib.md5(u.encode()).hexdigest())
    checked = 0
    for uid in order[variant:] + order[:variant]:
        if checked >= 8:
            break
        checked += 1
        meta = _meta(uid) or {}
        if meta.get("license") not in LICENSES or not (0 < meta.get("faces", 0) <= MAX_FACES):
            continue
        dest = LIB / f"{uid}.glb"
        if not (dest.exists() or (store.kind == "blob" and store.read("lib", f"ov-{uid}"))):
            try:
                data = _get(f"{HF}/{index()['paths'][uid]}")
            except ValueError:
                continue  # too large
            if data[:4] != b"glTF":
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            store.publish("lib", [dest])
            if store.kind == "blob":
                store.write("lib", f"ov-{uid}", {"ok": True})
        return {"asset": f"library/objaverse/{uid}.glb", "name": meta.get("name") or category,
                "license": f"{LICENSES[meta['license']]} - {meta.get('author') or 'unknown'} via Sketchfab/Objaverse",
                "source": meta.get("url"), "variant": variant}
    raise LookupError(f"no suitable {category} model found")
