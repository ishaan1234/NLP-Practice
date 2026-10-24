"""Free 3D furniture models (Poly Haven, CC0) for "add a table" style requests.

The catalog (names, categories, tags, real-world sizes) is cached locally for a week.
A model is downloaded (1k-texture glTF + its .bin and textures) the first time it is
used and then served from /library/polyhaven/<id>/ for every project - through the project
store, so a deployment's other instances can serve it too. Only ids in the catalog
can be fetched, so nobody can make the server download arbitrary URLs.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import PurePosixPath

import urllib.parse
import urllib.request

from .storage import DATA_DIR, store

log = logging.getLogger(__name__)

API = "https://api.polyhaven.com"
LIB_DIR = DATA_DIR / "lib"  # store key "lib", served at /library/
CATALOG_FILE = DATA_DIR.parent / "catalog.json"
MAX_AGE = 7 * 24 * 3600
MAX_MODEL_BYTES = 60 * 1024 * 1024
INDOOR = {"furniture", "seating", "table", "shelves", "bed", "lighting", "decorative", "plants",
          "potted plants", "vases", "electronics", "appliances", "wall decoration", "office",
          "containers", "dishes", "instrument"}
OUTDOOR_ONLY = {"rocks", "trees", "ground cover", "grass", "nature", "structures", "buildings", "ships", "food"}

_lock = threading.Lock()
_catalog: list[dict] | None = None


def _get(url: str, params: dict | None = None, limit: int = MAX_MODEL_BYTES) -> bytes:
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "room-twin"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read(limit + 1)
    if len(data) > limit:
        raise ValueError("download too large")
    return data


def _json(url: str, params: dict | None = None):
    return json.loads(_get(url, params))


def catalog() -> list[dict]:
    """[{id, name, categories, tags, size: [x, y, z] metres}], cached on disk."""
    global _catalog
    with _lock:
        if _catalog is not None:
            return _catalog
        if CATALOG_FILE.exists() and time.time() - CATALOG_FILE.stat().st_mtime < MAX_AGE:
            _catalog = json.loads(CATALOG_FILE.read_text())
            return _catalog
        try:
            assets = _json(f"{API}/assets", {"t": "models"})
            items = []
            for aid, a in assets.items():
                cats = set(a.get("categories", []))
                if not cats & INDOOR or (cats & OUTDOOR_ONLY and not cats & {"furniture", "decorative", "lighting"}):
                    continue
                dims = [round(v / 1000, 2) for v in (a.get("dimensions") or [])][:3]
                items.append({"id": aid, "name": a.get("name", aid),
                              "categories": sorted(c for c in cats if not c.startswith("collection")),
                              "tags": a.get("tags", [])[:8], "size": dims})
            items.sort(key=lambda i: i["id"])
            CATALOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            CATALOG_FILE.write_text(json.dumps(items))
            _catalog = items
        except Exception as e:  # offline: keep whatever we had, else empty (procedural pieces still work)
            log.warning("model catalog unavailable: %s", e)
            _catalog = json.loads(CATALOG_FILE.read_text()) if CATALOG_FILE.exists() else []
        return _catalog


def search(query: str, limit: int = 24) -> list[dict]:
    words = [w for w in query.lower().split() if w]
    scored = []
    for item in catalog():
        hay = " ".join([item["id"], item["name"], *item["categories"], *item["tags"]]).lower()
        score = sum(2 if w in item["name"].lower() else 1 for w in words if w in hay)
        if score or not words:
            scored.append((score, item))
    scored.sort(key=lambda s: (-s[0], s[1]["id"]))
    return [dict(i, thumb=f"https://cdn.polyhaven.com/asset_img/thumbs/{i['id']}.png?width=256&height=256")
            for _, i in scored[:limit]]


def fetch(model_id: str) -> dict:
    """Download a catalog model once; returns {asset: url, name, size}."""
    item = next((i for i in catalog() if i["id"] == model_id), None)
    if item is None:
        raise KeyError(f"unknown model {model_id!r}")
    doc = f"ph-{model_id}"
    known = store.read("lib", doc)
    if known:
        return known
    dest = LIB_DIR / "polyhaven" / model_id
    with _lock:
        files = _json(f"{API}/files/{model_id}")
        gltf = files["gltf"]["1k"]["gltf"]
        dest.mkdir(parents=True, exist_ok=True)
        total, written = 0, []
        for rel, meta in [(PurePosixPath(gltf["url"]).name, gltf), *gltf.get("include", {}).items()]:
            path = (dest / rel).resolve()
            if not path.is_relative_to(dest.resolve()):
                raise ValueError("bad path in model files")
            data = _get(meta["url"])
            total += len(data)
            if total > MAX_MODEL_BYTES:
                raise ValueError("model too large")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            written.append(path)
        info = {"asset": f"library/polyhaven/{model_id}/{PurePosixPath(gltf['url']).name}", "name": item["name"],
                "size": item["size"], "license": "CC0 (Poly Haven)"}
        store.publish("lib", written)
        store.write("lib", doc, info)
        return info
