"""Where projects are kept.

Locally a project is the folder data/projects/<id>/ and nothing else.

On Vercel each function instance has its own throwaway disk, so a request
still works in a local folder (under /tmp) and then `publish`es what it wrote
to Vercel Blob, where every instance can read it. Blob sits behind a CDN that
can keep serving an overwritten object for a while, so the JSON documents that
change (meta, status, scene) are never overwritten: every write is a new
object, readers take the newest and the older ones are deleted.

    projects/<id>/<path>                     inputs, views, cutouts, meshes, uploads
    docs/<meta|status|scene>/<id>/<ns>.json  versions of each document
"""
from __future__ import annotations

import json
import mimetypes
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse, Response

DEPLOYED = bool(os.environ.get("VERCEL"))
# Vercel's disk is read-only apart from /tmp
DATA_DIR = (Path("/tmp/room-twin") if DEPLOYED else Path(__file__).resolve().parents[2] / "data") / "projects"
DOCS = ("meta", "status", "scene")


class LocalStore:
    """Projects as folders under DATA_DIR. On Vercel that folder vanishes with the instance."""

    def __init__(self):
        self.kind = "temporary" if DEPLOYED else "local"

    def read(self, pid: str, doc: str) -> dict | None:
        f = DATA_DIR / pid / f"{doc}.json"
        return json.loads(f.read_text()) if f.exists() else None

    def write(self, pid: str, doc: str, data: dict):
        (DATA_DIR / pid / f"{doc}.json").write_text(json.dumps(data, indent=1))

    def publish(self, pid: str, paths: list[Path] | None = None):
        pass  # already where it belongs

    def projects(self) -> list[dict]:
        out = []
        for d in sorted(DATA_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            meta = self.read(d.name, "meta")
            if meta is not None:
                out.append({**meta, "id": d.name, "stage": (self.read(d.name, "status") or {}).get("stage")})
        return out

    def file(self, pid: str, path: str):
        f = (DATA_DIR / pid / path).resolve()
        if not f.is_relative_to(DATA_DIR.resolve() / pid) or not f.is_file():
            raise HTTPException(404)
        return FileResponse(f)

    def delete(self, pid: str):
        shutil.rmtree(DATA_DIR / pid)


class BlobStore:
    """Projects in Vercel Blob, for deployments whose disk doesn't last."""
    kind = "blob"

    def __init__(self):
        from vercel.blob import sync as blob

        self.blob = blob
        # must match the store's access mode; files reach the browser through /files either way
        self.access = os.environ.get("ROOM_TWIN_BLOB_ACCESS", "private")

    def _versions(self, pid: str, doc: str) -> list:
        return sorted(self.blob.query_blobs(prefix=f"docs/{doc}/{pid}/"), key=lambda b: b.pathname)

    def _get(self, url: str) -> bytes:
        return self.blob.get_blob(url, access=self.access).body

    def read(self, pid: str, doc: str) -> dict | None:
        versions = self._versions(pid, doc)
        return json.loads(self._get(versions[-1].url)) if versions else None

    def write(self, pid: str, doc: str, data: dict):
        old = self._versions(pid, doc)
        self.blob.put_blob(f"docs/{doc}/{pid}/{time.time_ns():020d}.json", json.dumps(data).encode(),
                           access=self.access, content_type="application/json")
        if old:
            self.blob.delete_blob([b.url for b in old])

    def publish(self, pid: str, paths: list[Path] | None = None):
        """Upload what this request wrote for `pid`, then drop the local copies.

        Without `paths` that is the whole project folder, documents included.
        """
        root = DATA_DIR / pid
        whole = paths is None
        if whole:
            paths = [p for p in root.rglob("*") if p.is_file() and not (p.parent == root and p.stem in DOCS)]

        def put(p: Path):
            self.blob.put_blob(f"projects/{pid}/{p.relative_to(root).as_posix()}", p.read_bytes(),
                               access=self.access, content_type=mimetypes.guess_type(p.name)[0],
                               allow_overwrite=True)
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(put, paths))

        if whole:
            docs = {d: json.loads(f.read_text()) for d in DOCS if (f := root / f"{d}.json").exists()}
            if "meta" in docs:  # so the project list needs one read per project
                docs["meta"]["stage"] = docs.get("status", {}).get("stage")
            for d in ("status", "scene", "meta"):  # meta last: it makes the project show up
                if d in docs:
                    self.write(pid, d, docs[d])
            shutil.rmtree(root, ignore_errors=True)
        else:
            for p in paths:
                p.unlink(missing_ok=True)

    def projects(self) -> list[dict]:
        newest = {}
        for b in self.blob.query_blobs(prefix="docs/meta/"):
            pid = b.pathname.split("/")[2]
            if pid not in newest or b.pathname > newest[pid].pathname:
                newest[pid] = b
        with ThreadPoolExecutor(8) as ex:
            metas = ex.map(lambda b: json.loads(self._get(b.url)), newest.values())
            out = [{**m, "id": pid} for pid, m in zip(newest, metas)]
        return sorted(out, key=lambda p: p.get("created") or 0, reverse=True)

    def file(self, pid: str, path: str):
        try:
            res = self.blob.get_blob(f"projects/{pid}/{path}", access=self.access)
        except self.blob.BlobNotFoundError:
            raise HTTPException(404)
        # file names are unique and never reused, so the CDN can keep them
        return Response(res.body, media_type=res.metadata.content_type,
                        headers={"Cache-Control": "public, max-age=86400, s-maxage=86400"})

    def delete(self, pid: str):
        # meta first, so a half-finished delete doesn't leave a listed project with missing files
        prefixes = [f"docs/{d}/{pid}/" for d in DOCS] + [f"projects/{pid}/"]
        urls = [b.url for prefix in prefixes for b in self.blob.query_blobs(prefix=prefix)]
        for i in range(0, len(urls), 1000):
            self.blob.delete_blob(urls[i:i + 1000])


# A connected Blob store puts one of these into the project's environment.
store = BlobStore() if os.environ.get("BLOB_STORE_ID") or os.environ.get("BLOB_READ_WRITE_TOKEN") else LocalStore()
