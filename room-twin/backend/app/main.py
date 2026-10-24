"""Room Twin API + static frontend.

Run:  uvicorn app.main:app --app-dir backend --reload   (from room-twin/)
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import Models, make_item_from_photo, project_dir, run_pipeline, write_status
from .storage import DATA_DIR, DEPLOYED, store

logging.basicConfig(level=logging.INFO)
FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
ALLOWED = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PID_RE = re.compile(r"^[a-f0-9]{12}$")
# Vercel turns away request bodies over 4.5 MB; the frontend shrinks photos to fit
MAX_UPLOAD = 4_400_000 if DEPLOYED else None

app = FastAPI(title="Room Twin")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _owner(key: str | None) -> str | None:
    """The browser's random key (X-Owner header), hashed before it is stored."""
    return hashlib.sha256(key.encode()).hexdigest()[:24] if key else None


def _meta(pid: str) -> dict:
    meta = store.read(pid, "meta") if PID_RE.match(pid) else None
    if meta is None:
        raise HTTPException(404, "unknown project")
    return meta


def _owned(pid: str, key: str | None) -> dict:
    """On a public deployment anyone with the link can view a room, but only its maker can change it."""
    meta = _meta(pid)
    if DEPLOYED and meta.get("owner") not in (None, _owner(key)):
        raise HTTPException(403, "this room belongs to someone else")
    return meta


def _save_upload(f: UploadFile, dest_dir: Path, stem: str) -> Path:
    ext = Path(f.filename or "").suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"unsupported file type: {f.filename}")
    dest = dest_dir / f"{stem}{ext}"
    with open(dest, "wb") as out:
        shutil.copyfileobj(f.file, out)
    return dest


@app.get("/api/capabilities")
def capabilities():
    return {**Models.info(), "storage": store.kind, "deployed": DEPLOYED, "maxUpload": MAX_UPLOAD}


@app.post("/api/projects")
def create_project(
    files: list[UploadFile] = File(...),
    name: str = Form("My room"),
    mode: str = Form("auto"),  # auto | 360 | multiview
    yaws: str = Form(""),  # JSON list of degrees, one per photo
    pitches: str = Form(""),  # JSON list of degrees or null (auto-detect), one per photo
    hfov: float | None = Form(None),
    camera_height: float = Form(1.5),
    prompts: str = Form(""),  # comma separated
    room_width: float = Form(5.0),
    room_depth: float = Form(4.0),
    room_height: float = Form(2.7),
    reconstruct: bool = Form(True),
    x_owner: str | None = Header(None),
):
    if mode not in {"auto", "360", "multiview"}:
        raise HTTPException(400, "mode must be auto, 360 or multiview")
    pid = uuid.uuid4().hex[:12]
    pdir = project_dir(pid)
    (pdir / "inputs").mkdir(parents=True)
    paths = [_save_upload(f, pdir / "inputs", f"img_{i:02d}") for i, f in enumerate(files)]
    if not paths:
        raise HTTPException(400, "no images")
    try:
        yaw_list = [float(y) for y in json.loads(yaws)] if yaws.strip() else None
    except (ValueError, TypeError):
        raise HTTPException(400, "yaws must be a JSON list of numbers")
    try:
        pitch_list = [None if v is None else float(v) for v in json.loads(pitches)] if pitches.strip() else None
    except (ValueError, TypeError):
        raise HTTPException(400, "pitches must be a JSON list of numbers or nulls")
    prompt_list = [p.strip() for p in prompts.split(",") if p.strip()] or None
    write_status(pid, "queued", 0.0)
    (pdir / "meta.json").write_text(json.dumps({"name": name, "created": time.time(), "owner": _owner(x_owner)}))
    args = (pid, paths, mode, yaw_list, hfov, camera_height, prompt_list,
            (room_width, room_depth, room_height), reconstruct, name, pitch_list)
    if DEPLOYED:  # a function can be frozen once it has answered: build now, then store the result
        run_pipeline(*args)
        store.publish(pid)
    else:
        threading.Thread(target=run_pipeline, daemon=True, args=args).start()
    return {"id": pid}


@app.post("/api/projects/blank")
def create_blank(name: str = Form("Blank room"), room_width: float = Form(5.0),
                 room_depth: float = Form(4.0), room_height: float = Form(2.7),
                 x_owner: str | None = Header(None)):
    """An empty room to furnish from scratch, no photos needed."""
    pid = uuid.uuid4().hex[:12]
    pdir = project_dir(pid)
    pdir.mkdir(parents=True)
    now = time.time()
    scene = {
        "version": 1, "id": pid, "name": name, "created": now,
        "room": {"width": room_width, "depth": room_depth, "height": room_height,
                 "floorColor": "#b08d6a", "wallColor": "#e9e4da", "cameraHeight": 1.5},
        "panorama": None, "views": [], "objects": [], "models": {},
    }
    (pdir / "scene.json").write_text(json.dumps(scene, indent=1))
    (pdir / "meta.json").write_text(json.dumps({"name": name, "created": now, "owner": _owner(x_owner)}))
    write_status(pid, "done", 1.0, objects=0)
    store.publish(pid)
    return {"id": pid}


@app.get("/api/projects")
def list_projects(x_owner: str | None = Header(None)):
    me = _owner(x_owner)
    return [{"id": p["id"], "name": p.get("name"), "created": p.get("created"), "stage": p.get("stage")}
            for p in store.projects() if not DEPLOYED or p.get("owner") in (None, me)]


@app.get("/api/projects/{pid}/status")
def status(pid: str):
    s = store.read(pid, "status") if PID_RE.match(pid) else None
    if s is None:
        raise HTTPException(404, "unknown project")
    return s


@app.get("/api/projects/{pid}/scene")
def get_scene(pid: str):
    scene = store.read(pid, "scene") if PID_RE.match(pid) else None
    if scene is None:
        _meta(pid)
        raise HTTPException(409, "scene not ready")
    return JSONResponse(scene, headers={"Cache-Control": "no-store"})


@app.put("/api/projects/{pid}/scene")
def put_scene(pid: str, scene: dict, x_owner: str | None = Header(None)):
    meta = _owned(pid, x_owner)
    scene["id"] = pid
    store.write(pid, "scene", scene)
    if "name" in scene and scene["name"] != meta.get("name"):
        store.write(pid, "meta", {**meta, "name": scene["name"]})
    return {"ok": True}


@app.post("/api/projects/{pid}/items")
def add_item(pid: str, file: UploadFile = File(...), prompt: str = Form(""),
             x_owner: str | None = Header(None)):
    _owned(pid, x_owner)
    d = project_dir(pid)
    (d / "uploads").mkdir(parents=True, exist_ok=True)
    path = _save_upload(file, d / "uploads", uuid.uuid4().hex[:8])
    item = make_item_from_photo(pid, path, prompt.strip() or None)
    made = [DATA_DIR / item[k].removeprefix("files/") for k in ("cutout", "asset") if item.get(k)]
    store.publish(pid, [path, *made])
    return item


@app.post("/api/projects/{pid}/assets")
def upload_asset(pid: str, file: UploadFile = File(...), x_owner: str | None = Header(None)):
    """Raw .glb/.gltf model the user drops into the scene."""
    _owned(pid, x_owner)
    ext = Path(file.filename or "").suffix.lower()
    if ext not in {".glb", ".ply"}:
        raise HTTPException(400, "only .glb or .ply")
    d = project_dir(pid) / "assets"
    d.mkdir(parents=True, exist_ok=True)
    dest = d / f"{uuid.uuid4().hex[:8]}{ext}"
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)
    store.publish(pid, [dest])
    return {"asset": f"files/{pid}/assets/{dest.name}", "kind": "mesh" if ext == ".glb" else "pointcloud"}


@app.delete("/api/projects/{pid}")
def delete_project(pid: str, x_owner: str | None = Header(None)):
    _owned(pid, x_owner)
    store.delete(pid)
    return {"ok": True}


@app.get("/files/{pid}/{path:path}")
def project_file(pid: str, path: str):
    if not PID_RE.match(pid) or any(s in ("", ".", "..") for s in path.split("/")):
        raise HTTPException(404)
    return store.file(pid, path)


app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
