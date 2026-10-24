"""Room Twin API + static frontend.

Run:  uvicorn app.main:app --app-dir backend --reload   (from room-twin/)
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import DATA_DIR, Models, make_item_from_photo, project_dir, run_pipeline, write_status

logging.basicConfig(level=logging.INFO)
FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
ALLOWED = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PID_RE = re.compile(r"^[a-f0-9]{12}$")

app = FastAPI(title="Room Twin")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _pdir(pid: str) -> Path:
    if not PID_RE.match(pid):
        raise HTTPException(404, "unknown project")
    d = project_dir(pid)
    if not d.exists():
        raise HTTPException(404, "unknown project")
    return d


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
    return Models.info()


@app.post("/api/projects")
async def create_project(
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
    (pdir / "meta.json").write_text(json.dumps({"name": name, "created": time.time()}))
    threading.Thread(
        target=run_pipeline, daemon=True,
        args=(pid, paths, mode, yaw_list, hfov, camera_height, prompt_list,
              (room_width, room_depth, room_height), reconstruct, name, pitch_list),
    ).start()
    return {"id": pid}


@app.post("/api/projects/blank")
def create_blank(name: str = Form("Blank room"), room_width: float = Form(5.0),
                 room_depth: float = Form(4.0), room_height: float = Form(2.7)):
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
    (pdir / "meta.json").write_text(json.dumps({"name": name, "created": now}))
    write_status(pid, "done", 1.0, objects=0)
    return {"id": pid}


@app.get("/api/projects")
def list_projects():
    out = []
    for d in sorted(DATA_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not (d / "meta.json").exists():
            continue
        meta = json.loads((d / "meta.json").read_text())
        status = json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else {}
        out.append({"id": d.name, "name": meta.get("name"), "created": meta.get("created"),
                    "stage": status.get("stage")})
    return out


@app.get("/api/projects/{pid}/status")
def status(pid: str):
    return json.loads((_pdir(pid) / "status.json").read_text())


@app.get("/api/projects/{pid}/scene")
def get_scene(pid: str):
    f = _pdir(pid) / "scene.json"
    if not f.exists():
        raise HTTPException(409, "scene not ready")
    return FileResponse(f, headers={"Cache-Control": "no-store"})


@app.put("/api/projects/{pid}/scene")
def put_scene(pid: str, scene: dict):
    d = _pdir(pid)
    scene["id"] = pid
    (d / "scene.json").write_text(json.dumps(scene, indent=1))
    if "name" in scene:
        meta = json.loads((d / "meta.json").read_text())
        meta["name"] = scene["name"]
        (d / "meta.json").write_text(json.dumps(meta))
    return {"ok": True}


@app.post("/api/projects/{pid}/items")
async def add_item(pid: str, file: UploadFile = File(...), prompt: str = Form("")):
    d = _pdir(pid)
    (d / "uploads").mkdir(exist_ok=True)
    path = _save_upload(file, d / "uploads", uuid.uuid4().hex[:8])
    return await run_in_threadpool(make_item_from_photo, pid, path, prompt.strip() or None)


@app.post("/api/projects/{pid}/assets")
async def upload_asset(pid: str, file: UploadFile = File(...)):
    """Raw .glb/.gltf model the user drops into the scene."""
    d = _pdir(pid)
    ext = Path(file.filename or "").suffix.lower()
    if ext not in {".glb", ".ply"}:
        raise HTTPException(400, "only .glb or .ply")
    (d / "assets").mkdir(exist_ok=True)
    dest = d / "assets" / f"{uuid.uuid4().hex[:8]}{ext}"
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)
    return {"asset": f"files/{pid}/assets/{dest.name}", "kind": "mesh" if ext == ".glb" else "pointcloud"}


@app.delete("/api/projects/{pid}")
def delete_project(pid: str):
    shutil.rmtree(_pdir(pid))
    return {"ok": True}


app.mount("/files", StaticFiles(directory=DATA_DIR), name="files")
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
