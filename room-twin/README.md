# Room Twin

Turn photos of a room (several angles, or a single 360° equirectangular image) into an
editable 3D "digital twin": the objects in the room are found, cut out, lifted to 3D and
placed at their real-world positions, and you can then move them around, delete them and
drop in new furniture.

```
photos / 360 image ──► views (360 → 8 perspective crops)
                         │
                         ▼
             SAM 3: text-prompted instance masks ("chair", "sofa", "lamp", …)
                         │
                         ▼
     mask → floor-contact ray-cast → metric position + size in the room
                         │
                         ▼
     SAM 3D Objects: (image, mask) → mesh / Gaussian splat per object
                         │
                         ▼
               scene.json  ──►  Three.js editor (move, rotate, scale, add, export GLB)
```

## Quick start

```bash
cd room-twin
pip install -r backend/requirements.txt
uvicorn app.main:app --app-dir backend --port 8000
# open http://localhost:8000
```

This runs with **no GPU and no models**: segmentation falls back to a colour-region heuristic
and objects appear as photo cut-out cards. The header shows which backends a scene was built with
(`segmenter: sam3 · 3D: sam3d-objects` vs `fallback · photo cutouts`).

### Segmentation backends (picked automatically, best first)

| Backend | Needs | Notes |
|---|---|---|
| **SAM 3** | GPU, gated checkpoint | best masks, text-prompted |
| **Grounding DINO + SAM** | `pip install torch transformers` | ungated, runs on CPU (~15–40 s per image) |
| colour-region fallback | nothing | demo only, labels everything "object", can't measure walls |

Force one with `ROOM_TWIN_SEGMENTER=sam3|grounded-sam|fallback`.

### Enabling SAM 3 and SAM 3D Objects (GPU)

Both models are gated on Hugging Face, so request access and run `huggingface-cli login` first.

```bash
# SAM 3 – segmentation
git clone https://github.com/facebookresearch/sam3 && pip install -e sam3

# SAM 3D Objects – single-image 3D reconstruction
git clone https://github.com/facebookresearch/sam-3d-objects
# follow its setup doc to install deps and download checkpoints into sam-3d-objects/checkpoints/hf
export SAM3D_REPO=$PWD/sam-3d-objects          # where notebook/inference.py lives
# optional: export SAM3D_CONFIG=/path/to/pipeline.yaml
```

Models load lazily on the first build. If either fails to import, the server logs why and
keeps going with the fallback.

## Deploy on Vercel

The editor and the photo → twin flow run on Vercel's Python runtime. The GPU models don't
(torch alone is bigger than the 500 MB function limit), so a deployed twin uses the fallback
segmenter and photo cut-outs. For SAM 3 quality, run the backend on a GPU machine as above.

1. Import the repo in Vercel and set **Root Directory** to `room-twin` (or run `vercel` from
   `room-twin/`). `pyproject.toml` says what to install and where the app is
   (`backend.app.main:app`); the frontend is served from Vercel's CDN.
2. Under **Storage**, create a **Blob** store (private), connect it to the project and redeploy.
   Without one, rooms only live on a function's temporary disk and soon disappear; the app
   shows a warning when that's the case. For a public store, set `ROOM_TWIN_BLOB_ACCESS=public`.

What's different when deployed:

- A build runs inside the upload request (Vercel allows up to 300 s) rather than in a
  background thread, so the progress bar only says "building…".
- Requests are capped at 4.5 MB. The browser shrinks photos to the size the pipeline works at
  anyway (1600 px wide, 4096 for a 360° image) and keeps the camera's lens info; several photos
  together still have to fit. `.glb` imports over 4.4 MB are refused.
- Each browser sees and edits only its own rooms (a random key kept in `localStorage`).
  Anyone with a room's link can view it.

## Using it

1. **New twin** – drop photos or one 360° image. Mode `auto` treats a ~2:1 image as a panorama.
   For multiple photos, pick which way each one faces compared to the first (ahead / right /
   behind / left …; the arrow under each thumbnail shows it). The default spreads them evenly. Stand in one spot, roughly in the middle of the room, and turn.
2. Optionally edit the **SAM 3 prompts** (comma-separated nouns) to target what's in your room.
   No photos? **Start a blank room** and furnish it from the library.
3. **Edit** – click an object to select it (outline + live size label), drag it across the floor.
   Objects stay inside the room and snap flush to a wall when dragged near one (hold `Alt` to
   skip the snap). The hint bar at the bottom always shows what you can do next; `?` lists every
   shortcut.
   - Drag: move · `Shift`+drag: rotate · arrows: nudge 5 cm (`Shift` 50 cm) · `[ ]`: rotate 15° (`Shift` 90°)
   - `W` / `E` / `R` gizmo modes, `F` focus, `H` hide, `L` lock, `Ctrl+D` duplicate, `Del` delete
   - `Ctrl+Z` / `Ctrl+Shift+Z` undo / redo
   - `T` top view, `P` perspective, `V` eye view (stand where the photo was taken)
   - **Photo walls** projects the 360° image onto the room box from the capture point
   - The **Objects** list has thumbnails plus hide 👁 / lock 🔒 toggles; lock the pieces you're happy with
4. **Add items** – click a library piece (it lands on a free patch of floor) or drag it to an exact
   spot. Drop or paste (`Ctrl+V`) any product photo onto the 3D view to cut it out and build it
   in 3D, or use **Import .glb…** for an existing model.
5. **Save** (it also autosaves), **Export GLB** for Blender / Unity, **Screenshot** for a PNG.

## How it works

The capture camera is at `(0, camera_height, 0)` with the floor at `y = 0`.

1. **Camera per photo** – field of view from the photo's camera info (phones save it) or the
   *Lens* setting. **Tilt is detected automatically**: vertical edges (door frames, corners, legs)
   lean toward a common point when the camera is tilted, and that lean gives the angle
   (`estimate_pitch` in `geometry.py`; median error 0.9° on test views at known tilts).
   Override per photo if needed.
2. **Segment** furniture *and the floor* in every view.
3. **Room layout** (`layout.py`) – where the floor mask ends in each image column is where a
   wall starts. Those points, projected onto the floor, outline the room: its rotation is
   the angle that lines them up best, and each wall sits where its points pile up. Walls never
   seen (e.g. behind a single photo) default to 0.6 m behind the camera.
4. **Place objects** – the lowest pixels of a mask touch the floor, which pins distance;
   width and height follow from the mask at that distance. Depth uses a per-type typical size
   (a sofa is ~0.9 m deep, a frame is flat), and sizes far from typical lower the
   confidence. Objects whose bottom is above eye level go onto the fitted walls. Everything is
   turned to line up with the walls.
5. **Merge** the same object seen from several views, preferring views where it isn't cut off.

## Known limitations

- Multi-photo mode assumes all shots are taken from one spot; the direction per photo comes
  from you, not from feature matching. Plugging in a pose estimator (e.g. VGGT / DUSt3R /
  COLMAP) in `build_views()` would lift this.
- Photos without camera info get a normal phone lens assumed; that only stretches depth, so
  pick *Wide* for real-estate style shots.
- Camera roll (a sideways-tilted horizon) isn't corrected.
- SAM 3D Objects output orientation can vary; use **Re-orient model** on a selected object.
  Gaussian splats are shown as coloured point clouds.
- The fallback segmenter is only a demo of the flow; it labels everything "object".

## Layout

```
room-twin/
  backend/app/
    main.py            FastAPI routes + serves the frontend and project files
    pipeline.py        upload → views → segment → place → reconstruct → scene.json
    geometry.py        camera model, 360→perspective resampling, mask → 3D placement
    segmenters.py      SAM 3 adapter + fallback segmenter
    reconstructors.py  SAM 3D Objects adapter + splat→point-cloud conversion
    storage.py         project files on local disk, or in Vercel Blob when deployed
  frontend/
    index.html, style.css, app.js   Three.js editor
    furniture.js                    procedural furniture library
  data/projects/<id>/  uploads, views, cutouts, meshes, scene.json (gitignored)
  pyproject.toml       Vercel's dependencies + entrypoint
```
