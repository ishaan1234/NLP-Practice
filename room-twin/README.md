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

## Easiest: the launcher (Windows, macOS, Linux)

```
git clone https://github.com/ishaan1234/roomroom
cd roomroom/room-twin
RoomTwin.bat                 # Windows: or double-click it.  macOS/Linux: python3 launch.py
```

It sets everything up, asks once for a **fal.ai key** (real 3D models + SAM 3, see below) and
optionally an **Anthropic key** (the AI assistant), then starts the app and opens your browser.
Keys are kept in `room-twin/.roomtwin.env` (not committed). Needs Python 3.10+.

- `RoomTwin.bat status` - what's configured
- `RoomTwin.bat --colab` - use your own Colab GPU instead of fal.ai (see `gpu-worker/`)

## Real 3D models: fal.ai

Turning photos into real 3D models needs SAM 3D Objects on a big GPU. The app uses Meta's models
hosted on **[fal.ai](https://fal.ai)**, so there is nothing to install and no GPU or gated
download needed:

| fal.ai endpoint | used for | price (Oct 2026) |
|---|---|---|
| `fal-ai/sam-3/3d-objects` | SAM 3D Objects: each detected object -> textured 3D mesh | $0.02 per object |
| `fal-ai/sam-3/image` | SAM 3: find the furniture, floor, doors, windows | $0.005 per prompt per image |

Create a key at fal.ai/dashboard/keys (add some credit) and set `FAL_KEY` - the launcher asks for
it; on Vercel add it under Environment Variables. A room appears right away with photo cut-outs,
then each object turns into a 3D model one by one (~$0.02 each). Segmenting with SAM 3 on fal costs
about $0.12 per photo (23 prompts), so ~$0.90 for a 360 photo (8 views); to segment for free on
your own CPU instead, `pip install torch transformers` and set `ROOM_TWIN_SEGMENTER=grounded-sam`.

## Quick start (manual)

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
| **Grounding DINO + SAM** | `pip install torch transformers` | ungated, runs on CPU (~15 s per view on 4 cores; a 360° photo is 8 views); first run downloads ~1 GB of weights |
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

## Alternative: your own GPU (Colab or any box)

Instead of fal.ai you can run SAM 3 + SAM 3D Objects yourself on a Linux machine with a >= 32 GB
NVIDIA GPU, e.g. a Colab A100, with [`gpu-worker/`](gpu-worker/README.md) and set
`ROOM_TWIN_GPU_URL` / `ROOM_TWIN_GPU_TOKEN` (`RoomTwin.bat --colab` does it for you). Needs
Hugging Face access to the gated `facebook/sam3` and `facebook/sam-3d-objects` models.

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
- **Phone capture works there too** (and needs no `--host`/Wi-Fi setup: the QR code points at
  the deployment's HTTPS address). Capture sessions and photos go through the same store as
  projects, so the phone and the desktop can land on different function instances; the phone
  page shrinks very large photos to fit the request cap, keeping the lens info.

## Using it

### Take the photos with your phone

Press **📱 Use phone camera** and scan the QR code. On the phone, pick **Room photos** or
**An item**, then take or choose photos; they appear on the computer within a couple of seconds
(room photos in the upload list, items straight into the open room).

The phone has to reach the computer: same Wi-Fi, and start the server so it listens on the
network:

```bash
uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000
```

If the QR shows the wrong address (VPNs, several network cards), set `PUBLIC_URL=http://<ip>:8000`.
The link's secret sits after `#`, so it never reaches server logs; a session lasts an hour, and
**Stop** ends it immediately.

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
6. **Layouts** – *Import layout* / *Export layout* (Room panel) read and write the
   `roomshift-layout` JSON used by RoomShift and its iPhone LiDAR (RoomPlan) exporter:
   exact room size, doors, windows and furniture boxes. Doors and windows are also in the
   library.

### LiDAR scan + photos (Apple RoomPlan)

On an iPhone/iPad **Pro** (LiDAR), Apple's RoomPlan measures walls, doors, windows and
furniture boxes to within a few centimetres – but only as grey boxes. RoomShift's
`native/RoomShiftRoomPlan` Swift package (build it into an iOS app with Xcode) exports that scan
as `roomshift-layout.json`. Importing it into a **photo twin** merges the two
(`frontend/scanmerge.js`):

- the scan is straightened so its walls run along the axes (RoomPlan's frame starts wherever the
  phone pointed),
- the photo twin is matched onto it: 4 quarter turns × scale × offsets proposed by pairs of
  same-type objects, then a least-squares refinement,
- matched photo objects snap onto their scanned boxes (exact position and size), scanned pieces
  without a photo come in as boxes, and the room takes the scan's exact size, doors and windows,
- the scale also **calibrates the camera height**, so the 360° projection stays lined up.

On a synthetic test (scan rotated 37°, quarter turn, scale 1.12, ±8 cm noise) it recovers the
rotation exactly, the scale to 1.12 and places objects within the noise (6 cm median).

### Uploads

Every uploaded photo is decoded, turned upright, size-limited (15 MB, 40 MP; 2560 px, or 6144 px
for 360° images) and re-saved as JPEG with its metadata (GPS, device info) removed – except the
lens focal length, which is needed for distances.


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

## Credits

Phone pairing via QR, upload hardening, the layout JSON format and the SAM 3D output handling
follow ideas from [RoomShift](https://github.com/tysuprawee/qr).

## Layout

```
room-twin/
  backend/app/
    main.py            FastAPI routes + serves the frontend and project files
    capture.py         QR phone capture sessions + upload sanitising
    layout.py          room outline fitting, furniture size priors
    pipeline.py        upload → views → segment → place → reconstruct → scene.json
    geometry.py        camera model, 360→perspective resampling, mask → 3D placement
    segmenters.py      SAM 3 adapter + fallback segmenter
    reconstructors.py  SAM 3D Objects adapter + splat→point-cloud conversion
    storage.py         project files on local disk, or in Vercel Blob when deployed
  frontend/
    index.html, style.css, app.js   Three.js editor
    capture.html                    phone capture page
    furniture.js                    procedural furniture library
  data/projects/<id>/  uploads, views, cutouts, meshes, scene.json (gitignored)
  pyproject.toml       Vercel's dependencies + entrypoint
```
