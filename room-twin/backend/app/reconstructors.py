"""Single-object 3D reconstruction.

`Sam3DReconstructor` runs Meta's SAM 3D Objects on (image, mask) and writes
either a textured mesh (.glb) or, if only Gaussian splats come back, a
coloured point cloud (.ply) the browser can display. Without the model the
pipeline keeps the RGBA cutout only and the editor shows it as a photo card.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)
SH_C0 = 0.28209479177387814


class Sam3DReconstructor:
    name = "sam3d-objects"

    def __init__(self):
        # The SAM 3D Objects repo ships its inference wrapper in notebook/inference.py.
        repo = Path(os.environ.get("SAM3D_REPO", "sam-3d-objects")).resolve()
        sys.path.insert(0, str(repo))  # its notebook helpers import from the repo root too
        sys.path.insert(0, str(repo / "notebook"))
        from inference import Inference  # type: ignore

        cfg = os.environ.get("SAM3D_CONFIG", str(repo / "checkpoints/hf/pipeline.yaml"))
        self.model = Inference(cfg, compile=False)

    def reconstruct(self, image: Image.Image, mask: np.ndarray, out_stem: Path) -> tuple[str, Path] | None:
        out = self.model(np.asarray(image.convert("RGB")), mask.astype(bool), seed=42)
        # output["glb"] is a trimesh already converted to Y-up (see RoomShift's worker), so it
        # drops straight into the editor; its scale is a guess and gets fitted to our box.
        mesh = out.get("glb") if isinstance(out, dict) else None
        if mesh is not None and hasattr(mesh, "export"):
            path = out_stem.with_suffix(".glb")
            mesh.export(path)
            return "mesh", path
        gs = out.get("gs") if isinstance(out, dict) else None
        if gs is not None:
            raw = out_stem.with_name(out_stem.name + "_splat.ply")
            gs.save_ply(str(raw))
            path = out_stem.with_suffix(".ply")
            splat_ply_to_points(raw, path)
            return "pointcloud", path
        return None


class RemoteReconstructor:
    """SAM 3D Objects on a remote GPU worker (gpu-worker/, e.g. a Colab A100)."""
    name = "sam3d-objects (remote GPU)"

    def __init__(self):
        from . import gpu_client
        self.client = gpu_client
        if not gpu_client.health().get("sam3d"):
            raise RuntimeError("the GPU worker has no SAM 3D Objects")

    def reconstruct(self, image: Image.Image, mask: np.ndarray, out_stem: Path) -> tuple[str, Path] | None:
        import io
        buf = io.BytesIO()
        Image.fromarray(mask.astype(np.uint8) * 255).save(buf, "PNG")
        kind, data = self.client.reconstruct(image.convert("RGB"), buf.getvalue())
        path = out_stem.with_suffix(".glb" if kind == "mesh" else ".ply")
        path.write_bytes(data)
        return kind, path


def load_reconstructor():
    from . import gpu_client

    if gpu_client.configured():
        try:
            rec = RemoteReconstructor()
            log.info("SAM 3D Objects on the remote GPU worker")
            return rec
        except Exception as e:
            log.warning("remote GPU worker unavailable for 3D (%s)", e)
    try:
        rec = Sam3DReconstructor()
        log.info("SAM 3D Objects loaded")
        return rec
    except Exception as e:
        log.warning("SAM 3D Objects unavailable (%s); objects will be photo cutouts", e)
        return None


# --- Gaussian-splat PLY -> plain coloured point cloud --------------------------------

_PLY_TYPES = {"float": "f4", "float32": "f4", "double": "f8", "uchar": "u1", "uint8": "u1",
              "int": "i4", "int32": "i4", "uint": "u4", "short": "i2", "ushort": "u2"}


def read_binary_ply(path: Path) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        props, n = [], 0
        while True:
            line = f.readline().decode("ascii", "ignore").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property") and "list" not in line:
                _, t, name = line.split()
                props.append((name, "<" + _PLY_TYPES[t]))
            elif line == "end_header":
                break
        data = np.frombuffer(f.read(n * np.dtype(props).itemsize), dtype=props, count=n)
    return {name: data[name] for name, _ in props}


def splat_ply_to_points(src: Path, dst: Path, max_points: int = 150_000) -> None:
    v = read_binary_ply(src)
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype("<f4")
    if "f_dc_0" in v:
        rgb = 0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1)
    else:
        rgb = np.stack([v.get("red", 200), v.get("green", 200), v.get("blue", 200)], 1) / 255.0
    keep = np.ones(len(xyz), bool)
    if "opacity" in v:
        keep = 1 / (1 + np.exp(-v["opacity"])) > 0.3
    xyz, rgb = xyz[keep], rgb[keep]
    if len(xyz) > max_points:
        idx = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    rgb8 = (np.clip(rgb, 0, 1) * 255).astype("u1")
    rec = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                    ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz.T
    rec["red"], rec["green"], rec["blue"] = rgb8.T
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(rec)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(dst, "wb") as f:
        f.write(header.encode())
        f.write(rec.tobytes())
