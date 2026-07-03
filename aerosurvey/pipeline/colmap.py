"""COLMAP orchestration: run real Structure-from-Motion and parse its output.

This wraps the COLMAP command-line pipeline
    feature_extractor -> exhaustive_matcher -> mapper -> model_converter
and reads the resulting sparse reconstruction (cameras / images / points3D)
into plain numpy structures the rest of the app understands.

If COLMAP is not installed (`available()` is False) the caller falls back to the
built-in simulation. Nothing here depends on the rest of the pipeline, so the
parsers are unit-testable without COLMAP present.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


def exe() -> str:
    return shutil.which("colmap") or shutil.which("colmap.exe") or ""


def available() -> bool:
    return bool(exe())


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def qvec2rotmat(q) -> np.ndarray:
    """COLMAP quaternion (w, x, y, z) -> 3x3 rotation (world->camera)."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def camera_center(qvec, tvec) -> np.ndarray:
    """World-space camera position C = -R^T t (COLMAP projects X_cam = R X + t)."""
    R = qvec2rotmat(qvec)
    return (-R.T @ np.asarray(tvec, dtype=np.float64))


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class CameraPose:
    name: str
    qvec: np.ndarray
    tvec: np.ndarray
    center: np.ndarray
    camera_id: int = 0


@dataclass
class ColmapResult:
    poses: Dict[str, CameraPose] = field(default_factory=dict)     # basename -> pose
    intrinsics: Dict[int, np.ndarray] = field(default_factory=dict)  # camera_id -> 3x3 K
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    colors: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.uint8))
    mean_reproj_error: float = 0.0
    model_dir: str = ""   # sparse model folder (for downstream OpenMVS)
    image_dir: str = ""   # staged image folder

    def projection(self, pose: "CameraPose"):
        """3x4 projection matrix for a pose, or None if intrinsics are unknown."""
        K = self.intrinsics.get(pose.camera_id)
        if K is None:
            return None
        return K @ np.hstack([qvec2rotmat(pose.qvec), np.asarray(pose.tvec).reshape(3, 1)])


# ---------------------------------------------------------------------------
# Text-model parsers (COLMAP sparse model exported with --output_type TXT)
# ---------------------------------------------------------------------------
def parse_images_txt(path: str) -> Dict[str, CameraPose]:
    poses: Dict[str, CameraPose] = {}
    with open(path, "r", encoding="utf-8") as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    # Records are two lines each: pose line, then a POINTS2D line we skip.
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        i += 2  # skip the points2D line
        if len(parts) < 10:
            continue
        qvec = np.array(list(map(float, parts[1:5])))
        tvec = np.array(list(map(float, parts[5:8])))
        camera_id = int(parts[8])
        name = os.path.basename(parts[9])
        poses[name] = CameraPose(name, qvec, tvec, camera_center(qvec, tvec), camera_id)
    return poses


def _k_from_model(model: str, params) -> np.ndarray:
    p = [float(v) for v in params]
    two_focal = {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV",
                 "THIN_PRISM_FISHEYE"}
    if model in two_focal and len(p) >= 4:
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    # SIMPLE_PINHOLE / SIMPLE_RADIAL / RADIAL / *_FISHEYE: f, cx, cy, [dist...]
    f = p[0]
    cx = p[1] if len(p) > 1 else 0.0
    cy = p[2] if len(p) > 2 else 0.0
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)


def parse_cameras_txt(path: str) -> Dict[int, np.ndarray]:
    intr: Dict[int, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            if ln.startswith("#") or not ln.strip():
                continue
            p = ln.split()
            if len(p) < 5:
                continue
            intr[int(p[0])] = _k_from_model(p[1], p[4:])
    return intr


def parse_points3d_txt(path: str):
    xyz: List[List[float]] = []
    rgb: List[List[int]] = []
    errs: List[float] = []
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            if ln.startswith("#") or not ln.strip():
                continue
            p = ln.split()
            if len(p) < 8:
                continue
            xyz.append([float(p[1]), float(p[2]), float(p[3])])
            rgb.append([int(p[4]), int(p[5]), int(p[6])])
            errs.append(float(p[7]))
    pts = np.array(xyz, dtype=np.float64) if xyz else np.zeros((0, 3))
    cols = np.array(rgb, dtype=np.uint8) if rgb else np.zeros((0, 3), np.uint8)
    err = float(np.mean(errs)) if errs else 0.0
    return pts, cols, err


def read_model_dir(model_dir: str) -> ColmapResult:
    res = ColmapResult()
    img_txt = os.path.join(model_dir, "images.txt")
    pts_txt = os.path.join(model_dir, "points3D.txt")
    cam_txt = os.path.join(model_dir, "cameras.txt")
    if os.path.exists(cam_txt):
        res.intrinsics = parse_cameras_txt(cam_txt)
    if os.path.exists(img_txt):
        res.poses = parse_images_txt(img_txt)
    if os.path.exists(pts_txt):
        res.points, res.colors, res.mean_reproj_error = parse_points3d_txt(pts_txt)
    return res


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------
def _stage_images(image_paths: List[str], dst_dir: str) -> str:
    """Collect the selected photos into one directory (hardlink, else copy)."""
    os.makedirs(dst_dir, exist_ok=True)
    for src in image_paths:
        dst = os.path.join(dst_dir, os.path.basename(src))
        if os.path.exists(dst):
            continue
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    return dst_dir


def _run_step(name: str, args: List[str], ctx) -> Optional[bool]:
    """Run one COLMAP subcommand. Returns True (ok) / False (failed) / None (cancelled)."""
    cmd = [exe()] + args
    ctx.log(f"COLMAP {name}...", "info")
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, encoding="utf-8", errors="replace")
    except OSError as exc:
        ctx.log(f"Could not launch COLMAP: {exc}", "error")
        return False
    tail: List[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            tail.append(line)
            del tail[:-12]
        if ctx.cancelled:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            ctx.log(f"COLMAP {name} cancelled.", "warn")
            return None
    rc = proc.wait()
    if rc != 0:
        ctx.log(f"COLMAP {name} failed (exit {rc}):", "error")
        for ln in tail[-8:]:
            ctx.log("  " + ln, "error")
        return False
    return True


def _largest_model(sparse_dir: str) -> Optional[str]:
    """mapper writes sparse/0, sparse/1, ... — pick the one with most images."""
    best, best_n = None, -1
    if not os.path.isdir(sparse_dir):
        return None
    for entry in sorted(os.listdir(sparse_dir)):
        sub = os.path.join(sparse_dir, entry)
        imgs = os.path.join(sub, "images.bin")
        imgs_txt = os.path.join(sub, "images.txt")
        if os.path.isdir(sub) and (os.path.exists(imgs) or os.path.exists(imgs_txt)):
            # crude size proxy: file size of the images model
            f = imgs if os.path.exists(imgs) else imgs_txt
            n = os.path.getsize(f)
            if n > best_n:
                best, best_n = sub, n
    return best


def run_sfm(image_paths: List[str], workdir: str, ctx, use_gpu: bool = False) -> Optional[ColmapResult]:
    """Full COLMAP SfM. Returns a ColmapResult, or None on failure/cancellation."""
    col_dir = os.path.join(workdir, "colmap")
    os.makedirs(col_dir, exist_ok=True)
    db = os.path.join(col_dir, "database.db")
    img_dir = _stage_images(image_paths, os.path.join(col_dir, "images"))
    sparse = os.path.join(col_dir, "sparse")
    os.makedirs(sparse, exist_ok=True)
    gpu = "1" if use_gpu else "0"

    steps = [
        ("feature_extractor", ["feature_extractor", "--database_path", db,
                               "--image_path", img_dir,
                               "--ImageReader.single_camera", "1",
                               "--SiftExtraction.use_gpu", gpu], 25),
        ("exhaustive_matcher", ["exhaustive_matcher", "--database_path", db,
                                "--SiftMatching.use_gpu", gpu], 55),
        ("mapper", ["mapper", "--database_path", db, "--image_path", img_dir,
                    "--output_path", sparse], 85),
    ]
    for name, args, pct in steps:
        r = _run_step(name, args, ctx)
        if r is None:
            return None
        if r is False:
            return None
        ctx.progress(pct)

    model = _largest_model(sparse)
    if model is None:
        ctx.log("COLMAP produced no reconstruction (too few matches?).", "error")
        return None

    # Ensure a text model exists for parsing.
    if not os.path.exists(os.path.join(model, "images.txt")):
        r = _run_step("model_converter", ["model_converter", "--input_path", model,
                                          "--output_path", model, "--output_type", "TXT"], ctx)
        if r is not True:
            return None

    ctx.progress(95)
    result = read_model_dir(model)
    result.model_dir = model
    result.image_dir = img_dir
    return result
