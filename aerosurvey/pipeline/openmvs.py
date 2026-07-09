"""OpenMVS orchestration: turn a COLMAP sparse reconstruction into a dense cloud.

Standard recipe (all off the UI thread, cancellable):
    colmap image_undistorter   (undistort images + export a dense workspace)
    InterfaceCOLMAP            (COLMAP workspace -> scene.mvs)
    DensifyPointCloud          (multi-view stereo -> scene_dense.ply)

The resulting PLY is read via Open3D. If OpenMVS is not installed the caller
falls back to the built-in simulation.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import List, Optional

import numpy as np


def _which(*names: str) -> str:
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return ""


def interface_exe() -> str:
    return _which("InterfaceCOLMAP", "InterfaceCOLMAP.exe")


def densify_exe() -> str:
    return _which("DensifyPointCloud", "DensifyPointCloud.exe")


def available() -> bool:
    return bool(densify_exe() and interface_exe())


def _run(cmd: List[str], name: str, ctx) -> Optional[bool]:
    """Run one subprocess. True=ok, False=failed, None=cancelled."""
    ctx.log(f"OpenMVS {name}...", "info")
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, encoding="utf-8", errors="replace")
    except OSError as exc:
        ctx.log(f"Could not launch {name}: {exc}", "error")
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
            ctx.log(f"OpenMVS {name} cancelled.", "warn")
            return None
    rc = proc.wait()
    if rc != 0:
        ctx.log(f"OpenMVS {name} failed (exit {rc}):", "error")
        for ln in tail[-8:]:
            ctx.log("  " + ln, "error")
        return False
    return True


# Dense matching quality -> DensifyPointCloud --resolution-level
# (how many times the images are halved before stereo matching).
QUALITY_LEVELS = {"ultra": 0, "high": 1, "medium": 2, "low": 3}


def run_dense(colmap_model_dir: str, image_dir: str, workdir: str, ctx,
              colmap_exe: str, quality: str = "high") -> Optional[str]:
    """Produce a dense PLY. Returns its path, or None on failure/cancellation."""
    mvs = os.path.join(workdir, "openmvs")
    os.makedirs(mvs, exist_ok=True)
    undist = os.path.join(mvs, "undistort")

    if _run([colmap_exe, "image_undistorter", "--image_path", image_dir,
             "--input_path", colmap_model_dir, "--output_path", undist,
             "--output_type", "COLMAP"], "image_undistorter", ctx) is not True:
        return None
    ctx.progress(30)

    scene = os.path.join(mvs, "scene.mvs")
    if _run([interface_exe(), "-i", undist, "-o", scene,
             "--image-folder", os.path.join(undist, "images")],
            "InterfaceCOLMAP", ctx) is not True:
        return None
    ctx.progress(45)

    dens = [densify_exe(), scene, "-o", "scene_dense.mvs", "-w", mvs]
    # env var overrides the project setting (debugging escape hatch)
    rl = os.environ.get("AEROSURVEY_MVS_RESOLUTION_LEVEL")
    if not rl:
        rl = str(QUALITY_LEVELS.get(quality, 1))
    dens += ["--resolution-level", str(rl)]  # higher = more downsampling = coarser
    if _run(dens, "DensifyPointCloud", ctx) is not True:
        return None
    ctx.progress(85)

    ply = os.path.join(mvs, "scene_dense.ply")
    if not os.path.exists(ply):
        ctx.log("DensifyPointCloud produced no scene_dense.ply.", "error")
        return None
    return ply


def load_ply(path: str):
    """Read a PLY point cloud -> (points Nx3 float64, colors Nx3 uint8).

    Uses plyfile (lightweight, pure-Python) so the core app has no Open3D
    dependency and can be bundled into a small portable executable.
    """
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"]
    names = v.data.dtype.names
    P = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    if all(k in names for k in ("red", "green", "blue")):
        C = np.column_stack([v["red"], v["green"], v["blue"]]).astype(np.uint8)
    else:
        C = np.full((len(P), 3), 200, np.uint8)
    return P, C
