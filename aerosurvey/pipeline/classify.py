"""Point-cloud classification.

Two backends:
  * PDAL (`filters.smrf` ground filter) via subprocess when the CLI is present;
  * a real in-process algorithm otherwise — a progressive morphological ground
    filter (Zhang et al. 2003) plus KDTree local-roughness to split non-ground
    into buildings (planar) and vegetation (rough), colour as a tie-breaker.

Both are genuine algorithms; the scipy path is the default and is unit-tested.
Classes follow ASPRS: 2 ground, 5 high vegetation, 6 building, 1 unclassified.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Optional, Tuple

import numpy as np


def pdal_available() -> bool:
    return bool(shutil.which("pdal") or shutil.which("pdal.exe"))


# ---------------------------------------------------------------------------
# Ground filter (progressive morphological filter on a min-Z surface)
# ---------------------------------------------------------------------------
def _min_surface(P: np.ndarray, cell: float):
    minx, miny = float(P[:, 0].min()), float(P[:, 1].min())
    maxx, maxy = float(P[:, 0].max()), float(P[:, 1].max())
    nx = max(int(np.ceil((maxx - minx) / cell)) + 1, 1)
    ny = max(int(np.ceil((maxy - miny) / cell)) + 1, 1)
    ix = np.clip(((P[:, 0] - minx) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((maxy - P[:, 1]) / cell).astype(int), 0, ny - 1)  # row 0 = north
    surf = np.full((ny, nx), np.inf, np.float64)
    np.minimum.at(surf, (iy, ix), P[:, 2])
    return surf, (ix, iy)


def _fill_infinite(surf: np.ndarray) -> np.ndarray:
    from scipy import ndimage
    mask = ~np.isfinite(surf)
    if not mask.any():
        return surf
    idx = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
    return surf[tuple(idx)]


def ground_mask(P: np.ndarray, cell: float = 1.0, max_window: int = 33,
                slope: float = 0.2, base: float = 0.3, max_dh: float = 2.5,
                htol: float = 0.4) -> Tuple[np.ndarray, np.ndarray]:
    """Boolean ground mask per point using a progressive morphological filter."""
    from scipy import ndimage
    surf, (ix, iy) = _min_surface(P, cell)
    surf = _fill_infinite(surf)
    dem = surf.copy()
    window = 1
    while window <= max_window:
        size = 2 * window + 1
        opened = ndimage.grey_opening(dem, size=size)
        dh_thresh = min(base + slope * (window * cell), max_dh)
        # keep terrain (small drops), erase objects (large drops) at this scale
        removed = (dem - opened) > dh_thresh
        dem = np.where(removed, opened, dem)
        window *= 2
    dem = ndimage.grey_opening(dem, size=3)  # light final smoothing
    height = P[:, 2] - dem[iy, ix]
    return height <= htol, height


# ---------------------------------------------------------------------------
# Non-ground split: building (planar) vs vegetation (rough)
# ---------------------------------------------------------------------------
def _local_roughness(pts: np.ndarray, k: int = 12) -> np.ndarray:
    """Std deviation of each point from its local best-fit plane."""
    from scipy.spatial import cKDTree
    if len(pts) < 3:
        return np.zeros(len(pts))
    tree = cKDTree(pts)
    k = min(k, len(pts))
    _, nn = tree.query(pts, k=k)
    rough = np.empty(len(pts))
    for i in range(len(pts)):
        nb = pts[nn[i]]
        cov = np.cov((nb - nb.mean(axis=0)).T)
        w = np.linalg.eigvalsh(cov)
        rough[i] = np.sqrt(max(w[0], 0.0))  # smallest eigenvalue -> planarity residual
    return rough


def classify_cloud(P: np.ndarray, C: np.ndarray, cell: float = 1.0,
                   rough_thresh: float = 0.25) -> np.ndarray:
    """Return an ASPRS classification array for the whole cloud."""
    cls = np.ones(len(P), np.uint8)  # 1 = unclassified
    gmask, _ = ground_mask(P, cell=cell)
    cls[gmask] = 2  # ground

    non_idx = np.where(~gmask)[0]
    if len(non_idx) == 0:
        return cls
    rough = _local_roughness(P[non_idx])
    green = (C[non_idx, 1].astype(int) > C[non_idx, 0].astype(int) + 10) & \
            (C[non_idx, 1].astype(int) > C[non_idx, 2].astype(int) + 10)
    is_veg = (rough > rough_thresh) | green
    sub = np.where(is_veg, 5, 6).astype(np.uint8)  # 5 veg, 6 building
    cls[non_idx] = sub
    return cls


# ---------------------------------------------------------------------------
# PDAL backend
# ---------------------------------------------------------------------------
def pdal_classify(in_las: str, out_las: str) -> bool:
    """Run a PDAL SMRF ground filter + HAG-based class assignment."""
    pipeline = {
        "pipeline": [
            in_las,
            {"type": "filters.smrf"},
            {"type": "filters.hag_nn"},
            {"type": "filters.assign",
             "value": ["Classification = 5 WHERE HeightAboveGround > 2",
                       "Classification = 6 WHERE HeightAboveGround > 0.5 "
                       "AND HeightAboveGround <= 2"]},
            {"type": "writers.las", "filename": out_las,
             "extra_dims": "all", "minor_version": 4},
        ]
    }
    exe = shutil.which("pdal") or shutil.which("pdal.exe")
    proc = subprocess.run([exe, "pipeline", "--stdin"],
                          input=json.dumps(pipeline), text=True,
                          capture_output=True)
    return proc.returncode == 0 and os.path.exists(out_las)
