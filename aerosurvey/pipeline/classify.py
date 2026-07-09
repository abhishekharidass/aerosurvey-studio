"""Point-cloud classification.

Two backends:
  * PDAL (`filters.smrf` ground filter + coplanarity) via subprocess when the
    CLI is present;
  * an in-process pipeline otherwise:
      1. statistical outlier removal (MVS clouds are full of flyers that
         corrupt any min-Z surface) -> ASPRS class 7 (noise);
      2. progressive morphological ground filter (Zhang et al. 2003);
      3. non-ground split by height-above-ground, local planarity and
         greenness: vegetation banded into low/medium/high (3/4/5),
         planar elevated surfaces -> building (6), the short planar
         remainder (cars, walls) stays unclassified (1).

Classes follow ASPRS: 1 unclassified, 2 ground, 3/4/5 vegetation,
6 building, 7 noise.
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


def default_cell(P: np.ndarray) -> float:
    """Ground-filter cell size from the cloud's actual point spacing."""
    minx, miny = float(P[:, 0].min()), float(P[:, 1].min())
    maxx, maxy = float(P[:, 0].max()), float(P[:, 1].max())
    area = max((maxx - minx) * (maxy - miny), 1e-6)
    spacing = float(np.sqrt(area / max(len(P), 1)))
    return float(np.clip(4.0 * spacing, 0.5, 3.0))


# ---------------------------------------------------------------------------
# Outlier / noise removal
# ---------------------------------------------------------------------------
def noise_mask(P: np.ndarray, k: int = 8, sigma: float = 3.0) -> np.ndarray:
    """True where a point is a statistical outlier (large mean kNN distance)."""
    from scipy.spatial import cKDTree
    n = len(P)
    if n < k + 1:
        return np.zeros(n, bool)
    tree = cKDTree(P)
    d, _ = tree.query(P, k=k + 1, workers=-1)   # first neighbour is the point itself
    mean_d = d[:, 1:].mean(axis=1)
    thr = mean_d.mean() + sigma * mean_d.std()
    return mean_d > thr


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
                slope: float = 0.15, base: float = 0.25, max_dh: float = 2.5,
                htol: float = 0.35) -> Tuple[np.ndarray, np.ndarray]:
    """(boolean ground mask, height above filtered terrain) per point."""
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
    # median smoothing keeps terrain shape but suppresses residual noise pits
    dem = ndimage.median_filter(dem, size=3)
    height = P[:, 2] - dem[iy, ix]
    return height <= htol, height


# ---------------------------------------------------------------------------
# Non-ground split: building (planar) vs vegetation (rough), banded by height
# ---------------------------------------------------------------------------
def _local_planarity(pts: np.ndarray, k: int = 12) -> np.ndarray:
    """Residual (m) of each point from its local best-fit plane (vectorised)."""
    from scipy.spatial import cKDTree
    n = len(pts)
    if n < 3:
        return np.zeros(n)
    tree = cKDTree(pts)
    k = min(k, n)
    out = np.empty(n)
    block = 500_000                                # bound peak memory
    for s in range(0, n, block):
        e = min(s + block, n)
        _, nn = tree.query(pts[s:e], k=k, workers=-1)
        nb = pts[nn]                               # (B, k, 3)
        centered = nb - nb.mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", centered, centered) / k
        evals = np.linalg.eigvalsh(cov)            # ascending
        out[s:e] = np.sqrt(np.maximum(evals[:, 0], 0.0))
    return out                                     # plane-fit residual


def classify_cloud(P: np.ndarray, C: np.ndarray, cell: Optional[float] = None,
                   rough_thresh: float = 0.18, denoise: bool = True) -> np.ndarray:
    """Return an ASPRS classification array for the whole cloud."""
    cls = np.ones(len(P), np.uint8)                       # 1 = unclassified
    if cell is None:
        cell = default_cell(P)

    keep = np.ones(len(P), bool)
    if denoise and len(P) > 100:
        noisy = noise_mask(P)
        cls[noisy] = 7                                    # noise
        keep = ~noisy
    Pk = P[keep]
    if len(Pk) < 10:
        return cls

    gmask, hag = ground_mask(Pk, cell=cell)
    kidx = np.where(keep)[0]
    cls[kidx[gmask]] = 2                                  # ground

    non = np.where(~gmask)[0]
    if len(non) == 0:
        return cls
    Pn, Cn, hn = Pk[non], C[kidx[non]], hag[non]
    rough = _local_planarity(Pn)
    green = (Cn[:, 1].astype(int) > Cn[:, 0].astype(int) + 10) & \
            (Cn[:, 1].astype(int) > Cn[:, 2].astype(int) + 10)
    veg = (rough > rough_thresh) | green

    sub = np.ones(len(non), np.uint8)                     # default: unclassified
    sub[veg & (hn <= 2.0)] = 3                            # low vegetation
    sub[veg & (hn > 2.0) & (hn <= 5.0)] = 4               # medium vegetation
    sub[veg & (hn > 5.0)] = 5                             # high vegetation
    sub[~veg & (hn > 2.5)] = 6                            # building (planar, tall)
    cls[kidx[non]] = sub
    return cls


# ---------------------------------------------------------------------------
# PDAL backend
# ---------------------------------------------------------------------------
def pdal_classify(in_las: str, out_las: str) -> bool:
    """PDAL: outlier removal + SMRF ground + HAG bands + coplanar buildings."""
    pipeline = {
        "pipeline": [
            in_las,
            {"type": "filters.outlier", "method": "statistical",
             "mean_k": 8, "multiplier": 3.0},
            {"type": "filters.smrf", "ignore": "Classification[7:7]"},
            {"type": "filters.hag_nn"},
            {"type": "filters.approximatecoplanar", "knn": 10},
            {"type": "filters.assign", "value": [
                "Classification = 3 WHERE HeightAboveGround > 0.35 "
                "AND HeightAboveGround <= 2 AND Classification != 2 AND Classification != 7",
                "Classification = 4 WHERE HeightAboveGround > 2 "
                "AND HeightAboveGround <= 5 AND Classification != 2 AND Classification != 7",
                "Classification = 5 WHERE HeightAboveGround > 5 "
                "AND Classification != 2 AND Classification != 7",
                "Classification = 6 WHERE Coplanar == 1 AND HeightAboveGround > 2.5 "
                "AND Classification != 2 AND Classification != 7"]},
            {"type": "writers.las", "filename": out_las,
             "extra_dims": "all", "minor_version": 4},
        ]
    }
    exe = shutil.which("pdal") or shutil.which("pdal.exe")
    proc = subprocess.run([exe, "pipeline", "--stdin"],
                          input=json.dumps(pipeline), text=True,
                          capture_output=True)
    return proc.returncode == 0 and os.path.exists(out_las)
