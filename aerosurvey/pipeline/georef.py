"""Georeferencing math: absolute orientation + triangulation + residuals.

Transforms a Structure-from-Motion reconstruction (in COLMAP's arbitrary local
frame) into the project's real-world CRS using either:
  * camera GPS positions (a 7-parameter similarity fit), or
  * surveyed Ground Control Points marked in the images (triangulated to the
    local frame, then fit to their known world coordinates).

Pure numpy — no COLMAP dependency — so every function is unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Similarity transform  (world = s * R @ local + t)
# ---------------------------------------------------------------------------
@dataclass
class Similarity:
    s: float
    R: np.ndarray  # 3x3 rotation
    t: np.ndarray  # (3,) translation

    def apply(self, P) -> np.ndarray:
        P = np.asarray(P, dtype=np.float64)
        single = P.ndim == 1
        P = np.atleast_2d(P)
        out = self.s * (P @ self.R.T) + self.t
        return out[0] if single else out

    def inverse(self) -> "Similarity":
        Rinv = self.R.T
        sinv = 1.0 / self.s
        return Similarity(sinv, Rinv, -sinv * (Rinv @ self.t))


def umeyama_similarity(src, dst, with_scale: bool = True) -> Similarity:
    """Least-squares similarity mapping src -> dst (Umeyama 1991).

    src, dst: (N, 3) arrays of corresponding points. Needs N >= 3
    non-collinear correspondences for a well-posed 3D solution.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n, dim = src.shape
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    sc = src - mu_s
    dc = dst - mu_d
    cov = (dc.T @ sc) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt
    if with_scale:
        var_src = (sc ** 2).sum() / n
        s = float((D * np.diag(S)).sum() / var_src) if var_src > 0 else 1.0
    else:
        s = 1.0
    t = mu_d - s * (R @ mu_s)
    return Similarity(s, R, t)


# ---------------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------------
@dataclass
class FitResult:
    rmse: float                 # total 3D RMSE
    rmse_axis: np.ndarray       # (3,) per-axis RMSE
    per_point: np.ndarray       # (N,) euclidean error per point


def residuals(src, dst, sim: Similarity) -> FitResult:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    pred = sim.apply(src)
    d = pred - dst
    per = np.linalg.norm(d, axis=1)
    rmse = float(np.sqrt(np.mean(per ** 2))) if len(per) else 0.0
    rmse_axis = np.sqrt(np.mean(d ** 2, axis=0)) if len(per) else np.zeros(3)
    return FitResult(rmse, rmse_axis, per)


# ---------------------------------------------------------------------------
# Projection + triangulation
# ---------------------------------------------------------------------------
def projection_matrix(K, R, t) -> np.ndarray:
    """3x4 projection P = K [R | t] with COLMAP convention X_cam = R X + t."""
    K = np.asarray(K, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    return K @ np.hstack([R, t])


def dlt_triangulate(proj_mats: Sequence[np.ndarray],
                    uvs: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Linear (DLT) triangulation of one 3D point from >=2 image observations."""
    rows: List[np.ndarray] = []
    for P, (u, v) in zip(proj_mats, uvs):
        P = np.asarray(P, dtype=np.float64)
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    A = np.asarray(rows, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-12:
        return X[:3]
    return X[:3] / X[3]
