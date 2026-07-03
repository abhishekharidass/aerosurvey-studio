"""Sparse bundle adjustment (scipy.optimize.least_squares).

Refines camera extrinsics (axis-angle rotation + translation) and 3D point
positions to minimise reprojection error. Points flagged ``fixed`` (e.g. Ground
Control Points pinned to their surveyed world coordinates) are held constant, so
the solve is GCP-constrained. Intrinsics are held fixed.

Pure numpy/scipy — unit-testable without any external engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation


def project(rvec, t, K, X) -> np.ndarray:
    """Project world points X (N,3) with camera (rvec, t) and intrinsics K -> (N,2)."""
    R = Rotation.from_rotvec(rvec).as_matrix()
    Xc = X @ R.T + t                      # X_cam = R X + t
    uv = Xc @ K.T
    return uv[:, :2] / uv[:, 2:3]


@dataclass
class BAResult:
    rvecs: np.ndarray       # (nc, 3)
    tvecs: np.ndarray       # (nc, 3)
    points: np.ndarray      # (np, 3)
    rmse_before: float
    rmse_after: float
    n_obs: int


def bundle_adjust(rvecs, tvecs, Ks, points, fixed_mask, obs,
                  refine_points: bool = True, max_nfev: int = 100) -> BAResult:
    """
    rvecs, tvecs : (nc, 3) camera extrinsics (X_cam = R(rvec) X + t)
    Ks           : (nc, 3, 3) intrinsics per camera (held fixed)
    points       : (np, 3) 3D points
    fixed_mask   : (np,) bool, True => point held constant (control/GCP)
    obs          : (m, 4) rows [cam_idx, point_idx, u, v]
    """
    rvecs = np.array(rvecs, float)
    tvecs = np.array(tvecs, float)
    Ks = np.asarray(Ks, float)
    points = np.array(points, float)
    fixed_mask = np.asarray(fixed_mask, bool)
    obs = np.asarray(obs, float)
    nc = len(rvecs)

    free = np.where(~fixed_mask)[0] if refine_points else np.empty(0, int)
    free_row = {int(p): i for i, p in enumerate(free)}
    ncam = 6 * nc
    n_free = len(free)

    cam_idx = obs[:, 0].astype(int)
    pt_idx = obs[:, 1].astype(int)
    uv = obs[:, 2:4]

    x0 = np.concatenate([np.hstack([rvecs, tvecs]).ravel(),
                         points[free].ravel()])

    def unpack(x):
        cams = x[:ncam].reshape(nc, 6)
        pts = points.copy()
        if n_free:
            pts[free] = x[ncam:].reshape(n_free, 3)
        return cams[:, :3], cams[:, 3:], pts

    def residuals(x):
        rv, tv, pts = unpack(x)
        res = np.empty((len(obs), 2))
        for ci in range(nc):
            sel = cam_idx == ci
            if sel.any():
                res[sel] = project(rv[ci], tv[ci], Ks[ci], pts[pt_idx[sel]]) - uv[sel]
        return res.ravel()

    # Jacobian sparsity: each residual pair depends on its camera (+ its point if free)
    A = lil_matrix((2 * len(obs), len(x0)), dtype=np.uint8)
    for k in range(len(obs)):
        ci = cam_idx[k]
        A[2 * k:2 * k + 2, 6 * ci:6 * ci + 6] = 1
        row = free_row.get(pt_idx[k])
        if row is not None:
            col = ncam + 3 * row
            A[2 * k:2 * k + 2, col:col + 3] = 1

    def rmse(r):
        return float(np.sqrt(np.mean(r ** 2))) if len(r) else 0.0

    r0 = residuals(x0)
    sol = least_squares(residuals, x0, jac_sparsity=A, method="trf", x_scale="jac",
                        max_nfev=max_nfev, xtol=1e-12, ftol=1e-12, verbose=0)
    rv, tv, pts = unpack(sol.x)
    return BAResult(rv, tv, pts, rmse(r0), rmse(residuals(sol.x)), len(obs))


# ---------------------------------------------------------------------------
# Conversions between camera centre and (rvec, t)
# ---------------------------------------------------------------------------
def center_from_rt(rvec, t) -> np.ndarray:
    R = Rotation.from_rotvec(rvec).as_matrix()
    return -R.T @ np.asarray(t)


def rt_from_qc(qvec, center) -> tuple:
    """COLMAP (qvec world->cam, camera centre) -> (rvec, t) with t = -R center."""
    from .colmap import qvec2rotmat
    R = qvec2rotmat(qvec)
    t = -R @ np.asarray(center)
    return Rotation.from_matrix(R).as_rotvec(), t
