"""Predict GCP image marks by projecting surveyed coordinates into photos.

Two tiers, best available wins:
  * reconstruction tier — after Align + Georeference, the GCP's world
    coordinates are mapped into COLMAP's local frame (inverse similarity)
    and projected through each solved camera: marks land on / next to the
    target, ready for sub-pixel refinement;
  * EXIF tier — before alignment, an approximate nadir camera model is
    built from the geotag, gimbal yaw/pitch and focal length: marks land
    in the right neighbourhood (GPS-accuracy), still a huge head start.

Existing user-placed marks are never overwritten.
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from ..core import crs as crsmod
from ..core import gsd as gsdmod
from . import recon


def _project(K, R, t, X) -> Optional[tuple]:
    """Project one 3D point (camera convention X_cam = R X + t). None if behind."""
    Xc = R @ np.asarray(X, float) + t
    if Xc[2] <= 1e-6:
        return None
    u = K[0][0] * Xc[0] / Xc[2] + K[0][2]
    v = K[1][1] * Xc[1] / Xc[2] + K[1][2]
    return float(u), float(v), float(Xc[2])


def _inside(u, v, w, h, margin: float = 0.02) -> bool:
    mx, my = w * margin, h * margin
    return mx <= u <= w - mx and my <= v <= h - my


# -- tier 1: solved reconstruction -------------------------------------------
def _mark_from_reconstruction(chunk, res, sim, gcps, max_images: int) -> int:
    inv = sim.inverse()
    by_name = {c.filename: c for c in chunk.cameras}
    n_marks = 0
    for g in gcps:
        Xl = inv.apply([g.x, g.y, g.z])
        hits = []
        for name, pose in res.poses.items():
            cam = by_name.get(name)
            K = res.intrinsics.get(pose.camera_id)
            if cam is None or K is None or not cam.width:
                continue
            R = _qrot(pose.qvec)
            p = _project(K, R, np.asarray(pose.tvec, float), Xl)
            if p is None:
                continue
            u, v, _depth = p
            if not _inside(u, v, cam.width, cam.height):
                continue
            # prefer views where the point is nearest the image centre
            centrality = math.hypot(u - cam.width / 2, v - cam.height / 2)
            hits.append((centrality, cam.id, u, v))
        hits.sort()
        for _, cam_id, u, v in hits[:max_images]:
            if cam_id not in g.observations:      # keep user-placed marks
                g.mark(cam_id, u, v)
                n_marks += 1
    return n_marks


def _qrot(q) -> np.ndarray:
    from . import colmap
    return colmap.qvec2rotmat(q)


# -- tier 2: EXIF-only nadir approximation ------------------------------------
def _exif_rotation(yaw_deg: Optional[float], pitch_deg: Optional[float]) -> np.ndarray:
    """World(ENU)->camera rotation for a (near-)nadir gimbal.

    yaw: heading of the image top, clockwise from north. pitch: gimbal pitch
    (-90 = straight down). Rows are the camera axes expressed in world coords.
    """
    psi = math.radians(yaw_deg or 0.0)
    x_cam = np.array([math.cos(psi), -math.sin(psi), 0.0])   # image right
    y_cam = np.array([-math.sin(psi), -math.cos(psi), 0.0])  # image down
    z_cam = np.array([0.0, 0.0, -1.0])                       # view direction
    R0 = np.vstack([x_cam, y_cam, z_cam])
    delta = math.radians((pitch_deg if pitch_deg is not None else -90.0) + 90.0)
    if abs(delta) > 1e-6:  # tilt off nadir, about the camera x axis, toward the heading
        c, s = math.cos(delta), math.sin(delta)
        Rx = np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
        R0 = Rx @ R0
    return R0


def _mark_from_exif(chunk, gcps, max_images: int) -> int:
    tf = crsmod.CrsTransform(chunk.epsg if chunk.crs_mode != "local" else None)
    geoid = (chunk.geoid_separation
             if getattr(chunk, "vertical_datum", "") == "orthometric" else 0.0)
    n_marks = 0
    for g in gcps:
        cands = []
        for cam in chunk.cameras:
            if not (cam.enabled and cam.has_geotag and cam.width):
                continue
            f_px = gsdmod.focal_px(cam)
            if not f_px:
                continue
            cx, cy, cz = tf.forward(cam.lon, cam.lat, cam.alt or 0.0)
            cz -= geoid                              # into the GCP's vertical datum
            if cz - g.z < 5:                         # need real height above the point
                if cam.rel_alt and cam.rel_alt > 5:
                    cz = g.z + cam.rel_alt
                else:
                    continue
            C = np.array([cx, cy, cz])
            R = _exif_rotation(cam.yaw, cam.pitch)
            K = [[f_px, 0, cam.width / 2], [0, f_px, cam.height / 2], [0, 0, 1]]
            p = _project(np.asarray(K), R, -R @ C, [g.x, g.y, g.z])
            if p is None:
                continue
            u, v, _ = p
            if _inside(u, v, cam.width, cam.height, margin=0.05):
                dist = math.hypot(cx - g.x, cy - g.y)
                cands.append((dist, cam.id, u, v))
        cands.sort()
        for _, cam_id, u, v in cands[:max_images]:
            if cam_id not in g.observations:
                g.mark(cam_id, u, v)
                n_marks += 1
    return n_marks


# -- entry point ---------------------------------------------------------------
def auto_mark(chunk, workdir: str, gcp_ids: Optional[List[int]] = None,
              max_images: int = 8) -> tuple:
    """Place predicted marks for the given (or all) GCPs.

    Returns (n_marks, method) where method is "reconstruction", "exif" or "".
    """
    gcps = [g for g in chunk.gcps
            if g.enabled and (gcp_ids is None or g.id in gcp_ids)
            and (g.x, g.y, g.z) != (0.0, 0.0, 0.0)]
    if not gcps:
        return 0, ""
    res = recon.get_reconstruction(chunk, workdir)
    sim = recon.get_sim(chunk, workdir)
    if res is not None and sim is not None and res.intrinsics:
        return _mark_from_reconstruction(chunk, res, sim, gcps, max_images), "reconstruction"
    return _mark_from_exif(chunk, gcps, max_images), "exif"
