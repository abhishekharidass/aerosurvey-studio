"""Ground Sample Distance estimation.

The GSD (metres per pixel of the source photos at ground level) drives the
default resolution of the DSM/DTM/orthomosaic so the outputs match what the
sensor actually captured instead of an arbitrary pixel budget.

Two estimators, best first:
  * reconstruction-based: flying height above ground (from solved camera
    positions and the point cloud) divided by the focal length in pixels
    (from the COLMAP intrinsics) — accurate, works even with broken EXIF;
  * EXIF-based: focal length + sensor width (or 35mm-equivalent focal) and
    the drone's RelativeAltitude XMP tag — available before any processing.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

# Sensor widths (mm) for common survey drones, keyed by a lowercase substring
# of the EXIF camera model. Used when only the real focal length is known.
SENSOR_WIDTHS_MM = {
    # DJI
    "m3e": 17.3, "m3t": 6.4, "m3m": 17.3, "mavic 3": 17.3,
    "fc220": 6.17, "fc300": 6.17, "fc330": 6.17,            # Phantom 3/4, Mavic Pro
    "fc6310": 13.2, "fc6360": 13.2,                          # Phantom 4 Pro / RTK
    "fc7303": 6.17,                                          # Mini 2
    "zenmusep1": 35.9, "p1": 35.9,                           # M300 P1 (full frame)
    "l1": 8.8, "l2": 17.3,
    "fc3411": 13.2,                                          # Air 2S
    "fc3170": 6.4,                                           # Mavic Air 2
    # Others
    "evo ii": 13.2, "xt701": 13.2,                           # Autel
    "anafi": 7.22,                                           # Parrot
    "wingtraone": 23.5, "aeria x": 23.5,
}


def sensor_width_mm(make: str, model: str) -> Optional[float]:
    key = f"{make} {model}".lower()
    for sub, w in SENSOR_WIDTHS_MM.items():
        if sub in key:
            return w
    return None


def focal_px(cam) -> Optional[float]:
    """Focal length in pixels from EXIF, or None if not derivable."""
    if not cam.width:
        return None
    if cam.focal_mm:
        sw = sensor_width_mm(cam.make, cam.model)
        if sw:
            return cam.width * cam.focal_mm / sw
    if cam.focal35_mm:
        return cam.width * cam.focal35_mm / 36.0
    return None


def height_agl(cam, ground_z: Optional[float] = None) -> Optional[float]:
    """Height above ground (m). Prefers the solved camera height over a known
    ground elevation; falls back to the drone's take-off-relative altitude."""
    if ground_z is not None and cam.est_z is not None:
        h = cam.est_z - ground_z
        if h > 2:
            return h
    if cam.rel_alt is not None and cam.rel_alt > 2:
        return cam.rel_alt
    return None


def estimate_gsd(chunk, ground_z: Optional[float] = None,
                 f_px_by_cam: Optional[Dict[int, float]] = None) -> Optional[float]:
    """Median per-photo GSD (m/px) across the chunk, or None if unknown.

    f_px_by_cam: optional camera-id -> focal(px) from the reconstruction;
    EXIF-derived focals fill the gaps.
    """
    vals = []
    for c in chunk.cameras:
        if not c.enabled:
            continue
        f = (f_px_by_cam or {}).get(c.id) or focal_px(c)
        h = height_agl(c, ground_z)
        if f and h:
            vals.append(h / f)
    if not vals:
        return None
    return float(np.median(vals))
