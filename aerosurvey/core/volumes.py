"""Volume / stockpile measurement over the DSM (DroneDeploy-style).

The user outlines a polygon (project CRS); a base surface is derived from
the DSM along the polygon boundary — or given explicitly — and cut/fill
volumes integrate the per-cell height difference inside the polygon:

    cut  = material above the base (would be removed),
    fill = void below the base (would be filled),
    net  = cut - fill.

Base modes: "lowest" (minimum boundary elevation — classic stockpile toe),
"mean" (average boundary elevation), "fit" (least-squares plane through the
boundary — sloping sites) and "custom" (explicit elevation).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

import numpy as np

BASE_MODES = ("lowest", "mean", "fit", "custom")


@dataclass
class VolumeResult:
    cut_m3: float
    fill_m3: float
    net_m3: float
    area_m2: float            # 2D polygon area (shoelace)
    measured_m2: float        # area of valid DSM cells actually integrated
    coverage: float           # measured / polygon area (0..1)
    base_mode: str
    base_z_min: float
    base_z_max: float
    cell_m: float
    n_cells: int
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        base = (f"{self.base_z_min:.2f} m" if self.base_z_min == self.base_z_max
                else f"{self.base_z_min:.2f}–{self.base_z_max:.2f} m")
        return (f"Cut {self.cut_m3:,.1f} m³ · Fill {self.fill_m3:,.1f} m³ · "
                f"Net {self.net_m3:+,.1f} m³  (area {self.area_m2:,.1f} m², "
                f"{self.coverage * 100:.0f}% measured, base [{self.base_mode}] "
                f"{base}, cell {self.cell_m:.2f} m)")


def polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _boundary_samples(poly: np.ndarray, spacing: float) -> np.ndarray:
    """Points along the closed polygon boundary at roughly `spacing` metres."""
    pts = []
    closed = np.vstack([poly, poly[:1]])
    for a, b in zip(closed[:-1], closed[1:]):
        seg = np.linalg.norm(b - a)
        n = max(int(seg / max(spacing, 1e-6)), 1)
        t = np.arange(n) / n
        pts.append(a + t[:, None] * (b - a))
    return np.vstack(pts)


def measure_volume(dsm_path: str, polygon: Sequence[Tuple[float, float]],
                   base_mode: str = "lowest",
                   custom_z: float = 0.0) -> VolumeResult:
    """Integrate cut/fill volume of the DSM inside a project-CRS polygon."""
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.windows import Window, from_bounds

    poly = np.asarray(polygon, np.float64)
    if poly.ndim != 2 or len(poly) < 3:
        raise ValueError("polygon needs at least 3 vertices")
    if base_mode not in BASE_MODES:
        raise ValueError(f"base_mode must be one of {BASE_MODES}")

    with rasterio.open(dsm_path) as src:
        cell_w, cell_h = abs(src.transform.a), abs(src.transform.e)
        pad = 2 * max(cell_w, cell_h)
        win = from_bounds(poly[:, 0].min() - pad, poly[:, 1].min() - pad,
                          poly[:, 0].max() + pad, poly[:, 1].max() + pad,
                          src.transform)
        try:
            win = win.round_offsets().round_lengths() \
                     .intersection(Window(0, 0, src.width, src.height))
        except Exception:
            raise ValueError("polygon lies outside the DSM")
        if win.width <= 0 or win.height <= 0:
            raise ValueError("polygon lies outside the DSM")
        z = src.read(1, window=win, masked=True).astype(np.float64)
        tr = src.window_transform(win)

    ring = poly.tolist() + [poly[0].tolist()]
    inside = geometry_mask([{"type": "Polygon", "coordinates": [ring]}],
                           out_shape=z.shape, transform=tr, invert=True)
    valid = inside & ~np.ma.getmaskarray(z)
    warnings = []
    if not valid.any():
        raise ValueError("no valid DSM cells inside the polygon")

    # base surface from DSM along the boundary
    if base_mode == "custom":
        base = np.full(z.shape, float(custom_z))
    else:
        bp = _boundary_samples(poly, spacing=max(cell_w, cell_h))
        inv = ~tr
        cols, rows = inv * (bp[:, 0], bp[:, 1])
        cols = np.clip(cols.astype(int), 0, z.shape[1] - 1)
        rows = np.clip(rows.astype(int), 0, z.shape[0] - 1)
        bz = z[rows, cols]
        keep = ~np.ma.getmaskarray(bz)
        if keep.sum() < 3:
            raise ValueError("polygon boundary has no valid DSM elevations")
        if keep.sum() < 0.5 * len(bp):
            warnings.append("over half the boundary lies on nodata — the base "
                            "surface may be unreliable")
        bz = np.asarray(bz[keep], np.float64)
        if base_mode == "lowest":
            base = np.full(z.shape, float(bz.min()))
        elif base_mode == "mean":
            base = np.full(z.shape, float(bz.mean()))
        else:  # fit: least-squares plane through boundary samples
            # regress against the sampled cell centres, not the raw boundary
            # points — otherwise the floor() in the index lookup biases the
            # plane by half a cell times the terrain slope
            bx, by = tr * (cols[keep] + 0.5, rows[keep] + 0.5)
            A = np.column_stack([bx - bx.mean(), by - by.mean(),
                                 np.ones(len(bx))])
            coef, *_ = np.linalg.lstsq(A, bz, rcond=None)
            jj, ii = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0]))
            gx, gy = tr * (jj + 0.5, ii + 0.5)
            base = (coef[0] * (gx - bx.mean()) + coef[1] * (gy - by.mean())
                    + coef[2])

    cell_area = cell_w * cell_h
    diff = np.where(valid, z.filled(0.0) - base, 0.0)
    cut = float(diff[diff > 0].sum() * cell_area)
    fill = float(-diff[diff < 0].sum() * cell_area)

    area2d = float(polygon_area(poly))
    measured = float(valid.sum() * cell_area)
    coverage = min(measured / max(area2d, 1e-9), 1.0)
    if coverage < 0.95:
        warnings.append(f"only {coverage * 100:.0f}% of the polygon has DSM "
                        "data — volumes underestimate the true amount")
    binside = base[valid]
    return VolumeResult(
        cut_m3=cut, fill_m3=fill, net_m3=cut - fill, area_m2=area2d,
        measured_m2=measured, coverage=coverage, base_mode=base_mode,
        base_z_min=float(binside.min()), base_z_max=float(binside.max()),
        cell_m=max(cell_w, cell_h), n_cells=int(valid.sum()),
        warnings=warnings)
