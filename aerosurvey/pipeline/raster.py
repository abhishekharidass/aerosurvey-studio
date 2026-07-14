"""Raster helpers shared by the DSM/DTM/orthomosaic stages.

Strategy for fine-GSD outputs from a point cloud whose density does not
support the target cell size directly: rasterise at the cloud's *native*
spacing (so nearly every cell is observed), fill the few remaining holes by
nearest-neighbour (EDT), then interpolate up to the requested GSD. Writing
happens block-wise through rasterio so multi-gigapixel rasters never require
a full-size intermediate array in memory.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np


# -- geometry ----------------------------------------------------------------
def extent(P: np.ndarray) -> Tuple[float, float, float, float]:
    return (float(P[:, 0].min()), float(P[:, 1].min()),
            float(P[:, 0].max()), float(P[:, 1].max()))


def median_spacing(P: np.ndarray) -> float:
    """Approximate mean point spacing (m) from the areal density."""
    minx, miny, maxx, maxy = extent(P)
    area = max((maxx - minx) * (maxy - miny), 1e-6)
    return float(np.sqrt(area / max(len(P), 1)))


def cell_indices(P, minx, maxy, cell, nx, ny):
    """Column/row indices per point (row 0 = north/top edge)."""
    ix = np.clip(((P[:, 0] - minx) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((maxy - P[:, 1]) / cell).astype(int), 0, ny - 1)
    return ix, iy


def grid_shape(P, cell) -> Tuple[int, int]:
    minx, miny, maxx, maxy = extent(P)
    nx = max(int(np.ceil((maxx - minx) / cell)) + 1, 1)
    ny = max(int(np.ceil((maxy - miny) / cell)) + 1, 1)
    return nx, ny


# -- native-resolution gridding ------------------------------------------
def fill_nearest(arr: np.ndarray) -> np.ndarray:
    """Fill NaN cells with the nearest finite value (Euclidean distance)."""
    from scipy import ndimage
    mask = np.isnan(arr)
    if not mask.any():
        return arr
    idx = ndimage.distance_transform_edt(mask, return_distances=False,
                                         return_indices=True)
    return arr[tuple(idx)]


def native_grid(P: np.ndarray, values: np.ndarray, cell: float,
                reducer: str = "max", orderby=None) -> np.ndarray:
    """Rasterise values over the point extent at `cell`, holes filled.

    orderby: optional per-point sort key — the largest key wins each cell
    (e.g. rasterise colours with orderby=Z so the top surface is kept).
    Otherwise `reducer` picks the max/min of `values` itself.
    """
    minx, miny, maxx, maxy = extent(P)
    nx, ny = grid_shape(P, cell)
    ix, iy = cell_indices(P, minx, maxy, cell, nx, ny)
    arr = np.full((ny, nx), np.nan, np.float32)
    if orderby is not None:
        order = np.argsort(orderby)
    else:
        order = np.argsort(values) if reducer == "max" else np.argsort(-values)
    arr[iy[order], ix[order]] = np.asarray(values, np.float32)[order]
    return fill_nearest(arr)


def fill_smooth(arr: np.ndarray) -> np.ndarray:
    """Fill NaN regions with *smooth* interpolation (coarse-to-fine pyramid).

    Nearest-neighbour filling is wrong for texture projection: a point-free
    slab interior inherits scaffold/ground heights from its rim and the
    projected imagery warps by metres. Diffusion filling spans holes with a
    smooth surface anchored at their edges instead."""
    import warnings
    if not np.isnan(arr).any():
        return arr
    levels = []
    cur = arr.astype(np.float32, copy=True)
    while np.isnan(cur).any() and min(cur.shape) > 2:
        levels.append(cur)
        ny, nx = cur.shape
        py, px = (ny + 1) // 2, (nx + 1) // 2
        pad = np.full((py * 2, px * 2), np.nan, np.float32)
        pad[:ny, :nx] = cur
        blocks = pad.reshape(py, 2, px, 2).transpose(0, 2, 1, 3).reshape(py, px, 4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            cur = np.nanmean(blocks, axis=2)
    if np.isnan(cur).any():
        cur = np.where(np.isnan(cur), np.float32(np.nanmean(cur)), cur)
    for lvl in reversed(levels):
        ny, nx = lvl.shape
        up = np.repeat(np.repeat(cur, 2, axis=0), 2, axis=1)[:ny, :nx]
        cur = np.where(np.isnan(lvl), up, lvl)
    return cur


def projection_surface(P: np.ndarray, cell: float,
                       frame: Tuple[float, float, float, float] = None
                       ) -> Tuple[np.ndarray, float]:
    """Height grid for orthorectification. Returns (grid, grid_cell).

    frame: (minx, miny, maxx, maxy) the grid must cover/align to — pass the
    output raster's extent so grids from different point sets (mesh samples,
    ground points) share one origin; points outside are clipped to the rim.

    Built at a coarse, stable cell (>= 1 m) as the *top* surface (per-cell
    max — a median falls onto interior floors of hollow buildings), then:
      * local-median outlier knockout (crane clusters, residual flyers)
        before any filling, so their heights cannot spread;
      * smooth diffusion fill for point-free areas (textureless slabs);
      * grey *opening* to strip thin spikes (cranes, poles, outliers);
      * median filter against residual matching ripple.
    A projection surface must be locally smooth: spatially varying height
    error warps the sampled imagery, while a locally constant offset only
    shifts it slightly under a near-nadir camera."""
    from scipy import ndimage
    cell_s = max(cell, 1.0)
    minx, miny, maxx, maxy = frame if frame is not None else extent(P)
    if frame is not None:  # drop points outside the frame, don't clip them in
        keep = ((P[:, 0] >= minx) & (P[:, 0] <= maxx)
                & (P[:, 1] >= miny) & (P[:, 1] <= maxy))
        P = P[keep]
    nx = max(int(np.ceil((maxx - minx) / cell_s)) + 1, 1)
    ny = max(int(np.ceil((maxy - miny) / cell_s)) + 1, 1)
    ix, iy = cell_indices(P, minx, maxy, cell_s, nx, ny)
    z = np.asarray(P[:, 2], np.float32)
    order = np.argsort(z)
    arr = np.full((ny, nx), np.nan, np.float32)
    arr[iy[order], ix[order]] = z[order]          # max: highest written last
    # knock out cells towering over their neighbourhood (crane clusters,
    # residual flyers) BEFORE filling — otherwise the diffusion fill drags
    # their heights across nearby empty areas as phantom plateaus
    med = ndimage.median_filter(fill_nearest(arr.copy()), size=9)
    outlier = ~np.isnan(arr) & (arr - med > 15.0)
    arr[outlier] = np.nan
    arr = fill_smooth(arr)                        # true holes only
    arr = ndimage.grey_opening(arr, size=5)       # thin spikes (cranes, wires)
    arr = ndimage.median_filter(arr, size=3)
    return arr, cell_s





def pick_native_cell(P: np.ndarray, target_cell: float,
                     factor: float = 1.5) -> float:
    """Grid cell the cloud can actually populate: max(target, ~point spacing)."""
    return max(target_cell, median_spacing(P) * factor)


# -- GeoTIFF writing -----------------------------------------------------
def finalize_output(tmp_path: str, out_path: str, log=None) -> str:
    """Move a finished temp raster onto its target name. If the target is
    locked by another program (GIS viewer with the old file open), keep the
    result under a timestamped name instead of losing the computation."""
    import time
    try:
        os.replace(tmp_path, out_path)
        return out_path
    except PermissionError:
        time.sleep(3)
        try:
            os.replace(tmp_path, out_path)
            return out_path
        except PermissionError:
            base, ext = os.path.splitext(out_path)
            alt = f"{base}_{time.strftime('%H%M%S')}{ext}"
            os.replace(tmp_path, alt)
            if log:
                log(f"{os.path.basename(out_path)} is locked by another program "
                    f"(close it in your GIS viewer) — result saved as "
                    f"{os.path.basename(alt)}.", "warn")
            return alt


def geotiff_profile(chunk, origin, cell, nx, ny, count, dtype, nodata):
    import rasterio
    from rasterio.transform import from_origin
    crs = None
    if chunk is not None and getattr(chunk, "epsg", None):
        try:
            crs = rasterio.crs.CRS.from_epsg(chunk.epsg)
        except Exception:
            crs = None
    minx, maxy = origin
    return dict(driver="GTiff", height=ny, width=nx, count=count, dtype=dtype,
                crs=crs, transform=from_origin(minx, maxy, cell, cell),
                compress="lzw", tiled=True, blockxsize=256, blockysize=256,
                nodata=nodata, BIGTIFF="IF_SAFER")


def write_interp_raster(path: str, natives: List[np.ndarray], chunk,
                        origin, native_cell: float, cell: float,
                        nx: int, ny: int, dtype: str = "float32",
                        nodata=None, z_offset: float = 0.0,
                        block: int = 2048, progress=None, log=None) -> str:
    """Write native-resolution band grids as a GeoTIFF at the target cell,
    interpolating block-by-block (bilinear) so memory stays bounded.

    natives: list of (ny_native, nx_native) float32 arrays (already filled).
    z_offset: subtracted from every value (vertical datum shift).
    Returns the final path (may differ if the target was locked).
    """
    import rasterio
    from scipy.ndimage import map_coordinates
    profile = geotiff_profile(chunk, origin, cell, nx, ny,
                              count=len(natives), dtype=dtype, nodata=nodata)
    ratio = cell / native_cell
    tmp = path + ".part.tif"
    with rasterio.open(tmp, "w", **profile) as dst:
        for r0 in range(0, ny, block):
            r1 = min(r0 + block, ny)
            # centre-of-cell coordinates of the target block in native units
            rows = (np.arange(r0, r1) + 0.5) * ratio - 0.5
            cols = (np.arange(0, nx) + 0.5) * ratio - 0.5
            rr, cc = np.meshgrid(rows, cols, indexing="ij")
            coords = np.stack([rr.ravel(), cc.ravel()])
            for b, nat in enumerate(natives, start=1):
                blk = map_coordinates(nat, coords, order=1, mode="nearest")
                blk = blk.reshape(r1 - r0, nx) - z_offset
                if dtype == "uint8":
                    blk = np.clip(np.rint(blk), 0, 255)
                dst.write(blk.astype(dtype),
                          indexes=b, window=((r0, r1), (0, nx)))
            if progress:
                progress(int(100 * r1 / ny))
    return finalize_output(tmp, path, log)
