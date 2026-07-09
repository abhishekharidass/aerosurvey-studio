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


def pick_native_cell(P: np.ndarray, target_cell: float,
                     factor: float = 1.5) -> float:
    """Grid cell the cloud can actually populate: max(target, ~point spacing)."""
    return max(target_cell, median_spacing(P) * factor)


# -- GeoTIFF writing -----------------------------------------------------
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
                        block: int = 2048, progress=None) -> None:
    """Write native-resolution band grids as a GeoTIFF at the target cell,
    interpolating block-by-block (bilinear) so memory stays bounded.

    natives: list of (ny_native, nx_native) float32 arrays (already filled).
    z_offset: subtracted from every value (vertical datum shift).
    """
    import rasterio
    from scipy.ndimage import map_coordinates
    profile = geotiff_profile(chunk, origin, cell, nx, ny,
                              count=len(natives), dtype=dtype, nodata=nodata)
    ratio = cell / native_cell
    with rasterio.open(path, "w", **profile) as dst:
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
