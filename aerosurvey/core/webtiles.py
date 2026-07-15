"""XYZ web tiles + Google Earth KML superoverlay from the orthomosaic.

Mirrors gdal2tiles: the raster is warped to web mercator (WarpedVRT, so any
window streams without loading the mosaic), cut into 256 px z/x/y PNG tiles,
and optionally wrapped in a KML superoverlay — per-tile KML files with
Region-based level of detail, so Google Earth streams only what is visible.
Tiles use the XYZ scheme (y from north) shared by OSM/MapLibre/Leaflet.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Optional, Set, Tuple

import numpy as np

from . import webmercator as wm

TILE = wm.TILE_PX


def _lonlat_box(tx: int, ty: int, z: int) -> Tuple[float, float, float, float]:
    """Tile -> (west, south, east, north) in WGS84 degrees."""
    w, s, e, n = wm.tile_bounds(tx, ty, z)
    lon0, lat0 = wm.mercator_to_lonlat(w, s)
    lon1, lat1 = wm.mercator_to_lonlat(e, n)
    return lon0, lat0, lon1, lat1


def export_web_tiles(raster_path: str, out_dir: str,
                     min_zoom: Optional[int] = None,
                     max_zoom: Optional[int] = None,
                     kml: bool = True, name: str = "Orthomosaic",
                     log: Callable[[str, str], None] = None,
                     cancelled: Callable[[], bool] = None) -> dict:
    """Write z/x/y.png tiles (+ optional KML superoverlay). Returns metadata."""
    import rasterio
    from PIL import Image
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    log = log or (lambda m, lvl="info": None)
    cancelled = cancelled or (lambda: False)

    from rasterio.transform import Affine
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError("raster has no CRS — georeference the project first")
        left, bottom, right, top = transform_bounds(src.crs, "EPSG:3857",
                                                    *src.bounds)
        native_res = (right - left) / src.width
        if max_zoom is None:
            max_zoom = wm.zoom_for_resolution(native_res, 1, 22)
        if min_zoom is None:
            # coarsest level where the extent still spans >= 1 tile side
            span = max(right - left, top - bottom)
            min_zoom = max(min(int(np.log2(2 * wm.ORIGIN / span)), max_zoom), 1)
        has_alpha = src.count >= 4

        written: Set[Tuple[int, int, int]] = set()
        total = 0
        for z in range(min_zoom, max_zoom + 1):
            # a VRT aligned to this zoom's tile grid: every tile is an exact
            # in-bounds 256x256 window (the gdal2tiles construction)
            tiles = wm.tiles_in_bounds(left, bottom, right, top, z, cap=4 ** 12)
            tx0 = min(t[0] for t in tiles)
            ty0 = min(t[1] for t in tiles)
            grid_w = (max(t[0] for t in tiles) - tx0 + 1) * TILE
            grid_h = (max(t[1] for t in tiles) - ty0 + 1) * TILE
            res = wm.resolution(z)
            gw, _, _, gn = wm.tile_bounds(tx0, ty0, z)
            vrt_tr = Affine(res, 0, gw, 0, -res, gn)
            with WarpedVRT(src, crs="EPSG:3857", transform=vrt_tr,
                           width=grid_w, height=grid_h,
                           resampling=Resampling.bilinear) as vrt:
                for tx, ty in tiles:
                    if cancelled():
                        log("Tile export cancelled.", "warn")
                        return {"tiles": total, "cancelled": True}
                    win = Window((tx - tx0) * TILE, (ty - ty0) * TILE,
                                 TILE, TILE)
                    data = vrt.read(window=win)
                    rgba = np.zeros((TILE, TILE, 4), np.uint8)
                    rgba[..., :3] = np.transpose(
                        data[:3] if src.count >= 3 else
                        np.repeat(data[:1], 3, axis=0), (1, 2, 0))
                    if has_alpha:
                        rgba[..., 3] = data[3]
                    else:
                        rgba[..., 3] = np.where(rgba[..., :3].any(axis=2), 255, 0)
                    if not rgba[..., 3].any():
                        continue        # fully transparent tile
                    d = os.path.join(out_dir, str(z), str(tx))
                    os.makedirs(d, exist_ok=True)
                    Image.fromarray(rgba).save(os.path.join(d, f"{ty}.png"))
                    written.add((z, tx, ty))
                    total += 1
            log(f"Zoom {z}: {sum(1 for k in written if k[0] == z)} tile(s).",
                "info")

    meta = {"format": "xyz", "min_zoom": min_zoom, "max_zoom": max_zoom,
            "tiles": total, "bounds_3857": [left, bottom, right, top],
            "tile_size": TILE, "source": os.path.basename(raster_path)}
    with open(os.path.join(out_dir, "tiles.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    if kml and total:
        _write_kml_superoverlay(out_dir, written, min_zoom, max_zoom, name)
        log("KML superoverlay written (open doc.kml in Google Earth).", "ok")
    log(f"Web tiles: {total} tiles, zoom {min_zoom}-{max_zoom} -> {out_dir}",
        "ok")
    return meta


# ---------------------------------------------------------------------------
# KML superoverlay (gdal2tiles-compatible layout)
# ---------------------------------------------------------------------------
def _kml_header(name: str) -> str:
    return ('<?xml version="1.0" encoding="utf-8"?>\n'
            '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
            f'<Document><name>{name}</name>')


def _region(box, min_px: int, max_px: int) -> str:
    w, s, e, n = box
    return (f"<Region><LatLonAltBox><north>{n:.10f}</north><south>{s:.10f}</south>"
            f"<east>{e:.10f}</east><west>{w:.10f}</west></LatLonAltBox>"
            f"<Lod><minLodPixels>{min_px}</minLodPixels>"
            f"<maxLodPixels>{max_px}</maxLodPixels></Lod></Region>")


def _write_kml_superoverlay(out_dir: str, written: Set[Tuple[int, int, int]],
                            min_zoom: int, max_zoom: int, name: str) -> None:
    for z, tx, ty in sorted(written):
        box = _lonlat_box(tx, ty, z)
        w, s, e, n = box
        parts = [_kml_header(f"{z}/{tx}/{ty}"),
                 _region(box, 64 if z > min_zoom else 0,
                         -1 if z == max_zoom else 2048),
                 f"<GroundOverlay><drawOrder>{z}</drawOrder>"
                 f"<Icon><href>{ty}.png</href></Icon>"
                 f"<LatLonBox><north>{n:.10f}</north><south>{s:.10f}</south>"
                 f"<east>{e:.10f}</east><west>{w:.10f}</west></LatLonBox>"
                 "</GroundOverlay>"]
        for cx in (2 * tx, 2 * tx + 1):          # stream children on zoom-in
            for cy in (2 * ty, 2 * ty + 1):
                if (z + 1, cx, cy) in written:
                    cbox = _lonlat_box(cx, cy, z + 1)
                    parts.append(
                        f"<NetworkLink><name>{z + 1}/{cx}/{cy}</name>"
                        + _region(cbox, 64, -1)
                        + f"<Link><href>../../{z + 1}/{cx}/{cy}.kml</href>"
                        "<viewRefreshMode>onRegion</viewRefreshMode></Link>"
                        "</NetworkLink>")
        parts.append("</Document></kml>")
        with open(os.path.join(out_dir, str(z), str(tx), f"{ty}.kml"), "w",
                  encoding="utf-8") as fh:
            fh.write("\n".join(parts))

    roots = [k for k in sorted(written) if k[0] == min_zoom]
    parts = [_kml_header(name)]
    for z, tx, ty in roots:
        box = _lonlat_box(tx, ty, z)
        parts.append(f"<NetworkLink><name>{z}/{tx}/{ty}</name>"
                     + _region(box, 0, -1)
                     + f"<Link><href>{z}/{tx}/{ty}.kml</href>"
                     "<viewRefreshMode>onRegion</viewRefreshMode></Link>"
                     "</NetworkLink>")
    parts.append("</Document></kml>")
    with open(os.path.join(out_dir, "doc.kml"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
