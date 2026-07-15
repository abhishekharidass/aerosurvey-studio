"""Vector contour lines from the DSM/DTM (Pix4D/Agisoft-style deliverable).

Marching squares runs on a lightly smoothed, decimated elevation grid via
contourpy (already present as matplotlib's contouring engine). Writers cover
the three formats clients ask for: Shapefile (GIS), DXF (CAD) and GeoJSON
(web). Every line carries its elevation; every 5th interval is flagged as an
index contour so styling matches survey drawings.
"""
from __future__ import annotations

import json
import math
import os
from typing import Callable, List, Optional, Tuple

import numpy as np

# A contour is (elevation, is_index, [ (N,2) arrays of map-CRS x/y ])
Contour = Tuple[float, bool, List[np.ndarray]]

INDEX_EVERY = 5           # every 5th level is an "index" (major) contour
MAX_GRID_DIM = 6000       # decimate huge DEMs before contouring


def generate_contours(dem_path: str, interval: float,
                      smooth_sigma: float = 1.0,
                      log: Callable[[str, str], None] = None) -> List[Contour]:
    """Trace contour lines at a fixed interval from a DEM GeoTIFF."""
    import contourpy
    import rasterio
    from rasterio.transform import Affine
    from scipy import ndimage

    log = log or (lambda m, lvl="info": None)
    if interval <= 0:
        raise ValueError("contour interval must be positive")

    with rasterio.open(dem_path) as src:
        scale = max(max(src.width, src.height) / float(MAX_GRID_DIM), 1.0)
        h = max(int(src.height / scale), 2)
        w = max(int(src.width / scale), 2)
        z = src.read(1, out_shape=(h, w), masked=True).astype(np.float64)
        transform = src.transform * Affine.scale(src.width / w, src.height / h)

    grid = z.filled(np.nan)
    valid = np.isfinite(grid)
    if valid.sum() < 4:
        log("Contours: DEM has no valid elevations.", "warn")
        return []

    if smooth_sigma > 0:
        # smooth only where data exists (normalised convolution keeps edges)
        filled = np.where(valid, grid, 0.0)
        weight = ndimage.gaussian_filter(valid.astype(np.float64), smooth_sigma)
        blur = ndimage.gaussian_filter(filled, smooth_sigma)
        grid = np.where(valid, blur / np.maximum(weight, 1e-9), np.nan)

    zmin = float(np.nanmin(grid))
    zmax = float(np.nanmax(grid))
    first = math.ceil(zmin / interval) * interval
    levels = np.arange(first, zmax, interval)
    if len(levels) == 0:
        log(f"Contours: relief ({zmax - zmin:.2f} m) is below one interval.", "warn")
        return []
    if len(levels) > 2000:
        raise ValueError(
            f"{len(levels)} levels at {interval} m interval — choose a larger interval")

    gen = contourpy.contour_generator(
        z=np.ma.masked_invalid(grid), name="serial",
        line_type=contourpy.LineType.Separate, corner_mask=True)

    a, b, c, d, e, f = (transform.a, transform.b, transform.c,
                        transform.d, transform.e, transform.f)
    out: List[Contour] = []
    for i, level in enumerate(levels):
        lines = []
        for seg in gen.lines(float(level)):
            if len(seg) < 2:
                continue
            col = seg[:, 0] + 0.5   # contourpy works in array index space;
            row = seg[:, 1] + 0.5   # +0.5 lands on pixel centres
            xy = np.column_stack([a * col + b * row + c,
                                  d * col + e * row + f])
            lines.append(xy)
        if lines:
            is_index = (round(level / interval) % INDEX_EVERY) == 0
            out.append((float(level), is_index, lines))
    n = sum(len(ls) for _, _, ls in out)
    log(f"Traced {n} contour line(s) across {len(out)} level(s) "
        f"({levels[0]:.2f}–{levels[-1]:.2f} m @ {interval} m).", "ok")
    return out


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def write_geojson(contours: List[Contour], path: str,
                  epsg: Optional[int] = None) -> str:
    features = []
    for level, is_index, lines in contours:
        for xy in lines:
            features.append({
                "type": "Feature",
                "properties": {"elevation": round(level, 3),
                               "index": 1 if is_index else 0},
                "geometry": {"type": "LineString",
                             "coordinates": [[round(x, 3), round(y, 3)]
                                             for x, y in xy]},
            })
    doc = {"type": "FeatureCollection", "features": features}
    if epsg:
        doc["crs"] = {"type": "name",
                      "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"}}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
    return path


def write_shapefile(contours: List[Contour], path: str,
                    epsg: Optional[int] = None) -> str:
    import shapefile
    with shapefile.Writer(path, shapeType=shapefile.POLYLINEZ) as shp:
        shp.field("ELEV", "N", decimal=3)
        shp.field("INDEX", "N", decimal=0)
        for level, is_index, lines in contours:
            for xy in lines:
                shp.linez([[(float(x), float(y), float(level)) for x, y in xy]])
                shp.record(round(level, 3), 1 if is_index else 0)
    if epsg:
        try:
            from pyproj import CRS
            from pyproj.enums import WktVersion
            wkt = CRS.from_epsg(epsg).to_wkt(WktVersion.WKT1_ESRI)
            with open(os.path.splitext(path)[0] + ".prj", "w",
                      encoding="utf-8") as fh:
                fh.write(wkt)
        except Exception:
            pass
    return path


def write_dxf(contours: List[Contour], path: str) -> str:
    """Minimal DXF R12: 3D POLYLINEs on CONTOUR_MINOR / CONTOUR_MAJOR layers."""
    lines = ["0", "SECTION", "2", "HEADER",
             "9", "$ACADVER", "1", "AC1009",
             "0", "ENDSEC",
             "0", "SECTION", "2", "TABLES",
             "0", "TABLE", "2", "LAYER", "70", "2"]
    for name, colour in (("CONTOUR_MINOR", "8"), ("CONTOUR_MAJOR", "1")):
        lines += ["0", "LAYER", "2", name, "70", "0", "62", colour,
                  "6", "CONTINUOUS"]
    lines += ["0", "ENDTAB", "0", "ENDSEC", "0", "SECTION", "2", "ENTITIES"]
    for level, is_index, polys in contours:
        layer = "CONTOUR_MAJOR" if is_index else "CONTOUR_MINOR"
        for xy in polys:
            lines += ["0", "POLYLINE", "8", layer, "66", "1", "70", "8"]
            for x, y in xy:
                lines += ["0", "VERTEX", "8", layer,
                          "10", f"{x:.3f}", "20", f"{y:.3f}",
                          "30", f"{level:.3f}", "70", "32"]
            lines += ["0", "SEQEND"]
    lines += ["0", "ENDSEC", "0", "EOF"]
    with open(path, "w", encoding="ascii") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def export_contours(chunk, out_dir: str, interval: float, source: str = "dtm",
                    formats: Tuple[str, ...] = ("shp", "dxf", "geojson"),
                    log: Callable[[str, str], None] = None) -> List[str]:
    """Generate contours from the chunk's DTM (or DSM) and write each format."""
    log = log or (lambda m, lvl="info": None)
    dem = getattr(chunk.outputs, source, "") or ""
    if not dem or not os.path.exists(dem):
        raise FileNotFoundError(
            f"No {source.upper()} available — run the surface stage first.")
    contours = generate_contours(dem, interval, log=log)
    if not contours:
        return []
    epsg = chunk.epsg if chunk.crs_mode != "local" else None
    base = os.path.join(out_dir, f"contours_{source}_{interval:g}m")
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for fmt in formats:
        if fmt == "shp":
            written.append(write_shapefile(contours, base + ".shp", epsg))
        elif fmt == "dxf":
            written.append(write_dxf(contours, base + ".dxf"))
        elif fmt == "geojson":
            written.append(write_geojson(contours, base + ".geojson", epsg))
    for p in written:
        log(f"Wrote {os.path.basename(p)} "
            f"({os.path.getsize(p) / 1e6:.2f} MB)", "ok")
    return written
