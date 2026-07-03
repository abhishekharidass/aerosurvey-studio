"""Pipeline stage definitions.

Each stage mirrors a step in the Metashape/Pix4D workflow. Until the real
engines (COLMAP / OpenMVS / PDAL / GDAL) are wired in, the stages run a
*simulated* implementation that produces genuine, well-formed outputs (a LAS
point cloud, float32 GeoTIFF DSM/DTM/ortho) so the rest of the app is fully
exercisable. Replacing a stage's ``run`` with a real engine call is the only
change needed to go from scaffold to production for that step.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np

from ..model.project import Chunk


# ---------------------------------------------------------------------------
# Stage execution context
# ---------------------------------------------------------------------------
class StageContext:
    """Handed to every stage. Bridges to the worker thread's signals."""

    def __init__(self, chunk: Chunk, workdir: str, log, progress, cancelled):
        self.chunk = chunk
        self.workdir = workdir
        self.log = log                # (msg: str, level: str="info")
        self.progress = progress      # (pct: int 0..100)
        self._cancelled = cancelled   # () -> bool

    @property
    def cancelled(self) -> bool:
        return self._cancelled()

    def sleep(self, secs: float) -> bool:
        """Cancellable sleep used to pace the simulation. Returns False if cancelled."""
        end = time.time() + secs
        while time.time() < end:
            if self.cancelled:
                return False
            time.sleep(0.02)
        return not self.cancelled


# ---------------------------------------------------------------------------
# Synthetic scene generation (stands in for real MVS output)
# ---------------------------------------------------------------------------
DOMAIN = 200.0  # metres, scene is DOMAIN x DOMAIN


def _synth_scene(seed: int = 7):
    rng = np.random.default_rng(seed)

    def ground_z(x, y):
        return 12 + 3 * np.sin(x / 34) + 2.2 * np.cos(y / 27) + 0.6 * np.sin((x + y) / 15)

    pts, cols, cls = [], [], []

    # Ground
    n_g = 55000
    gx = rng.uniform(0, DOMAIN, n_g)
    gy = rng.uniform(0, DOMAIN, n_g)
    gz = ground_z(gx, gy) + rng.normal(0, 0.04, n_g)
    pts.append(np.column_stack([gx, gy, gz]))
    base = np.array([150, 121, 78])
    cols.append((base + rng.normal(0, 10, (n_g, 3))).clip(0, 255))
    cls.append(np.full(n_g, 2, np.uint8))  # ground truth (recomputed in classify)

    # Buildings (flat-topped prisms)
    for (bx, by, w, h, height) in [(60, 55, 26, 20, 11),
                                    (130, 120, 30, 24, 8),
                                    (95, 150, 18, 18, 14)]:
        n_b = 4200
        px = rng.uniform(bx, bx + w, n_b)
        py = rng.uniform(by, by + h, n_b)
        top = ground_z(bx + w / 2, by + h / 2) + height
        pz = top + rng.normal(0, 0.06, n_b)
        pts.append(np.column_stack([px, py, pz]))
        cols.append((np.array([182, 184, 190]) + rng.normal(0, 8, (n_b, 3))).clip(0, 255))
        cls.append(np.full(n_b, 6, np.uint8))

    # Trees (green blobs)
    n_clusters = 40
    for _ in range(n_clusters):
        cx, cy = rng.uniform(10, DOMAIN - 10, 2)
        n_t = rng.integers(250, 600)
        tx = cx + rng.normal(0, 3, n_t)
        ty = cy + rng.normal(0, 3, n_t)
        th = rng.uniform(3, 9)
        tz = ground_z(cx, cy) + rng.uniform(0.5, th, n_t)
        pts.append(np.column_stack([tx, ty, tz]))
        cols.append((np.array([48, 122, 46]) + rng.normal(0, 14, (n_t, 3))).clip(0, 255))
        cls.append(np.full(n_t, 5, np.uint8))

    P = np.vstack(pts)
    C = np.vstack(cols).astype(np.uint8)
    K = np.concatenate(cls)
    return P, C, K


def _write_las(path: str, P: np.ndarray, C: np.ndarray, cls: np.ndarray) -> None:
    import laspy
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = P.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    las.x, las.y, las.z = P[:, 0], P[:, 1], P[:, 2]
    las.red = (C[:, 0].astype(np.uint16)) * 257
    las.green = (C[:, 1].astype(np.uint16)) * 257
    las.blue = (C[:, 2].astype(np.uint16)) * 257
    las.classification = cls
    las.write(path)


def _grid(P, values, cell=1.0, reducer="max"):
    """Rasterise scattered points to a regular grid. Returns (arr, transform_origin)."""
    nx = ny = int(DOMAIN / cell)
    ix = np.clip(((P[:, 0]) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((P[:, 1]) / cell).astype(int), 0, ny - 1)
    arr = np.full((ny, nx), np.nan, np.float32)
    # row 0 = top (north) so flip Y
    row = ny - 1 - iy
    order = np.argsort(values) if reducer == "max" else np.argsort(-values)
    for r, c, v in zip(row[order], ix[order], values[order]):
        arr[r, c] = v  # last write wins -> extreme value for chosen reducer
    return arr, cell, nx, ny


def _fill_nan(arr: np.ndarray) -> np.ndarray:
    """Cheap hole-fill by nearest finite neighbour via iterative dilation."""
    out = arr.copy()
    mask = np.isnan(out)
    if not mask.any():
        return out
    for _ in range(25):
        if not np.isnan(out).any():
            break
        shifted = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            s = np.full_like(out, np.nan)
            r0 = max(dr, 0); r1 = out.shape[0] + min(dr, 0)
            c0 = max(dc, 0); c1 = out.shape[1] + min(dc, 0)
            s[r0:r1, c0:c1] = out[max(-dr, 0):out.shape[0] - max(dr, 0) or None,
                                  max(-dc, 0):out.shape[1] - max(dc, 0) or None]
            shifted.append(s)
        stack = np.stack(shifted)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean = np.nanmean(stack, axis=0)
        take = np.isnan(out) & ~np.isnan(mean)
        out[take] = mean[take]
    return out


def _write_geotiff(path, arr, chunk: Chunk, bands=1, cell=1.0):
    import rasterio
    from rasterio.transform import from_origin
    epsg = chunk.epsg
    crs = None
    if epsg:
        try:
            crs = rasterio.crs.CRS.from_epsg(epsg)
        except Exception:
            crs = None
    transform = from_origin(0, DOMAIN, cell, cell)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    profile = dict(driver="GTiff", height=arr.shape[1], width=arr.shape[2],
                   count=arr.shape[0], dtype="float32", crs=crs, transform=transform,
                   compress="lzw", nodata=(np.nan if bands == 1 else None))
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32))


def _load_cloud(path):
    import laspy
    las = laspy.read(path)
    P = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
    C = np.column_stack([las.red, las.green, las.blue]) // 257
    cls = np.array(las.classification)
    return P, C.astype(np.uint8), cls.astype(np.uint8)


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------
def run_align(ctx: StageContext) -> bool:
    ch = ctx.chunk
    cams = [c for c in ch.cameras if c.enabled]
    ctx.log(f"Detecting features on {len(cams)} images (SIFT-equivalent)...", "info")
    for i, cam in enumerate(cams):
        if not ctx.sleep(0.03):
            return False
        ctx.progress(int(30 * (i + 1) / max(len(cams), 1)))
    ctx.log("Matching features across overlapping pairs...", "info")
    if not ctx.sleep(0.6):
        return False
    ctx.progress(55)
    ctx.log("Bundle adjustment (Levenberg-Marquardt on sparse Jacobian)...", "info")
    if not ctx.sleep(0.8):
        return False
    # Assign estimated positions from geotags (already in project CRS via ingest) or synthetic
    rng = np.random.default_rng(1)
    for cam in cams:
        cam.est_x = cam.est_x if cam.est_x is not None else float(rng.uniform(0, DOMAIN))
        cam.est_y = cam.est_y if cam.est_y is not None else float(rng.uniform(0, DOMAIN))
        cam.est_z = cam.est_z if cam.est_z is not None else float(rng.uniform(60, 90))
        cam.aligned = True
    ch.aligned = True
    err = np.abs(rng.normal(0, 0.35, 4000))
    ctx.log(f"Aligned {len(cams)} cameras. Mean reprojection error "
            f"{err.mean():.3f} px (Gaussian, sigma {err.std():.3f}).", "ok")
    ctx.progress(100)
    return True


def run_dense(ctx: StageContext) -> bool:
    if not ctx.chunk.aligned:
        ctx.log("Cameras are not aligned. Run 'Align Photos' first.", "error")
        return False
    ctx.log("Computing per-pixel depth maps (Multi-View Stereo)...", "info")
    if not ctx.sleep(1.0):
        return False
    ctx.progress(45)
    ctx.log("Fusing depth maps into dense point cloud...", "info")
    P, C, K = _synth_scene()
    if not ctx.sleep(0.6):
        return False
    # store unclassified initially
    K0 = np.ones_like(K)  # class 1 = unclassified
    out = os.path.join(ctx.workdir, "dense_cloud.las")
    _write_las(out, P, C, K0)
    ctx.chunk.outputs.dense_cloud = out
    ctx.progress(100)
    ctx.log(f"Dense cloud: {len(P):,} points -> {out}", "ok")
    return True


def run_classify(ctx: StageContext) -> bool:
    path = ctx.chunk.outputs.dense_cloud
    if not path or not os.path.exists(path):
        ctx.log("No dense cloud to classify. Run 'Build Dense Cloud' first.", "error")
        return False
    ctx.log("Classifying points (ground filter + height/colour rules)...", "info")
    P, C, _ = _load_cloud(path)
    if not ctx.sleep(0.5):
        return False
    # Ground surface = min Z on a coarse grid
    cell = 5.0
    nx = int(DOMAIN / cell)
    ix = np.clip((P[:, 0] / cell).astype(int), 0, nx - 1)
    iy = np.clip((P[:, 1] / cell).astype(int), 0, nx - 1)
    key = iy * nx + ix
    ground = np.full(nx * nx, np.inf)
    np.minimum.at(ground, key, P[:, 2])
    height = P[:, 2] - ground[key]
    cls = np.ones(len(P), np.uint8)
    cls[height < 0.6] = 2                       # ground
    non = height >= 0.6
    green = (C[:, 1].astype(int) > C[:, 0].astype(int) + 12) & \
            (C[:, 1].astype(int) > C[:, 2].astype(int) + 12)
    cls[non & green] = 5                         # high vegetation
    cls[non & ~green] = 6                        # building
    ctx.progress(70)
    out = os.path.join(ctx.workdir, "classified_cloud.las")
    _write_las(out, P, C, cls)
    ctx.chunk.outputs.classified_cloud = out
    counts = {int(k): int((cls == k).sum()) for k in np.unique(cls)}
    ctx.log(f"Classified: {counts} -> {out}", "ok")
    ctx.progress(100)
    return True


def _cloud_for_surfaces(ch: Chunk):
    path = ch.outputs.classified_cloud or ch.outputs.dense_cloud
    return path


def run_dsm(ctx: StageContext) -> bool:
    path = _cloud_for_surfaces(ctx.chunk)
    if not path or not os.path.exists(path):
        ctx.log("No point cloud available for DSM.", "error")
        return False
    ctx.log("Rasterising DSM (max Z per cell)...", "info")
    P, _, _ = _load_cloud(path)
    arr, cell, nx, ny = _grid(P, P[:, 2], cell=1.0, reducer="max")
    arr = _fill_nan(arr)
    if not ctx.sleep(0.4):
        return False
    out = os.path.join(ctx.workdir, "dsm.tif")
    _write_geotiff(out, arr, ctx.chunk, bands=1, cell=cell)
    ctx.chunk.outputs.dsm = out
    ctx.log(f"DSM {nx}x{ny} @ {cell} m -> {out}", "ok")
    ctx.progress(100)
    return True


def run_dtm(ctx: StageContext) -> bool:
    path = ctx.chunk.outputs.classified_cloud
    if not path or not os.path.exists(path):
        ctx.log("DTM needs a classified cloud. Run 'Classify Points' first.", "error")
        return False
    ctx.log("Rasterising DTM from ground-classified points (min Z)...", "info")
    P, _, cls = _load_cloud(path)
    ground = P[cls == 2]
    if len(ground) == 0:
        ctx.log("No ground points found.", "error")
        return False
    arr, cell, nx, ny = _grid(ground, ground[:, 2], cell=1.0, reducer="min")
    arr = _fill_nan(arr)
    if not ctx.sleep(0.4):
        return False
    out = os.path.join(ctx.workdir, "dtm.tif")
    _write_geotiff(out, arr, ctx.chunk, bands=1, cell=cell)
    ctx.chunk.outputs.dtm = out
    ctx.log(f"DTM {nx}x{ny} @ {cell} m -> {out}", "ok")
    ctx.progress(100)
    return True


def run_ortho(ctx: StageContext) -> bool:
    path = _cloud_for_surfaces(ctx.chunk)
    if not path or not os.path.exists(path):
        ctx.log("No point cloud available for orthomosaic.", "error")
        return False
    ctx.log("Projecting images onto DSM and mosaicking (nadir seamlines)...", "info")
    P, C, _ = _load_cloud(path)
    cell = 0.5
    nx = ny = int(DOMAIN / cell)
    ix = np.clip((P[:, 0] / cell).astype(int), 0, nx - 1)
    iy = np.clip((P[:, 1] / cell).astype(int), 0, ny - 1)
    row = ny - 1 - iy
    order = np.argsort(P[:, 2])  # highest last -> top surface wins
    rgb = np.zeros((3, ny, nx), np.float32)
    for b in range(3):
        band = np.full((ny, nx), np.nan, np.float32)
        for r, c, v in zip(row[order], ix[order], C[order, b].astype(np.float32)):
            band[r, c] = v
        rgb[b] = _fill_nan(band)
        if not ctx.sleep(0.15):
            return False
        ctx.progress(30 + b * 20)
    out = os.path.join(ctx.workdir, "orthomosaic.tif")
    _write_geotiff(out, rgb, ctx.chunk, bands=3, cell=cell)
    ctx.chunk.outputs.orthomosaic = out
    ctx.log(f"Orthomosaic {nx}x{ny} @ {cell} m (3-band RGB) -> {out}", "ok")
    ctx.progress(100)
    return True


@dataclass
class Stage:
    key: str
    name: str
    engine: str
    run: Callable[[StageContext], bool]
    produces: str = ""


PIPELINE: List[Stage] = [
    Stage("align", "Align Photos", "colmap", run_align, "Sparse cloud + camera poses"),
    Stage("dense", "Build Dense Cloud", "openmvs", run_dense, "Dense point cloud (LAS)"),
    Stage("classify", "Classify Points", "pdal", run_classify, "Classified cloud"),
    Stage("dsm", "Build DSM", "gdal", run_dsm, "Digital Surface Model (GeoTIFF)"),
    Stage("dtm", "Build DTM", "gdal", run_dtm, "Digital Terrain Model (GeoTIFF)"),
    Stage("ortho", "Build Orthomosaic", "gdal", run_ortho, "Orthomosaic (GeoTIFF)"),
]


def stage_by_key(key: str) -> Stage:
    return next(s for s in PIPELINE if s.key == key)
