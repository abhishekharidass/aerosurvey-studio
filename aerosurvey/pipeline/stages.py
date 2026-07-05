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


def _extent(P):
    """(minx, miny, maxx, maxy) of a point set."""
    return (float(P[:, 0].min()), float(P[:, 1].min()),
            float(P[:, 0].max()), float(P[:, 1].max()))


def _cell_indices(P, minx, maxy, cell, nx, ny):
    """Column/row indices for each point (row 0 = north/top)."""
    ix = np.clip(((P[:, 0] - minx) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((maxy - P[:, 1]) / cell).astype(int), 0, ny - 1)
    return ix, iy


def _grid(P, values, cell, reducer="max"):
    """Rasterise scattered points over their own extent. Returns (arr, (minx, maxy), nx, ny)."""
    minx, miny, maxx, maxy = _extent(P)
    nx = max(int(np.ceil((maxx - minx) / cell)) + 1, 1)
    ny = max(int(np.ceil((maxy - miny) / cell)) + 1, 1)
    ix, iy = _cell_indices(P, minx, maxy, cell, nx, ny)
    arr = np.full((ny, nx), np.nan, np.float32)
    # ascending sort -> largest written last (max); reverse for min. Last write wins.
    order = np.argsort(values) if reducer == "max" else np.argsort(-values)
    arr[iy[order], ix[order]] = np.asarray(values, np.float32)[order]
    return arr, (minx, maxy), nx, ny


def _raster_cell(P, target_px=1000, min_cell=0.02):
    """Pick a ground sample distance so the longer side is ~target_px cells."""
    minx, miny, maxx, maxy = _extent(P)
    span = max(maxx - minx, maxy - miny, 1.0)
    return max(span / target_px, min_cell)


def _dsm_cell(P):
    return _raster_cell(P, target_px=800)


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


def _write_geotiff(path, arr, chunk: Chunk, origin, cell):
    """Write a float32 GeoTIFF. origin = (minx, maxy) top-left corner."""
    import rasterio
    from rasterio.transform import from_origin
    crs = None
    if chunk.epsg:
        try:
            crs = rasterio.crs.CRS.from_epsg(chunk.epsg)
        except Exception:
            crs = None
    minx, maxy = origin
    transform = from_origin(minx, maxy, cell, cell)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    profile = dict(driver="GTiff", height=arr.shape[1], width=arr.shape[2],
                   count=arr.shape[0], dtype="float32", crs=crs, transform=transform,
                   compress="lzw", nodata=(np.nan if arr.shape[0] == 1 else None))
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
    """Aerial triangulation. Uses COLMAP if installed, else the simulation."""
    from . import colmap
    cams = [c for c in ctx.chunk.cameras if c.enabled]
    if not cams:
        ctx.log("No enabled cameras to align.", "error")
        return False
    if colmap.available():
        ctx.log(f"COLMAP detected at {colmap.exe()}", "info")
        return _align_colmap(ctx, cams, colmap)
    ctx.log("COLMAP not found on PATH — using built-in simulation for alignment.", "warn")
    return _align_simulated(ctx, cams)


def _align_colmap(ctx: StageContext, cams, colmap) -> bool:
    image_paths = [c.path for c in cams if os.path.exists(c.path)]
    if len(cams) - len(image_paths):
        ctx.log(f"{len(cams) - len(image_paths)} image file(s) missing on disk; excluded.", "warn")
    if len(image_paths) < 2:
        ctx.log("Need at least 2 images on disk for COLMAP SfM.", "error")
        return False
    ctx.log(f"Structure-from-Motion on {len(image_paths)} images (feature extraction, "
            "matching, incremental mapping)...", "info")
    res = colmap.run_sfm(image_paths, ctx.workdir, ctx)
    if res is None:
        return False
    ctx.chunk._colmap_result = res  # transient: consumed by the georef stage

    by_name = {c.filename: c for c in cams}
    matched = 0
    for name, pose in res.poses.items():
        cam = by_name.get(name)
        if cam is not None:
            cam.est_x, cam.est_y, cam.est_z = (float(pose.center[0]),
                                               float(pose.center[1]),
                                               float(pose.center[2]))
            cam.aligned = True
            matched += 1
    ctx.chunk.aligned = matched > 0

    if len(res.points):
        out = os.path.join(ctx.workdir, "sparse_cloud.las")
        _write_las(out, res.points, res.colors, np.ones(len(res.points), np.uint8))
        ctx.chunk.outputs.sparse_cloud = out
        ctx.log(f"Sparse cloud: {len(res.points):,} tie points -> {out}", "info")

    ctx.log(f"COLMAP aligned {matched}/{len(cams)} cameras "
            f"(mean reprojection error {res.mean_reproj_error:.3f} px).", "ok")
    ctx.log("Note: poses are in COLMAP's local frame; georeferencing to GCPs/GPS is a "
            "later step (model_aligner + GCP-constrained bundle adjustment).", "info")
    ctx.progress(100)
    return matched > 0


def _align_simulated(ctx: StageContext, cams) -> bool:
    ctx.log(f"Detecting features on {len(cams)} images (SIFT-equivalent)...", "info")
    for i, _cam in enumerate(cams):
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
    rng = np.random.default_rng(1)
    for cam in cams:
        cam.est_x = cam.est_x if cam.est_x is not None else float(rng.uniform(0, DOMAIN))
        cam.est_y = cam.est_y if cam.est_y is not None else float(rng.uniform(0, DOMAIN))
        cam.est_z = cam.est_z if cam.est_z is not None else float(rng.uniform(60, 90))
        cam.aligned = True
    ctx.chunk.aligned = True
    err = np.abs(rng.normal(0, 0.35, 4000))
    ctx.log(f"Aligned {len(cams)} cameras. Mean reprojection error "
            f"{err.mean():.3f} px (Gaussian, sigma {err.std():.3f}).", "ok")
    ctx.progress(100)
    return True


def _transform_las(path: str, sim) -> None:
    P, C, cls = _load_cloud(path)
    _write_las(path, sim.apply(P), C, cls)


def run_georef(ctx: StageContext) -> bool:
    """Optimize / georeference: fit the reconstruction into the project CRS,
    preferring surveyed GCPs and falling back to camera GPS."""
    from . import georef
    from ..core import crs as crsmod
    ch = ctx.chunk
    if not ch.aligned:
        ctx.log("Run 'Align Photos' before georeferencing.", "error")
        return False
    for g in ch.gcps:
        g.error = None

    tf = crsmod.CrsTransform(ch.epsg if ch.crs_mode != "local" else None)
    res = getattr(ch, "_colmap_result", None)
    sim = None

    if res is not None and ch.gcps:
        sim = _georef_from_gcps(ctx, ch, res, georef)   # accurate, real reconstruction
    if sim is None:
        sim = _georef_from_gps(ctx, ch, tf, georef)     # GPS similarity fallback
    if sim is None:
        ctx.log("Not enough control to georeference (need >=3 geotagged aligned "
                "cameras, or >=3 triangulable control GCPs).", "error")
        return False

    for c in ch.cameras:
        if c.est_x is not None:
            x, y, z = sim.apply([c.est_x, c.est_y, c.est_z])
            c.est_x, c.est_y, c.est_z = float(x), float(y), float(z)
    if ch.outputs.sparse_cloud and os.path.exists(ch.outputs.sparse_cloud):
        _transform_las(ch.outputs.sparse_cloud, sim)
        ctx.log("Sparse cloud transformed into project CRS.", "info")

    ch._georef_sim = sim  # transient: applied to the dense cloud (still local on disk)
    if res is not None and getattr(res, "observations", None):
        try:
            _refine_bundle(ctx, ch, res, sim)
        except Exception as exc:
            ctx.log(f"Bundle adjustment skipped ({exc}).", "warn")
    ch.optimized = True
    ctx.progress(100)
    return True


def _refine_bundle(ctx: StageContext, ch, res, sim) -> None:
    """GCP-constrained bundle adjustment: refine world camera poses + tie points
    so both tie-point and GCP marks reproject consistently. GCPs are held fixed."""
    from scipy.spatial.transform import Rotation
    from . import bundle, colmap
    if not res.intrinsics or not res.poses:
        return

    Rs, s, ts = sim.R, sim.s, sim.t
    cam_index, rvecs, tvecs, Ks = {}, [], [], []
    for name, pose in res.poses.items():
        K = res.intrinsics.get(pose.camera_id)
        if K is None:
            continue
        Rw = colmap.qvec2rotmat(pose.qvec) @ Rs.T           # local pose -> world pose
        tw = s * np.asarray(pose.tvec) - Rw @ ts
        cam_index[name] = len(rvecs)
        rvecs.append(Rotation.from_matrix(Rw).as_rotvec())
        tvecs.append(tw)
        Ks.append(K)
    if len(rvecs) < 2:
        return

    # tie points (subsampled) into the world frame, all free
    rng = np.random.default_rng(0)
    tie = sim.apply(res.points) if len(res.points) else np.zeros((0, 3))
    keep = (rng.choice(len(tie), 4000, replace=False) if len(tie) > 4000
            else np.arange(len(tie)))
    tie = tie[keep]
    pid_row = {int(pid): i for i, pid in enumerate(res.point_ids[keep])}

    # GCP points, fixed at surveyed world coords
    gcps = [g for g in ch.gcps if g.enabled and g.observations]
    gcp_row = {g.id: len(tie) + i for i, g in enumerate(gcps)}
    gcp_world = np.array([[g.x, g.y, g.z] for g in gcps]) if gcps else np.zeros((0, 3))
    points = np.vstack([tie, gcp_world]) if len(gcp_world) else tie
    fixed = np.concatenate([np.zeros(len(tie), bool), np.ones(len(gcp_world), bool)])

    obs = [[cam_index[n], pid_row[p], u, v] for (n, p, u, v) in res.observations
           if n in cam_index and p in pid_row]
    id2cam = {c.id: c for c in ch.cameras}
    n_gcp_obs = 0
    for g in gcps:
        row = gcp_row[g.id]
        for cam_id, ob in g.observations.items():
            cam = id2cam.get(cam_id)
            ci = cam_index.get(cam.filename) if cam else None
            if ci is not None:
                obs.append([ci, row, ob.px, ob.py])
                n_gcp_obs += 1
    obs = np.array(obs, dtype=float)

    if len(gcp_world) < 3 or n_gcp_obs < 6 or len(obs) < 6 * len(rvecs):
        ctx.log("Bundle adjustment skipped (need >=3 fixed GCPs with enough marks).", "info")
        return

    ctx.log(f"GCP-constrained bundle adjustment: {len(rvecs)} cameras, {len(points)} "
            f"points ({len(gcp_world)} fixed GCPs), {len(obs)} observations...", "info")
    # Centre the problem: world coords are ~1e6 (UTM), which ill-conditions the solve.
    rvecs = np.array(rvecs)
    tvecs = np.array(tvecs)
    offset = points.mean(axis=0)
    points_c = points - offset
    tvecs_c = np.array([tvecs[i] + Rotation.from_rotvec(rvecs[i]).as_matrix() @ offset
                        for i in range(len(rvecs))])

    # Self-calibrate a shared intrinsic model only when it is well-conditioned:
    # a single camera model and plenty of observations. Otherwise hold K fixed.
    cam_ids = {res.poses[n].camera_id for n in cam_index}
    shared, refine_intr = None, False
    if len(cam_ids) == 1 and len(obs) > 15 * len(rvecs):
        K0 = np.asarray(Ks[0], float)
        shared = np.array([(K0[0, 0] + K0[1, 1]) / 2.0, K0[0, 2], K0[1, 2], 0.0])
        refine_intr = True

    r = bundle.bundle_adjust(rvecs, tvecs_c, np.array(Ks), points_c, fixed, obs,
                             refine_points=True, shared_intrinsics=shared,
                             refine_intrinsics=refine_intr)
    name_to_cam = {c.filename: c for c in ch.cameras}
    for name, ci in cam_index.items():
        cam = name_to_cam.get(name)
        if cam is not None:
            center = bundle.center_from_rt(r.rvecs[ci], r.tvecs[ci]) + offset
            cam.est_x, cam.est_y, cam.est_z = float(center[0]), float(center[1]), float(center[2])
    ctx.log(f"Bundle adjustment: reprojection RMSE {r.rmse_before:.3f} -> "
            f"{r.rmse_after:.3f} px over {r.n_obs} observations.", "ok")
    if r.intrinsics is not None:
        ctx.log(f"Self-calibration: focal {shared[0]:.1f} -> {r.intrinsics[0]:.1f} px, "
                f"principal ({r.intrinsics[1]:.1f}, {r.intrinsics[2]:.1f}), "
                f"k1 {r.intrinsics[3]:+.5f}.", "info")


def _georef_from_gps(ctx, ch, tf, georef):
    local, world = [], []
    for c in ch.cameras:
        if c.aligned and c.est_x is not None and c.has_geotag:
            local.append([c.est_x, c.est_y, c.est_z])
            world.append(list(tf.forward(c.lon, c.lat, c.alt or 0.0)))
    if len(local) < 3:
        return None
    sim = georef.umeyama_similarity(np.array(local), np.array(world))
    fit = georef.residuals(np.array(local), np.array(world), sim)
    ctx.log(f"GPS georeferencing from {len(local)} cameras: RMSE {fit.rmse:.3f} m "
            f"(X {fit.rmse_axis[0]:.3f}, Y {fit.rmse_axis[1]:.3f}, Z {fit.rmse_axis[2]:.3f}).",
            "ok")
    return sim


def _georef_from_gcps(ctx, ch, res, georef):
    pose_by_cam = {c.id: res.poses[c.filename] for c in ch.cameras
                   if c.filename in res.poses}
    local, world, labels, checks = [], [], [], []
    for g in ch.gcps:
        if not g.enabled:
            continue
        proj, uvs = [], []
        for cam_id, obs in g.observations.items():
            pose = pose_by_cam.get(cam_id)
            if pose is None:
                continue
            P = res.projection(pose)
            if P is None:
                continue
            proj.append(P)
            uvs.append((obs.px, obs.py))
        if len(proj) < 2:  # need >=2 rays to triangulate
            continue
        local.append(georef.dlt_triangulate(proj, uvs))
        world.append([g.x, g.y, g.z])
        labels.append(g)
        checks.append(g.is_check)
    control = [i for i, chk in enumerate(checks) if not chk]
    if len(control) < 3:
        if labels:
            ctx.log(f"Only {len(control)} triangulable control GCP(s) (need >=3); "
                    "using GPS instead.", "warn")
        return None
    L, W, ctrl = np.array(local), np.array(world), np.array(control)
    sim = georef.umeyama_similarity(L[ctrl], W[ctrl])
    fit = georef.residuals(L, W, sim)
    for g, e in zip(labels, fit.per_point):
        g.error = float(e)
    ctx.log(f"GCP georeferencing: {len(control)} control, {len(checks) - len(control)} check. "
            f"Control RMSE {np.sqrt(np.mean(fit.per_point[ctrl] ** 2)):.3f} m.", "ok")
    check = np.array([i for i, chk in enumerate(checks) if chk], dtype=int)
    if len(check):
        ctx.log(f"Independent check-point RMSE "
                f"{np.sqrt(np.mean(fit.per_point[check] ** 2)):.3f} m.", "ok")
    return sim


def run_dense(ctx: StageContext) -> bool:
    """Dense reconstruction. Uses OpenMVS if installed (with a COLMAP result),
    else the built-in simulation."""
    from . import colmap, openmvs
    ch = ctx.chunk
    if not ch.aligned:
        ctx.log("Cameras are not aligned. Run 'Align Photos' first.", "error")
        return False
    res = getattr(ch, "_colmap_result", None)
    if openmvs.available() and colmap.available() and res is not None and res.model_dir:
        return _dense_openmvs(ctx, res, colmap, openmvs)
    if res is None or not colmap.available():
        ctx.log("No real reconstruction available — using built-in simulation for the "
                "dense cloud.", "warn")
    else:
        ctx.log("OpenMVS not found on PATH — using built-in simulation for the dense "
                "cloud.", "warn")
    return _dense_simulated(ctx)


def _dense_openmvs(ctx: StageContext, res, colmap, openmvs) -> bool:
    ctx.log(f"Dense multi-view stereo via OpenMVS ({openmvs.densify_exe()})...", "info")
    ply = openmvs.run_dense(res.model_dir, res.image_dir, ctx.workdir, ctx, colmap.exe())
    if ply is None:
        return False
    P, C = openmvs.load_ply(ply)
    if len(P) == 0:
        ctx.log("OpenMVS dense cloud is empty.", "error")
        return False
    sim = getattr(ctx.chunk, "_georef_sim", None)
    if sim is not None:
        P = sim.apply(P)  # carry local dense cloud into the project CRS
        ctx.log("Dense cloud transformed into project CRS.", "info")
    out = os.path.join(ctx.workdir, "dense_cloud.las")
    _write_las(out, P, C, np.ones(len(P), np.uint8))
    ctx.chunk.outputs.dense_cloud = out
    ctx.log(f"Dense cloud: {len(P):,} points -> {out}", "ok")
    ctx.progress(100)
    return True


def _dense_simulated(ctx: StageContext) -> bool:
    ctx.log("Computing per-pixel depth maps (Multi-View Stereo)...", "info")
    if not ctx.sleep(1.0):
        return False
    ctx.progress(45)
    ctx.log("Fusing depth maps into dense point cloud...", "info")
    P, C, K = _synth_scene()
    if not ctx.sleep(0.6):
        return False
    out = os.path.join(ctx.workdir, "dense_cloud.las")
    _write_las(out, P, C, np.ones_like(K))  # class 1 = unclassified
    ctx.chunk.outputs.dense_cloud = out
    ctx.progress(100)
    ctx.log(f"Dense cloud: {len(P):,} points -> {out}", "ok")
    return True


def _classify_cell(P) -> float:
    """Ground-filter cell size: ~1 m for typical drone extents, larger for big areas."""
    minx, miny, maxx, maxy = _extent(P)
    span = max(maxx - minx, maxy - miny, 1.0)
    return min(max(span / 500.0, 1.0), 3.0)


def run_classify(ctx: StageContext) -> bool:
    from . import classify, ml_classify
    path = ctx.chunk.outputs.dense_cloud
    if not path or not os.path.exists(path):
        ctx.log("No dense cloud to classify. Run 'Build Dense Cloud' first.", "error")
        return False
    out = os.path.join(ctx.workdir, "classified_cloud.las")

    if classify.pdal_available():
        ctx.log("Classifying with PDAL (SMRF ground filter + HAG)...", "info")
        if classify.pdal_classify(path, out):
            ctx.chunk.outputs.classified_cloud = out
            _, _, cls = _load_cloud(out)
            counts = {int(k): int((cls == k).sum()) for k in np.unique(cls)}
            ctx.log(f"Classified (PDAL): {counts} -> {out}", "ok")
            ctx.progress(100)
            return True
        ctx.log("PDAL pipeline failed; falling back to built-in classifier.", "warn")

    P, C, _ = _load_cloud(path)
    if not ctx.sleep(0.2):
        return False
    if os.environ.get("AEROSURVEY_USE_ML_CLASSIFIER") and ml_classify.available():
        ctx.log("Classifying with the trained Random Forest model...", "info")
        cls = ml_classify.classify(P, C)
    else:
        ctx.log("Classifying (progressive morphological ground filter + roughness split)...",
                "info")
        cls = classify.classify_cloud(P, C, cell=_classify_cell(P))
    if ctx.cancelled:
        return False
    ctx.progress(85)
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
    cell = _dsm_cell(P)
    arr, origin, nx, ny = _grid(P, P[:, 2], cell=cell, reducer="max")
    arr = _fill_nan(arr)
    if not ctx.sleep(0.4):
        return False
    out = os.path.join(ctx.workdir, "dsm.tif")
    _write_geotiff(out, arr, ctx.chunk, origin, cell)
    ctx.chunk.outputs.dsm = out
    ctx.log(f"DSM {nx}x{ny} @ {cell:.3f} m -> {out}", "ok")
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
    cell = _dsm_cell(P)
    arr, origin, nx, ny = _grid(ground, ground[:, 2], cell=cell, reducer="min")
    arr = _fill_nan(arr)
    if not ctx.sleep(0.4):
        return False
    out = os.path.join(ctx.workdir, "dtm.tif")
    _write_geotiff(out, arr, ctx.chunk, origin, cell)
    ctx.chunk.outputs.dtm = out
    ctx.log(f"DTM {nx}x{ny} @ {cell:.3f} m -> {out}", "ok")
    ctx.progress(100)
    return True


def run_ortho(ctx: StageContext) -> bool:
    path = _cloud_for_surfaces(ctx.chunk)
    if not path or not os.path.exists(path):
        ctx.log("No point cloud available for orthomosaic.", "error")
        return False
    ctx.log("Projecting images onto DSM and mosaicking (nadir seamlines)...", "info")
    P, C, _ = _load_cloud(path)
    cell = _raster_cell(P, target_px=1600)
    minx, miny, maxx, maxy = _extent(P)
    nx = max(int(np.ceil((maxx - minx) / cell)) + 1, 1)
    ny = max(int(np.ceil((maxy - miny) / cell)) + 1, 1)
    ix, iy = _cell_indices(P, minx, maxy, cell, nx, ny)
    order = np.argsort(P[:, 2])  # highest last -> top surface wins
    rgb = np.zeros((3, ny, nx), np.float32)
    for b in range(3):
        band = np.full((ny, nx), np.nan, np.float32)
        band[iy[order], ix[order]] = C[order, b].astype(np.float32)
        rgb[b] = _fill_nan(band)
        if not ctx.sleep(0.15):
            return False
        ctx.progress(30 + b * 20)
    out = os.path.join(ctx.workdir, "orthomosaic.tif")
    _write_geotiff(out, rgb, ctx.chunk, (minx, maxy), cell)
    ctx.chunk.outputs.orthomosaic = out
    ctx.log(f"Orthomosaic {nx}x{ny} @ {cell:.3f} m (3-band RGB) -> {out}", "ok")
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
    Stage("georef", "Optimize / Georeference", "internal", run_georef,
          "Cameras in project CRS + GCP accuracy report"),
    Stage("dense", "Build Dense Cloud", "openmvs", run_dense, "Dense point cloud (LAS)"),
    Stage("classify", "Classify Points", "pdal", run_classify, "Classified cloud"),
    Stage("dsm", "Build DSM", "gdal", run_dsm, "Digital Surface Model (GeoTIFF)"),
    Stage("dtm", "Build DTM", "gdal", run_dtm, "Digital Terrain Model (GeoTIFF)"),
    Stage("ortho", "Build Orthomosaic", "gdal", run_ortho, "Orthomosaic (GeoTIFF)"),
]


def stage_by_key(key: str) -> Stage:
    return next(s for s in PIPELINE if s.key == key)
