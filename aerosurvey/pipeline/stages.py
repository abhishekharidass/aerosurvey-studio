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
import shutil
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


def _z_offset(chunk) -> float:
    """Vertical-datum shift: subtracting the geoid separation N converts
    ellipsoidal heights to orthometric (MSL)."""
    if getattr(chunk, "vertical_datum", "ellipsoidal") == "orthometric" and chunk.geoid_separation:
        return float(chunk.geoid_separation)
    return 0.0


def _image_gsd(ctx: "StageContext", P: np.ndarray, cls: np.ndarray = None):
    """Best estimate of the source-photo GSD (m/px), cached in chunk.stats."""
    from ..core import gsd as gsdmod
    from . import recon
    ch = ctx.chunk
    cached = ch.stats.get("image_gsd_m")
    if cached:
        return float(cached)
    f_map = {}
    res = recon.get_reconstruction(ch, ctx.workdir)
    if res is not None and res.intrinsics:
        for cam in ch.cameras:
            pose = res.poses.get(cam.filename)
            K = res.intrinsics.get(pose.camera_id) if pose else None
            if K is not None:
                f_map[cam.id] = (float(K[0][0]) + float(K[1][1])) / 2.0
    # ground elevation only comparable to est_z once georeferenced
    ground_z = None
    if ch.optimized:
        if cls is not None and (cls == 2).any():
            ground_z = float(np.median(P[cls == 2, 2]))   # classified ground
        else:
            ground_z = float(np.percentile(P[:, 2], 5))
    g = gsdmod.estimate_gsd(ch, ground_z, f_map)
    if g:
        ch.stats["image_gsd_m"] = round(g, 4)
    return g


def _target_cell(ctx: "StageContext", P: np.ndarray, kind: str,
                 cls: np.ndarray = None) -> float:
    """Output cell size for 'ortho' or 'surface' rasters, honouring settings."""
    from . import raster
    s = ctx.chunk.settings
    if kind == "ortho":
        mode, custom = s.ortho_gsd_mode, s.ortho_gsd
    else:
        mode, custom = s.surface_gsd_mode, s.surface_gsd
    if mode == "custom" and custom > 0:
        cell, src = float(custom), "custom"
    else:
        g = _image_gsd(ctx, P, cls)
        if g:
            cell, src = float(g), "auto — estimated image GSD"
        else:
            cell = max(raster.median_spacing(P), 0.02)
            src = "point density (no GSD info available)"
    minx, miny, maxx, maxy = raster.extent(P)
    span = max(maxx - minx, maxy - miny, 1.0)
    min_cell = span / max(int(s.max_raster_dim), 1000)
    if cell < min_cell:
        ctx.log(f"Requested {cell*100:.1f} cm cell needs a raster larger than "
                f"{s.max_raster_dim} px — coarsened to {min_cell*100:.1f} cm "
                "(raise 'Max raster dimension' in Processing Settings).", "warn")
        cell = min_cell
    ctx.log(f"{'Orthomosaic' if kind == 'ortho' else 'Surface'} resolution: "
            f"{cell*100:.1f} cm/px ({src}).", "info")
    return cell


def _voxel_downsample(P: np.ndarray, C: np.ndarray, density: float):
    """Thin a cloud to ~density points/m² (3D voxel grid, keeps first hit)."""
    if density <= 0 or len(P) == 0:
        return P, C
    cell = 1.0 / np.sqrt(density)
    ijk = np.floor(P / cell).astype(np.int64)
    key = (ijk[:, 0] * 73856093) ^ (ijk[:, 1] * 19349663) ^ (ijk[:, 2] * 83492791)
    _, idx = np.unique(key, return_index=True)
    idx.sort()
    return P[idx], C[idx]


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
    res = colmap.run_sfm(image_paths, ctx.workdir, ctx,
                         max_features=ctx.chunk.settings.sfm_max_features)
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

    ctx.chunk.stats.update({"cameras_total": len(cams), "cameras_aligned": matched,
                            "mean_reproj_px": round(float(res.mean_reproj_error), 3),
                            "sparse_points": int(len(res.points)), "align_engine": "COLMAP"})
    if res.intrinsics:
        K = next(iter(res.intrinsics.values()))
        ctx.chunk.stats["calibration"] = {
            "focal_px": round(float((K[0][0] + K[1][1]) / 2.0), 2),
            "cx": round(float(K[0][2]), 2), "cy": round(float(K[1][2]), 2),
            "source": "COLMAP"}
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
    ctx.chunk.stats.update({"cameras_total": len(cams), "cameras_aligned": len(cams),
                            "mean_reproj_px": round(float(err.mean()), 3),
                            "align_engine": "simulation"})
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
    from . import georef, recon
    from ..core import crs as crsmod
    ch = ctx.chunk
    if not ch.aligned:
        ctx.log("Run 'Align Photos' before georeferencing.", "error")
        return False
    for g in ch.gcps:
        g.error = None

    tf = crsmod.CrsTransform(ch.epsg if ch.crs_mode != "local" else None)
    res = recon.get_reconstruction(ch, ctx.workdir, ctx)
    sim = None

    if res is not None and ch.gcps:
        sim = _georef_from_gcps(ctx, ch, res, georef)   # accurate, real reconstruction
    if sim is None:
        sim = _georef_from_gps(ctx, ch, tf, georef, res)  # GPS similarity fallback
    if sim is None:
        ctx.log("Not enough control to georeference (need >=3 geotagged aligned "
                "cameras, or >=3 triangulable control GCPs).", "error")
        return False

    if res is not None:
        # sim maps COLMAP's local frame -> project CRS: derive camera positions
        # and the sparse cloud from the reconstruction (idempotent on re-runs).
        by_name = {c.filename: c for c in ch.cameras}
        for name, pose in res.poses.items():
            cam = by_name.get(name)
            if cam is not None:
                x, y, z = sim.apply(pose.center)
                cam.est_x, cam.est_y, cam.est_z = float(x), float(y), float(z)
        if ch.outputs.sparse_cloud and len(res.points):
            _write_las(ch.outputs.sparse_cloud, sim.apply(res.points), res.colors,
                       np.ones(len(res.points), np.uint8))
            ctx.log("Sparse cloud written in project CRS.", "info")
    else:
        for c in ch.cameras:
            if c.est_x is not None:
                x, y, z = sim.apply([c.est_x, c.est_y, c.est_z])
                c.est_x, c.est_y, c.est_z = float(x), float(y), float(z)
        if ch.outputs.sparse_cloud and os.path.exists(ch.outputs.sparse_cloud):
            _transform_las(ch.outputs.sparse_cloud, sim)
            ctx.log("Sparse cloud transformed into project CRS.", "info")

    ch._georef_sim = sim  # transient: applied to the dense cloud (still local on disk)
    recon.save_sim(ctx.workdir, sim, ch.stats.get("georef_method", ""))
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
    ch.stats["ba_rmse_px"] = round(float(r.rmse_after), 3)
    ctx.log(f"Bundle adjustment: reprojection RMSE {r.rmse_before:.3f} -> "
            f"{r.rmse_after:.3f} px over {r.n_obs} observations.", "ok")
    if r.intrinsics is not None:
        ch.stats["calibration"] = {
            "focal_px": round(float(r.intrinsics[0]), 2),
            "cx": round(float(r.intrinsics[1]), 2),
            "cy": round(float(r.intrinsics[2]), 2),
            "k1": round(float(r.intrinsics[3]), 6),
            "source": "self-calibrated (bundle adjustment)"}
        ctx.log(f"Self-calibration: focal {shared[0]:.1f} -> {r.intrinsics[0]:.1f} px, "
                f"principal ({r.intrinsics[1]:.1f}, {r.intrinsics[2]:.1f}), "
                f"k1 {r.intrinsics[3]:+.5f}.", "info")


def _georef_from_gps(ctx, ch, tf, georef, res=None):
    """Fit local->world from camera geotags. With a real reconstruction the
    local side comes from the COLMAP pose centres (always in the local frame,
    so re-running georeference stays idempotent)."""
    local, world = [], []
    for c in ch.cameras:
        if not c.has_geotag:
            continue
        if res is not None:
            pose = res.poses.get(c.filename)
            if pose is None:
                continue
            local.append([float(pose.center[0]), float(pose.center[1]),
                          float(pose.center[2])])
        elif c.aligned and c.est_x is not None:
            local.append([c.est_x, c.est_y, c.est_z])
        else:
            continue
        world.append(list(tf.forward(c.lon, c.lat, c.alt or 0.0)))
    if len(local) < 3:
        return None
    sim = georef.umeyama_similarity(np.array(local), np.array(world))
    fit = georef.residuals(np.array(local), np.array(world), sim)
    ch.stats.update({"georef_method": "Camera GPS", "georef_rmse_m": round(float(fit.rmse), 3),
                     "georef_rmse_xyz_m": [round(float(v), 3) for v in fit.rmse_axis]})
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
    control_rmse = float(np.sqrt(np.mean(fit.per_point[ctrl] ** 2)))
    ctrl_fit = georef.residuals(L[ctrl], W[ctrl], sim)
    ch.stats.update({"georef_method": "Ground Control Points", "gcp_control": len(control),
                     "gcp_check": len(checks) - len(control),
                     "control_rmse_m": round(control_rmse, 3),
                     "georef_rmse_xyz_m": [round(float(v), 3)
                                           for v in ctrl_fit.rmse_axis]})
    ctx.log(f"GCP georeferencing: {len(control)} control, {len(checks) - len(control)} check. "
            f"Control RMSE {control_rmse:.3f} m.", "ok")
    check = np.array([i for i, chk in enumerate(checks) if chk], dtype=int)
    if len(check):
        ch.stats["check_rmse_m"] = round(float(np.sqrt(np.mean(fit.per_point[check] ** 2))), 3)
        ctx.log(f"Independent check-point RMSE "
                f"{np.sqrt(np.mean(fit.per_point[check] ** 2)):.3f} m.", "ok")
    return sim


def run_dense(ctx: StageContext) -> bool:
    """Dense reconstruction. Uses OpenMVS if installed (with a COLMAP result),
    else the built-in simulation."""
    from . import colmap, openmvs, recon
    ch = ctx.chunk
    if not ch.aligned:
        ctx.log("Cameras are not aligned. Run 'Align Photos' first.", "error")
        return False
    res = recon.get_reconstruction(ch, ctx.workdir, ctx)
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
    from . import recon
    s = ctx.chunk.settings
    ctx.log(f"Dense multi-view stereo via OpenMVS ({openmvs.densify_exe()}), "
            f"quality: {s.dense_quality}...", "info")
    ply = openmvs.run_dense(res.model_dir, res.image_dir, ctx.workdir, ctx,
                            colmap.exe(), quality=s.dense_quality)
    if ply is None:
        return False
    P, C = openmvs.load_ply(ply)
    if len(P) == 0:
        ctx.log("OpenMVS dense cloud is empty.", "error")
        return False
    sim = recon.get_sim(ctx.chunk, ctx.workdir)
    if sim is not None:
        P = sim.apply(P)  # carry local dense cloud into the project CRS
        ctx.log("Dense cloud transformed into project CRS.", "info")
    if s.dense_target_density > 0:
        n0 = len(P)
        P, C = _voxel_downsample(P, C, s.dense_target_density)
        ctx.log(f"Density limited to {s.dense_target_density:g} pts/m²: "
                f"{n0:,} -> {len(P):,} points.", "info")
    out = os.path.join(ctx.workdir, "dense_cloud.las")
    _write_las(out, P, C, np.ones(len(P), np.uint8))
    ctx.chunk.outputs.dense_cloud = out
    ctx.chunk.stats["dense_points"] = int(len(P))
    dens = len(P) / max((P[:, 0].max() - P[:, 0].min()) *
                        (P[:, 1].max() - P[:, 1].min()), 1e-6)
    ctx.chunk.stats["dense_density_ppm2"] = round(float(dens), 1)
    ctx.log(f"Dense cloud: {len(P):,} points (~{dens:.1f} pts/m²) -> {out}", "ok")
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
    ctx.chunk.stats["dense_points"] = int(len(P))
    ctx.progress(100)
    ctx.log(f"Dense cloud: {len(P):,} points -> {out}", "ok")
    return True


def run_mesh(ctx: StageContext) -> bool:
    """Textured 3D mesh (OBJ). OpenMVS ReconstructMesh + TextureMesh over the
    densified scene; falls back to a coloured heightfield from the dense cloud."""
    from . import mesh as meshmod
    from . import openmvs, recon
    ch = ctx.chunk
    if not ch.outputs.dense_cloud or not os.path.exists(ch.outputs.dense_cloud):
        ctx.log("No dense cloud. Run 'Build Dense Cloud' first.", "error")
        return False
    epsg = ch.epsg if ch.crs_mode != "local" else None
    out_dir = os.path.join(ctx.workdir, "mesh")
    os.makedirs(out_dir, exist_ok=True)

    scene_dense = os.path.join(ctx.workdir, "openmvs", "scene_dense.mvs")
    if openmvs.mesh_available() and os.path.exists(scene_dense):
        ctx.log(f"Meshing via OpenMVS ({openmvs.reconstruct_exe()})...", "info")
        obj = openmvs.run_mesh(ctx.workdir, ctx,
                               max_faces=ch.settings.mesh_max_faces)
        if obj is None:
            return False
        # textures + material file travel with the OBJ
        src_dir = os.path.dirname(obj)
        for fn in os.listdir(src_dir):
            base, ext = os.path.splitext(fn)
            if base.startswith("scene_textured") and ext.lower() not in (
                    ".obj", ".mvs", ".log", ".dmap"):
                shutil.copy2(os.path.join(src_dir, fn), os.path.join(out_dir, fn))
        sim = recon.get_sim(ch, ctx.workdir)
        v0 = meshmod.peek_first_vertex(obj)
        if v0 is None:
            ctx.log("Textured mesh has no vertices.", "error")
            return False
        offset = meshmod.auto_offset(sim.apply(v0) if sim is not None else v0)
        out_obj = os.path.join(out_dir, "model.obj")
        nv, nf = meshmod.transform_obj(obj, out_obj, sim, offset, epsg)
        if sim is not None:
            ctx.log("Mesh transformed into project CRS "
                    f"(offset {offset[0]:.0f}, {offset[1]:.0f} recorded in "
                    "model_offset.txt).", "info")
    else:
        if not openmvs.mesh_available():
            ctx.log("OpenMVS ReconstructMesh/TextureMesh not found — building a "
                    "coloured heightfield mesh from the dense cloud.", "warn")
        else:
            ctx.log("No OpenMVS dense scene on disk — building a coloured "
                    "heightfield mesh from the dense cloud.", "warn")
        import laspy
        las = laspy.read(ch.outputs.dense_cloud)
        P = np.column_stack([las.x, las.y, las.z])
        C = (np.column_stack([las.red, las.green, las.blue]) // 257).astype(np.uint8)
        ctx.progress(30)
        span = max(P[:, 0].max() - P[:, 0].min(),
                   P[:, 1].max() - P[:, 1].min(), 1e-6)
        cell = max(span / 400.0, 0.05)
        verts, colors, faces = meshmod.heightfield_mesh(P, C, cell)
        if len(faces) == 0:
            ctx.log("Mesh triangulation produced no faces.", "error")
            return False
        ctx.progress(70)
        out_obj = os.path.join(out_dir, "model.obj")
        offset = meshmod.auto_offset(verts.min(axis=0))
        meshmod.write_obj_with_colors(out_obj, verts, colors, faces, offset, epsg)
        nv, nf = len(verts), len(faces)

    ch.outputs.mesh = out_obj
    ch.stats["mesh_vertices"] = int(nv)
    ch.stats["mesh_faces"] = int(nf)
    mb = os.path.getsize(out_obj) / 1e6
    ctx.log(f"Textured mesh: {nv:,} vertices, {nf:,} faces -> {out_obj} "
            f"({mb:.1f} MB)", "ok")
    ctx.progress(100)
    return True


def run_classify(ctx: StageContext) -> bool:
    from . import classify, ml_classify
    path = ctx.chunk.outputs.dense_cloud
    if not path or not os.path.exists(path):
        ctx.log("No dense cloud to classify. Run 'Build Dense Cloud' first.", "error")
        return False
    out = os.path.join(ctx.workdir, "classified_cloud.las")

    if classify.pdal_available():
        ctx.log("Classifying with PDAL (outlier + SMRF ground + HAG + coplanarity)...", "info")
        if classify.pdal_classify(path, out):
            ctx.chunk.outputs.classified_cloud = out
            _, _, cls = _load_cloud(out)
            counts = {int(k): int((cls == k).sum()) for k in np.unique(cls)}
            ctx.chunk.stats["class_counts"] = counts
            ctx.log(f"Classified (PDAL): {counts} -> {out}", "ok")
            ctx.progress(100)
            return True
        ctx.log("PDAL pipeline failed; falling back to built-in classifier.", "warn")

    P, C, _ = _load_cloud(path)
    use_ml = (ctx.chunk.settings.classifier == "ml"
              or os.environ.get("AEROSURVEY_USE_ML_CLASSIFIER"))
    if use_ml and ml_classify.available():
        ctx.log("Classifying with the trained Random Forest model...", "info")
        cls = ml_classify.classify(P, C)
    else:
        cell = classify.default_cell(P)
        ctx.log(f"Classifying: outlier removal, progressive morphological ground "
                f"filter ({cell:.1f} m cells), HAG + planarity split...", "info")
        ctx.progress(10)
        cls = classify.classify_cloud(P, C, cell=cell)
    if ctx.cancelled:
        return False
    ctx.progress(85)
    _write_las(out, P, C, cls)
    ctx.chunk.outputs.classified_cloud = out
    counts = {int(k): int((cls == k).sum()) for k in np.unique(cls)}
    ctx.chunk.stats["class_counts"] = counts
    ctx.log(f"Classified: {counts} -> {out}", "ok")
    ctx.progress(100)
    return True


def _cloud_for_surfaces(ch: Chunk):
    path = ch.outputs.classified_cloud or ch.outputs.dense_cloud
    return path


def _write_surface(ctx: StageContext, P: np.ndarray, values: np.ndarray,
                   out: str, reducer: str, cls: np.ndarray = None) -> dict:
    """DSM/DTM writer: native-density grid, hole-fill, interpolate to GSD."""
    from . import raster
    cell = _target_cell(ctx, P, "surface", cls)
    native_cell = raster.pick_native_cell(P, cell)
    nat = raster.native_grid(P, values, native_cell, reducer=reducer)
    minx, miny, maxx, maxy = raster.extent(P)
    nx, ny = raster.grid_shape(P, cell)
    final = raster.write_interp_raster(
        out, [nat], ctx.chunk, (minx, maxy), native_cell, cell, nx, ny,
        dtype="float32", nodata=np.nan, z_offset=_z_offset(ctx.chunk),
        progress=lambda p: ctx.progress(20 + int(p * 0.75)), log=ctx.log)
    return {"w": nx, "h": ny, "gsd_m": round(cell, 4), "path": final}


def run_dsm(ctx: StageContext) -> bool:
    path = _cloud_for_surfaces(ctx.chunk)
    if not path or not os.path.exists(path):
        ctx.log("No point cloud available for DSM.", "error")
        return False
    ctx.log("Rasterising DSM (top surface)...", "info")
    P, _, cls = _load_cloud(path)
    keep = cls != 7                      # never build surfaces from noise points
    if keep.any() and not keep.all():
        P, cls = P[keep], cls[keep]
    out = os.path.join(ctx.workdir, "dsm.tif")
    info = _write_surface(ctx, P, P[:, 2], out, reducer="max", cls=cls)
    if ctx.cancelled:
        return False
    out = info.pop("path", out)
    ctx.chunk.outputs.dsm = out
    ctx.chunk.stats["dsm"] = info
    ctx.log(f"DSM {info['w']}x{info['h']} @ {info['gsd_m']:.3f} m -> {out}", "ok")
    ctx.progress(100)
    return True


def run_dtm(ctx: StageContext) -> bool:
    path = ctx.chunk.outputs.classified_cloud
    if not path or not os.path.exists(path):
        ctx.log("DTM needs a classified cloud. Run 'Classify Points' first.", "error")
        return False
    ctx.log("Rasterising DTM from ground-classified points...", "info")
    P, _, cls = _load_cloud(path)
    ground = P[cls == 2]
    if len(ground) == 0:
        ctx.log("No ground points found.", "error")
        return False
    out = os.path.join(ctx.workdir, "dtm.tif")
    info = _write_surface(ctx, ground, ground[:, 2], out, reducer="min",
                          cls=np.full(len(ground), 2, np.uint8))
    if ctx.cancelled:
        return False
    out = info.pop("path", out)
    ctx.chunk.outputs.dtm = out
    ctx.chunk.stats["dtm"] = info
    ctx.log(f"DTM {info['w']}x{info['h']} @ {info['gsd_m']:.3f} m -> {out}", "ok")
    ctx.progress(100)
    return True


def run_ortho(ctx: StageContext) -> bool:
    from . import ortho as orthomod
    from . import raster, recon
    path = _cloud_for_surfaces(ctx.chunk)
    if not path or not os.path.exists(path):
        ctx.log("No point cloud available for orthomosaic.", "error")
        return False
    P, C, cls = _load_cloud(path)
    keep = cls != 7
    if keep.any() and not keep.all():
        P, C, cls = P[keep], C[keep], cls[keep]
    cell = _target_cell(ctx, P, "ortho", cls)
    out = os.path.join(ctx.workdir, "orthomosaic.tif")

    res = recon.get_reconstruction(ctx.chunk, ctx.workdir, ctx)
    sim = recon.get_sim(ctx.chunk, ctx.workdir)
    if res is not None and sim is not None and res.intrinsics:
        ctx.log("True orthorectification: projecting source photos onto the "
                "surface model (nearest-nadir view per cell)...", "info")
        info = orthomod.build_true_ortho(ctx, res, sim, P, cell, out, cls=cls)
        if ctx.cancelled:
            return False
        if info is not None:
            out = info.pop("path", out)
            ctx.chunk.outputs.orthomosaic = out
            info["engine"] = "true-ortho"
            ctx.chunk.stats["ortho"] = info
            ctx.log(f"Orthomosaic {info['w']}x{info['h']} @ {info['gsd_m']:.3f} m "
                    f"({info['coverage_pct']}% covered, RGBA) -> {out}", "ok")
            ctx.progress(100)
            return True
        ctx.log("True orthorectification unavailable — falling back to "
                "point-cloud colour splatting.", "warn")
    else:
        ctx.log("No reconstruction/georeference on disk — orthomosaic from "
                "point-cloud colours (run Align + Georeference for the "
                "photo-projected version).", "warn")

    native_cell = raster.pick_native_cell(P, cell)
    natives = [raster.native_grid(P, C[:, b].astype(np.float32), native_cell,
                                  orderby=P[:, 2]) for b in range(3)]
    minx, miny, maxx, maxy = raster.extent(P)
    nx, ny = raster.grid_shape(P, cell)
    out = raster.write_interp_raster(
        out, natives, ctx.chunk, (minx, maxy), native_cell, cell, nx, ny,
        dtype="uint8", nodata=None,
        progress=lambda p: ctx.progress(30 + int(p * 0.65)), log=ctx.log)
    if ctx.cancelled:
        return False
    ctx.chunk.outputs.orthomosaic = out
    ctx.chunk.stats["ortho"] = {"w": nx, "h": ny, "gsd_m": round(cell, 4),
                                "engine": "splat"}
    ctx.log(f"Orthomosaic {nx}x{ny} @ {cell:.3f} m (RGB) -> {out}", "ok")
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
    Stage("mesh", "Build Textured Mesh", "openmvs", run_mesh, "Textured 3D mesh (OBJ)"),
    Stage("classify", "Classify Points", "pdal", run_classify, "Classified cloud"),
    Stage("dsm", "Build DSM", "gdal", run_dsm, "Digital Surface Model (GeoTIFF)"),
    Stage("dtm", "Build DTM", "gdal", run_dtm, "Digital Terrain Model (GeoTIFF)"),
    Stage("ortho", "Build Orthomosaic", "gdal", run_ortho, "Orthomosaic (GeoTIFF)"),
]


def stage_by_key(key: str) -> Stage:
    return next(s for s in PIPELINE if s.key == key)
