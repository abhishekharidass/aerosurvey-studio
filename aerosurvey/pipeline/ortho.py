"""True orthorectification: project the source photos onto the surface model.

This is how Pix4D/Metashape build orthomosaics — for every output cell the
surface height is looked up, the best camera observing that spot is chosen
(nearest nadir view), and the colour is sampled from the actual photo at the
projected pixel. The result carries full image detail at the native GSD,
unlike splatting dense-cloud points (the fallback when no reconstruction
is available).

Inputs: the COLMAP poses/intrinsics (local frame), the georeferencing
similarity (local -> project CRS), and the staged (ideally undistorted)
images. Runs block-wise; memory stays bounded regardless of raster size.
"""
from __future__ import annotations

import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import colmap as colmod
from . import raster


# -- image cache -----------------------------------------------------------
class _ImageCache:
    """Small LRU of decoded photos (~64 MB each for a 20 MP frame)."""

    def __init__(self, cap: int = 10):
        self.cap = cap
        self._d: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def get(self, path: str) -> Optional[np.ndarray]:
        if path in self._d:
            self._d.move_to_end(path)
            return self._d[path]
        from PIL import Image
        try:
            with Image.open(path) as im:
                arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        except Exception:
            return None
        self._d[path] = arr
        if len(self._d) > self.cap:
            self._d.popitem(last=False)
        return arr


def _bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Sample HxWx3 uint8 at float pixel coords (u=x, v=y) -> (N,3) float32."""
    h, w = img.shape[:2]
    u = np.clip(u, 0, w - 1.001)
    v = np.clip(v, 0, h - 1.001)
    u0 = u.astype(int); v0 = v.astype(int)
    du = (u - u0)[:, None]; dv = (v - v0)[:, None]
    p00 = img[v0, u0].astype(np.float32)
    p01 = img[v0, u0 + 1].astype(np.float32)
    p10 = img[v0 + 1, u0].astype(np.float32)
    p11 = img[v0 + 1, u0 + 1].astype(np.float32)
    return (p00 * (1 - du) * (1 - dv) + p01 * du * (1 - dv)
            + p10 * (1 - du) * dv + p11 * du * dv)


# -- camera set --------------------------------------------------------------
class _OrthoCam:
    __slots__ = ("name", "R", "t", "K", "path", "center_w", "wh")

    def __init__(self, name, R, t, K, path, center_w, wh):
        self.name = name; self.R = R; self.t = t; self.K = K
        self.path = path; self.center_w = center_w; self.wh = wh


def _undistorted_model(workdir: str, ctx) -> Optional[str]:
    """TXT model dir of the undistorted workspace, converting if needed."""
    model = os.path.join(workdir, "openmvs", "undistort", "sparse")
    if not os.path.isdir(model):
        return None
    if os.path.exists(os.path.join(model, "images.txt")):
        return model
    if not colmod.available():
        return None
    r = colmod._run_step("model_converter (undistorted)",
                         ["model_converter", "--input_path", model,
                          "--output_path", model, "--output_type", "TXT"], ctx)
    return model if r is True else None


def _image_size(path: str) -> Optional[Tuple[int, int]]:
    from PIL import Image
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def gather_cameras(res, sim, workdir: str, ctx) -> List[_OrthoCam]:
    """Build the projection set, preferring undistorted images + intrinsics."""
    poses, intr, img_dir = res.poses, res.intrinsics, res.image_dir
    und = _undistorted_model(workdir, ctx)
    if und is not None:
        u_poses, _ = colmod.parse_images_txt(os.path.join(und, "images.txt"))
        u_intr = colmod.parse_cameras_txt(os.path.join(und, "cameras.txt"))
        u_dir = os.path.join(workdir, "openmvs", "undistort", "images")
        if u_poses and u_intr and os.path.isdir(u_dir):
            poses, intr, img_dir = u_poses, u_intr, u_dir
            ctx.log("Orthomosaic uses undistorted images + intrinsics.", "info")
    cams: List[_OrthoCam] = []
    for name, pose in poses.items():
        K = intr.get(pose.camera_id)
        path = os.path.join(img_dir, name) if img_dir else ""
        if K is None or not os.path.exists(path):
            continue
        size = _image_size(path)
        if size is None:
            continue
        R = colmod.qvec2rotmat(pose.qvec)
        cams.append(_OrthoCam(name, R, np.asarray(pose.tvec, float),
                              np.asarray(K, float), path,
                              sim.apply(pose.center), size))
    return cams


# -- projection helpers -------------------------------------------------------
def _project_cam(cam: _OrthoCam, local_pts: np.ndarray):
    """Project local-frame points into a camera. Returns (u, v, ok)."""
    Xc = local_pts @ cam.R.T + cam.t
    zc = Xc[:, 2]
    ok = zc > 1e-6
    uv = Xc[:, :2] / np.maximum(zc, 1e-9)[:, None]
    u = cam.K[0, 0] * uv[:, 0] + cam.K[0, 2]
    v = cam.K[1, 1] * uv[:, 1] + cam.K[1, 2]
    w, h = cam.wh
    ok &= (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    return u, v, ok


def _border_feather(cam: _OrthoCam, u: np.ndarray, v: np.ndarray,
                    frac: float = 0.06) -> np.ndarray:
    """0..1 weight that fades to 0 at the image border, so a camera's
    footprint edge never produces a hard seam."""
    w, h = cam.wh
    feather = max(frac * min(w, h), 1.0)
    d = np.minimum(np.minimum(u, w - 1 - u), np.minimum(v, h - 1 - v))
    return np.clip(d / feather, 0.02, 1.0)


def estimate_gains(cams, P: np.ndarray, inv, tree, cache,
                   n_samples: int = 25000, seed: int = 0) -> np.ndarray:
    """Per-camera RGB gains that equalise exposure between overlapping photos.

    Samples cloud points seen by their two nearest cameras, collects the mean
    log colour ratio per camera pair, and solves a least-squares system for
    log-gains (anchored to mean 0). Returns (n_cams, 3) multiplicative gains.
    """
    n_cams = len(cams)
    gains = np.ones((n_cams, 3), np.float64)
    if n_cams < 2 or len(P) == 0:
        return gains
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(P), min(n_samples, len(P)), replace=False)
    world = P[pick]
    local = inv.apply(world)
    _, nearest = tree.query(world[:, :2], k=2)
    nearest = np.atleast_2d(nearest.reshape(len(world), -1))

    # colours of each sample in its two candidate cameras
    cols = np.full((2, len(world), 3), np.nan, np.float32)
    for slot in range(2):
        cand = nearest[:, slot]
        for ci in np.unique(cand):
            cam = cams[ci]
            sel = np.where(cand == ci)[0]
            u, v, ok = _project_cam(cam, local[sel])
            if not ok.any():
                continue
            img = cache.get(cam.path)
            if img is None:
                continue
            cols[slot, sel[ok]] = _bilinear(img, u[ok], v[ok])

    both = ~np.isnan(cols[0, :, 0]) & ~np.isnan(cols[1, :, 0])
    if both.sum() < 50:
        return gains
    ci, cj = nearest[both, 0], nearest[both, 1]
    log_ratio = (np.log(np.clip(cols[1, both], 8, 255))
                 - np.log(np.clip(cols[0, both], 8, 255)))  # log(cj_col)-log(ci_col)

    # aggregate per (i, j) pair, then solve  g_i - g_j = mean log ratio
    for ch in range(3):
        pairs = {}
        for a, b, r in zip(ci, cj, log_ratio[:, ch]):
            key = (int(a), int(b))
            s, n = pairs.get(key, (0.0, 0))
            pairs[key] = (s + float(r), n + 1)
        rows, rhs, wts = [], [], []
        for (a, b), (s, n) in pairs.items():
            if n < 5:
                continue
            row = np.zeros(n_cams)
            row[a], row[b] = 1.0, -1.0
            rows.append(row)
            rhs.append(s / n)
            wts.append(np.sqrt(n))
        if len(rows) < 1:
            continue
        A = np.asarray(rows) * np.asarray(wts)[:, None]
        y = np.asarray(rhs) * np.asarray(wts)
        # anchor: mean log-gain = 0
        A = np.vstack([A, np.full((1, n_cams), 10.0)])
        y = np.append(y, 0.0)
        g, *_ = np.linalg.lstsq(A, y, rcond=None)
        gains[:, ch] = np.clip(np.exp(g), 0.6, 1.6)
    return gains


# -- main --------------------------------------------------------------------
def build_true_ortho(ctx, res, sim, P: np.ndarray, cell: float,
                     out_path: str, block: int = 1024,
                     k_blend: int = 2) -> Optional[dict]:
    """Write an RGBA orthomosaic GeoTIFF with feathered multi-view blending
    and global colour balancing. Returns stats dict, or None."""
    import rasterio
    from scipy.ndimage import map_coordinates
    from scipy.spatial import cKDTree

    cams = gather_cameras(res, sim, ctx.workdir, ctx)
    if len(cams) < 1:
        ctx.log("No projectable cameras (poses/intrinsics/images missing).", "warn")
        return None

    minx, miny, maxx, maxy = raster.extent(P)
    nx, ny = raster.grid_shape(P, cell)
    origin = (minx, maxy)

    # surface heights at native cloud resolution (project-CRS frame, no
    # vertical-datum shift — the projection must stay in the georef frame)
    native_cell = raster.pick_native_cell(P, cell)
    zgrid = raster.native_grid(P, P[:, 2], native_cell, reducer="max")
    ratio = cell / native_cell

    centers = np.array([c.center_w[:2] for c in cams])
    tree = cKDTree(centers)
    k = min(4, len(cams))
    inv = sim.inverse()
    cache = _ImageCache()

    ctx.log("Colour balancing: estimating per-photo exposure gains from "
            "overlaps...", "info")
    gains = estimate_gains(cams, P, inv, tree, cache)
    spread = float(gains.max() - gains.min())
    ctx.log(f"Colour gains solved for {len(cams)} photos "
            f"(spread {spread:.2f}).", "info")
    ctx.progress(8)

    profile = raster.geotiff_profile(ctx.chunk, origin, cell, nx, ny,
                                     count=4, dtype="uint8", nodata=None)
    profile["photometric"] = "RGB"
    n_filled = 0
    with rasterio.open(out_path, "w", **profile) as dst:
        for r0 in range(0, ny, block):
            if ctx.cancelled:
                return None
            r1 = min(r0 + block, ny)
            nrows = r1 - r0
            xs = minx + (np.arange(nx) + 0.5) * cell
            ys = maxy - (np.arange(r0, r1) + 0.5) * cell
            X, Y = np.meshgrid(xs, ys)                       # (nrows, nx)
            rows_n = (np.arange(r0, r1) + 0.5) * ratio - 0.5
            cols_n = (np.arange(nx) + 0.5) * ratio - 0.5
            rr, cc = np.meshgrid(rows_n, cols_n, indexing="ij")
            Z = map_coordinates(zgrid, np.stack([rr.ravel(), cc.ravel()]),
                                order=1, mode="nearest")
            world = np.column_stack([X.ravel(), Y.ravel(), Z])
            local = inv.apply(world)

            _, nearest = tree.query(world[:, :2], k=k)
            nearest = np.atleast_2d(nearest.reshape(len(world), -1))

            # sample the two best views per cell (nearest-camera order)
            n = len(world)
            rgb1 = np.zeros((n, 3), np.float32); w1 = np.zeros(n, np.float32)
            rgb2 = np.zeros((n, 3), np.float32); w2 = np.zeros(n, np.float32)
            n_views = np.zeros(n, np.uint8)
            for attempt in range(k):
                active = np.where(n_views < k_blend)[0]
                if not len(active):
                    break
                cand = nearest[active, attempt]
                for ci in np.unique(cand):
                    cam = cams[ci]
                    sel = active[cand == ci]
                    u, v, ok = _project_cam(cam, local[sel])
                    if not ok.any():
                        continue
                    img = cache.get(cam.path)
                    if img is None:
                        continue
                    hit = sel[ok]
                    col = _bilinear(img, u[ok], v[ok]) * gains[ci]
                    d2 = ((world[hit, 0] - cam.center_w[0]) ** 2
                          + (world[hit, 1] - cam.center_w[1]) ** 2)
                    wgt = (_border_feather(cam, u[ok], v[ok])
                           / (d2 + 100.0)).astype(np.float32)
                    first = n_views[hit] == 0
                    h1, h2 = hit[first], hit[~first]
                    rgb1[h1], w1[h1] = col[first], wgt[first]
                    rgb2[h2], w2[h2] = col[~first], wgt[~first]
                    n_views[hit] += 1

            # feather seams only where the two views agree; where they
            # disagree (occlusion, surface-model error) pick one decisively
            # — blending disagreeing content produces ghosting.
            got1, got2 = w1 > 0, w2 > 0
            rgb = rgb1.copy()
            both = got1 & got2
            if both.any():
                diff = np.abs(rgb1[both] - rgb2[both]).max(axis=1)
                agree = diff < 30.0
                bidx = np.where(both)[0]
                aidx = bidx[agree]
                ws = w1[aidx] + w2[aidx]
                rgb[aidx] = (rgb1[aidx] * w1[aidx, None]
                             + rgb2[aidx] * w2[aidx, None]) / ws[:, None]
                didx = bidx[~agree]
                take2 = w2[didx] > w1[didx]
                rgb[didx[take2]] = rgb2[didx[take2]]
            only2 = ~got1 & got2
            rgb[only2] = rgb2[only2]
            valid = got1 | got2
            alpha = np.where(valid, 255, 0).astype(np.uint8)

            n_filled += int(valid.sum())
            band = rgb.reshape(nrows, nx, 3)
            for b in range(3):
                dst.write(np.clip(np.rint(band[:, :, b]), 0, 255).astype(np.uint8),
                          indexes=b + 1, window=((r0, r1), (0, nx)))
            dst.write(alpha.reshape(nrows, nx), indexes=4,
                      window=((r0, r1), (0, nx)))
            ctx.progress(10 + int(85 * r1 / ny))

    return {"w": nx, "h": ny, "gsd_m": round(cell, 4),
            "cameras": len(cams), "blend_views": k_blend,
            "coverage_pct": round(100.0 * n_filled / max(nx * ny, 1), 1)}
