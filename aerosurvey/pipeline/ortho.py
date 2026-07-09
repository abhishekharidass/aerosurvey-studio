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

    def __init__(self, cap: int = 6):
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


# -- main --------------------------------------------------------------------
def build_true_ortho(ctx, res, sim, P: np.ndarray, cell: float,
                     out_path: str, block: int = 1024) -> Optional[dict]:
    """Write an RGBA orthomosaic GeoTIFF. Returns stats dict, or None."""
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

            rgb = np.zeros((len(world), 3), np.float32)
            alpha = np.zeros(len(world), np.uint8)
            todo = np.arange(len(world))
            for attempt in range(k):
                if not len(todo):
                    break
                cand = nearest[todo, attempt]
                for ci in np.unique(cand):
                    cam = cams[ci]
                    sel = todo[cand == ci]
                    Xc = local[sel] @ cam.R.T + cam.t
                    zc = Xc[:, 2]
                    ok = zc > 1e-6
                    uv = (Xc[:, :2] / np.maximum(zc, 1e-9)[:, None])
                    u = cam.K[0, 0] * uv[:, 0] + cam.K[0, 2]
                    v = cam.K[1, 1] * uv[:, 1] + cam.K[1, 2]
                    w, h = cam.wh
                    ok &= (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
                    if not ok.any():
                        continue
                    img = cache.get(cam.path)
                    if img is None:
                        continue
                    hit = sel[ok]
                    rgb[hit] = _bilinear(img, u[ok], v[ok])
                    alpha[hit] = 255
                todo = todo[alpha[todo] == 0]

            n_filled += int((alpha == 255).sum())
            band = rgb.reshape(nrows, nx, 3)
            for b in range(3):
                dst.write(np.clip(np.rint(band[:, :, b]), 0, 255).astype(np.uint8),
                          indexes=b + 1, window=((r0, r1), (0, nx)))
            dst.write(alpha.reshape(nrows, nx), indexes=4,
                      window=((r0, r1), (0, nx)))
            ctx.progress(10 + int(85 * r1 / ny))

    return {"w": nx, "h": ny, "gsd_m": round(cell, 4),
            "cameras": len(cams), "coverage_pct":
            round(100.0 * n_filled / max(nx * ny, 1), 1)}
