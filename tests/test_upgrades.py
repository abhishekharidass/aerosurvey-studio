"""Tests for the GSD/ortho/classification/auto-marking upgrades."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aerosurvey.core import gsd as gsdmod
from aerosurvey.model.camera import Camera
from aerosurvey.model.project import Chunk, ProcessingSettings
from aerosurvey.pipeline import classify, gcp_project, raster, recon
from aerosurvey.pipeline.georef import Similarity, umeyama_similarity


# ---------------------------------------------------------------------------
# GSD estimation
# ---------------------------------------------------------------------------
def _m3e_cam(cid=1, **kw):
    d = dict(id=cid, path=f"img{cid}.jpg", width=5280, height=3956,
             make="DJI", model="M3E", focal_mm=12.29, lat=22.58, lon=88.49,
             alt=108.5, rel_alt=100.0, yaw=0.0, pitch=-90.0, enabled=True)
    d.update(kw)
    return Camera(**d)


def test_focal_px_from_sensor_db():
    f = gsdmod.focal_px(_m3e_cam())
    assert f == pytest.approx(5280 * 12.29 / 17.3, rel=1e-6)


def test_focal_px_from_35mm_equiv():
    cam = _m3e_cam(make="Unknown", model="NoDB", focal_mm=None, focal35_mm=24.0)
    assert gsdmod.focal_px(cam) == pytest.approx(5280 * 24 / 36.0)


def test_estimate_gsd_exif_path():
    ch = Chunk()
    ch.cameras = [_m3e_cam(i) for i in range(1, 4)]
    g = gsdmod.estimate_gsd(ch)
    # 100 m AGL, f ~3750 px -> ~2.7 cm
    assert g == pytest.approx(100.0 / (5280 * 12.29 / 17.3), rel=1e-6)
    assert 0.02 < g < 0.04


def test_estimate_gsd_prefers_reconstruction_height():
    ch = Chunk()
    ch.cameras = [_m3e_cam(1, est_z=150.0)]
    g = gsdmod.estimate_gsd(ch, ground_z=30.0, f_px_by_cam={1: 4000.0})
    assert g == pytest.approx(120.0 / 4000.0)


# ---------------------------------------------------------------------------
# Settings round-trip
# ---------------------------------------------------------------------------
def test_settings_serialise_roundtrip():
    ch = Chunk()
    ch.settings.ortho_gsd_mode = "custom"
    ch.settings.ortho_gsd = 0.03
    ch.settings.dense_quality = "ultra"
    d = ch.to_dict()
    ch2 = Chunk.from_dict(d)
    assert ch2.settings.ortho_gsd_mode == "custom"
    assert ch2.settings.ortho_gsd == 0.03
    assert ch2.settings.dense_quality == "ultra"


def test_settings_ignores_unknown_keys():
    s = ProcessingSettings.from_dict({"ortho_gsd": 0.1, "bogus": 1})
    assert s.ortho_gsd == 0.1


# ---------------------------------------------------------------------------
# Raster helpers
# ---------------------------------------------------------------------------
def test_native_grid_and_interp_write(tmp_path):
    rng = np.random.default_rng(0)
    P = np.column_stack([rng.uniform(0, 50, 20000), rng.uniform(0, 50, 20000)])
    z = 10 + 0.1 * P[:, 0]
    pts = np.column_stack([P, z])
    cell = 0.05
    native = raster.pick_native_cell(pts, cell)
    assert native > cell                      # density can't support 5 cm
    nat = raster.native_grid(pts, z, native, reducer="max")
    assert not np.isnan(nat).any()
    nx, ny = raster.grid_shape(pts, cell)
    out = str(tmp_path / "dsm.tif")
    raster.write_interp_raster(out, [nat], None, (0.0, 50.0), native, cell,
                               nx, ny, dtype="float32", nodata=np.nan,
                               z_offset=2.0)
    import rasterio
    with rasterio.open(out) as src:
        assert (src.width, src.height) == (nx, ny)
        assert src.transform.a == pytest.approx(cell)
        arr = src.read(1)
    # plane z = 10 + 0.1x, shifted down by z_offset=2
    mid = arr[ny // 2, nx // 2]
    assert mid == pytest.approx(10 + 0.1 * 25 - 2.0, abs=0.3)


def test_native_grid_orderby_top_surface_wins():
    # two points in the same cell: colour of the higher one must win
    pts = np.array([[1.2, 1.2, 5.0], [1.3, 1.3, 20.0], [3.0, 3.0, 1.0]])
    col = np.array([100.0, 200.0, 50.0])
    arr = raster.native_grid(pts, col, cell=1.0, orderby=pts[:, 2])
    minx, maxy = 1.0, 3.0
    ix, iy = raster.cell_indices(pts[:1], minx, maxy, 1.0, *raster.grid_shape(pts, 1.0))
    assert arr[iy[0], ix[0]] == 200.0


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def _synthetic_town(seed=3):
    rng = np.random.default_rng(seed)
    pts, cols, truth = [], [], []
    n = 30000
    gx = rng.uniform(0, 100, n)
    gy = rng.uniform(0, 100, n)
    gz = 5 + 0.02 * gx + rng.normal(0, 0.03, n)
    pts.append(np.column_stack([gx, gy, gz]))
    cols.append(np.tile([120, 100, 80], (n, 1)))
    truth.append(np.full(n, 2))
    # building: 20x20, 8 m tall, flat roof
    nb = 6000
    bx = rng.uniform(30, 50, nb)
    by = rng.uniform(30, 50, nb)
    bz = 5 + 0.02 * bx + 8 + rng.normal(0, 0.04, nb)
    pts.append(np.column_stack([bx, by, bz]))
    cols.append(np.tile([180, 180, 185], (nb, 1)))
    truth.append(np.full(nb, 6))
    # trees: rough green blobs ~7 m
    nt = 4000
    tx = rng.uniform(70, 90, nt)
    ty = rng.uniform(70, 90, nt)
    tz = 5 + 0.02 * tx + rng.uniform(3, 7, nt)
    pts.append(np.column_stack([tx, ty, tz]))
    cols.append(np.tile([40, 130, 40], (nt, 1)))
    truth.append(np.full(nt, 5))
    return (np.vstack(pts), np.vstack(cols).astype(np.uint8),
            np.concatenate(truth))


def test_classifier_ground_building_veg():
    P, C, truth = _synthetic_town()
    cls = classify.classify_cloud(P, C)
    ground_acc = (cls[truth == 2] == 2).mean()
    bld = cls[truth == 6]
    veg = cls[truth == 5]
    assert ground_acc > 0.9
    assert (bld == 6).mean() > 0.75
    assert np.isin(veg, (3, 4, 5)).mean() > 0.75


def test_noise_becomes_class_7():
    P, C, _ = _synthetic_town()
    flyers = np.array([[50, 50, 80], [20, 20, -40], [60, 60, 120]])
    P2 = np.vstack([P, flyers])
    C2 = np.vstack([C, np.zeros((3, 3), np.uint8)])
    cls = classify.classify_cloud(P2, C2)
    assert (cls[-3:] == 7).all()


# ---------------------------------------------------------------------------
# GCP auto-marking
# ---------------------------------------------------------------------------
def test_exif_rotation_nadir_north():
    R = gcp_project._exif_rotation(0.0, -90.0)
    # point 10 m north of, 50 m below the camera at origin
    Xc = R @ np.array([0.0, 10.0, -50.0])
    assert Xc[2] > 0                       # in front
    assert Xc[1] < 0                       # north -> up in the image (−v)
    assert abs(Xc[0]) < 1e-9


def test_exif_rotation_tilt_towards_heading():
    R_nadir = gcp_project._exif_rotation(0.0, -90.0)
    R_tilt = gcp_project._exif_rotation(0.0, -60.0)
    # a point ~31 deg ahead (north) of straight-down
    ahead = np.array([0.0, 30.0, -50.0])
    v_nadir = (R_nadir @ ahead)[1] / (R_nadir @ ahead)[2]
    v_tilt = (R_tilt @ ahead)[1] / (R_tilt @ ahead)[2]
    assert v_nadir == pytest.approx(-0.6)  # well above centre when nadir
    # tilting 30 deg towards the heading brings it almost onto the axis
    assert abs(v_tilt) < 0.05


def test_auto_mark_exif_places_center(tmp_path):
    ch = Chunk()
    ch.crs_mode = "local"                  # identity CRS: world == lon/lat won't fly
    # use a local frame: fake geotags equal to local coords via CrsTransform(None)
    cam = _m3e_cam(1, lat=100.0, lon=100.0, alt=120.0)  # tf identity: x=lon,y=lat
    ch.cameras = [cam]
    g = ch.add_gcp("T1", x=100.0, y=100.0, z=20.0)      # directly beneath
    n, method = gcp_project.auto_mark(ch, str(tmp_path))
    assert method == "exif"
    assert n == 1
    obs = g.observations[1]
    assert obs.px == pytest.approx(5280 / 2, abs=1.0)
    assert obs.py == pytest.approx(3956 / 2, abs=1.0)


def test_auto_mark_reconstruction(tmp_path):
    from aerosurvey.pipeline.colmap import CameraPose, ColmapResult
    ch = Chunk()
    cam = _m3e_cam(1)
    ch.cameras = [cam]
    g = ch.add_gcp("T1", x=500100.0, y=4000100.0, z=50.0)

    # camera 100 m above the GCP, looking straight down, in a world frame
    # that is shifted from the local frame by (500000, 4000000, 0)
    sim = Similarity(1.0, np.eye(3), np.array([500000.0, 4000000.0, 0.0]))
    f = 3700.0
    K = np.array([[f, 0, 5280 / 2], [0, f, 3956 / 2], [0, 0, 1]])
    R = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])  # nadir, det=+1
    C_local = np.array([100.0, 100.0, 150.0])
    t = -R @ C_local
    qvec = _rot_to_qvec(R)
    res = ColmapResult()
    res.poses = {cam.filename: CameraPose(cam.filename, qvec, t, C_local, 1)}
    res.intrinsics = {1: K}
    ch._colmap_result = res
    ch._georef_sim = sim

    n, method = gcp_project.auto_mark(ch, str(tmp_path))
    assert method == "reconstruction"
    assert n == 1
    obs = g.observations[1]
    assert obs.px == pytest.approx(5280 / 2, abs=0.5)
    assert obs.py == pytest.approx(3956 / 2, abs=0.5)


def _rot_to_qvec(R):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


# ---------------------------------------------------------------------------
# Recon persistence
# ---------------------------------------------------------------------------
def test_sim_save_load_roundtrip(tmp_path):
    src = np.random.default_rng(1).uniform(0, 10, (5, 3))
    sim = umeyama_similarity(src, src * 2.0 + np.array([100.0, 200.0, 5.0]))
    recon.save_sim(str(tmp_path), sim, "test")
    sim2 = recon.load_sim(str(tmp_path))
    assert sim2 is not None
    p = np.array([1.0, 2.0, 3.0])
    assert np.allclose(sim.apply(p), sim2.apply(p))


def test_voxel_downsample_density():
    from aerosurvey.pipeline.stages import _voxel_downsample
    rng = np.random.default_rng(0)
    P = np.column_stack([rng.uniform(0, 100, 200000),
                         rng.uniform(0, 100, 200000),
                         rng.normal(10, 0.05, 200000)])   # 20 pts/m² flat-ish
    C = np.zeros((len(P), 3), np.uint8)
    P2, C2 = _voxel_downsample(P, C, 5.0)
    dens = len(P2) / 1e4
    assert 2.0 < dens <= 6.0                # roughly the requested density
    assert len(C2) == len(P2)
