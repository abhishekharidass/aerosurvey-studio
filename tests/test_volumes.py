"""Tests for DSM volume / stockpile measurement."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aerosurvey.core import volumes as vol


def _write_dsm(path, z, x0=500000.0, y0=2500000.0, cell=0.5, nodata=-9999.0):
    import rasterio
    from rasterio.transform import from_origin
    with rasterio.open(path, "w", driver="GTiff", width=z.shape[1],
                       height=z.shape[0], count=1, dtype="float32",
                       crs="EPSG:32645", nodata=nodata,
                       transform=from_origin(x0, y0, cell, cell)) as dst:
        dst.write(z.astype(np.float32), 1)
    return path


@pytest.fixture
def mound_dsm(tmp_path):
    """Flat ground at 100 m with a 10x10 m mound of +5 m in the middle.

    Grid: 100x100 cells @ 0.5 m => 50x50 m, origin (500000, 2500000) top-left.
    Mound occupies x 500020..500030, y 2499970..2499980.
    """
    z = np.full((100, 100), 100.0)
    z[40:60, 40:60] = 105.0
    return _write_dsm(str(tmp_path / "dsm.tif"), z)


def test_mound_cut_volume(mound_dsm):
    # polygon around the mound with margin on flat ground
    poly = [(500015, 2499965), (500035, 2499965),
            (500035, 2499985), (500015, 2499985)]
    r = vol.measure_volume(mound_dsm, poly, "lowest")
    # 10x10 m x 5 m = 500 m3
    assert r.cut_m3 == pytest.approx(500.0, rel=0.02)
    assert r.fill_m3 == pytest.approx(0.0, abs=1.0)
    assert r.net_m3 == pytest.approx(500.0, rel=0.02)
    assert r.area_m2 == pytest.approx(400.0)
    assert r.coverage == pytest.approx(1.0, abs=0.02)
    assert r.base_z_min == pytest.approx(100.0)
    assert not r.warnings
    assert "Cut" in r.summary()


def test_pit_fill_volume(tmp_path):
    z = np.full((100, 100), 100.0)
    z[40:60, 40:60] = 96.0          # 10x10 m pit, 4 m deep
    dsm = _write_dsm(str(tmp_path / "pit.tif"), z)
    poly = [(500015, 2499965), (500035, 2499965),
            (500035, 2499985), (500015, 2499985)]
    r = vol.measure_volume(dsm, poly, "lowest")
    # base = lowest boundary = 100 (boundary is on flat ground)
    assert r.fill_m3 == pytest.approx(400.0, rel=0.02)
    assert r.cut_m3 == pytest.approx(0.0, abs=1.0)
    assert r.net_m3 == pytest.approx(-400.0, rel=0.02)


def test_fit_plane_on_slope(tmp_path):
    # sloped terrain: z = 100 + 0.1 * (x - x0); a fit base should zero it out
    jj, ii = np.meshgrid(np.arange(100), np.arange(100))
    z = 100.0 + 0.1 * (jj * 0.5)
    dsm = _write_dsm(str(tmp_path / "slope.tif"), z)
    poly = [(500010, 2499960), (500040, 2499960),
            (500040, 2499990), (500010, 2499990)]
    r_fit = vol.measure_volume(dsm, poly, "fit")
    assert abs(r_fit.net_m3) < 5.0          # plane fits the slope
    r_low = vol.measure_volume(dsm, poly, "lowest")
    assert r_low.cut_m3 > 100.0             # lowest-point base sees a wedge


def test_custom_base(mound_dsm):
    poly = [(500020, 2499970), (500030, 2499970),
            (500030, 2499980), (500020, 2499980)]
    r = vol.measure_volume(mound_dsm, poly, "custom", custom_z=105.0)
    # polygon exactly on the mound top, base at the top -> nothing
    assert r.cut_m3 == pytest.approx(0.0, abs=1.0)


def test_nodata_coverage_warning(tmp_path):
    z = np.full((100, 100), 100.0)
    z[45:55, 45:55] = -9999.0       # hole in the middle
    dsm = _write_dsm(str(tmp_path / "holes.tif"), z)
    poly = [(500020, 2499970), (500030, 2499970),
            (500030, 2499980), (500020, 2499980)]
    r = vol.measure_volume(dsm, poly, "lowest")
    assert r.coverage < 0.8
    assert any("DSM data" in w for w in r.warnings)


def test_validation_errors(mound_dsm):
    with pytest.raises(ValueError):
        vol.measure_volume(mound_dsm, [(0, 0), (1, 1)])          # too few pts
    with pytest.raises(ValueError):
        vol.measure_volume(mound_dsm, [(0, 0), (1, 0), (1, 1)])  # outside DSM
    with pytest.raises(ValueError):
        vol.measure_volume(mound_dsm,
                           [(500015, 2499965), (500035, 2499965),
                            (500035, 2499985)], base_mode="nope")


def test_polygon_area_shoelace():
    sq = np.array([(0, 0), (10, 0), (10, 10), (0, 10)], float)
    assert vol.polygon_area(sq) == 100.0
