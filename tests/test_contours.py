"""Tests for vector contour generation and the SHP/DXF/GeoJSON writers."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aerosurvey.core import contours as cont
from aerosurvey.model.project import Chunk


@pytest.fixture
def cone_dem(tmp_path):
    """A 200x200 cone: 100 m at the rim -> 120 m at the centre, EPSG:32645."""
    import rasterio
    from rasterio.transform import from_origin
    n = 200
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(xx - n / 2, yy - n / 2) / (n / 2)
    z = (100.0 + 20.0 * (1.0 - np.clip(r, 0, 1))).astype(np.float32)
    path = str(tmp_path / "dem.tif")
    tr = from_origin(500000, 2500000, 1.0, 1.0)
    with rasterio.open(path, "w", driver="GTiff", width=n, height=n, count=1,
                       dtype="float32", crs="EPSG:32645", transform=tr,
                       nodata=-9999.0) as dst:
        dst.write(z, 1)
    return path


def test_generate_contours_levels_and_coords(cone_dem):
    cs = cont.generate_contours(cone_dem, interval=5.0, smooth_sigma=0)
    levels = [lv for lv, _, _ in cs]
    # cone spans 100..120 -> levels at 105/110/115 at least (100 clipped rim)
    assert {105.0, 110.0, 115.0}.issubset(set(levels))
    for lv, is_index, lines in cs:
        assert is_index == (round(lv / 5.0) % cont.INDEX_EVERY == 0)
        for xy in lines:
            # all coordinates inside the raster footprint
            assert xy[:, 0].min() >= 500000 and xy[:, 0].max() <= 500200
            assert xy[:, 1].min() >= 2499800 and xy[:, 1].max() <= 2500000


def test_contour_circle_radius(cone_dem):
    """The 110 m contour is a circle of known radius (~50 m)."""
    cs = cont.generate_contours(cone_dem, interval=10.0, smooth_sigma=0)
    ring = next(lines for lv, _, lines in cs if lv == 110.0)
    xy = max(ring, key=len)
    cx, cy = xy[:, 0].mean(), xy[:, 1].mean()
    rad = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
    assert rad.mean() == pytest.approx(50.0, abs=2.0)


def test_interval_validation(cone_dem):
    with pytest.raises(ValueError):
        cont.generate_contours(cone_dem, interval=-1)
    with pytest.raises(ValueError):
        cont.generate_contours(cone_dem, interval=0.001)  # >2000 levels


def test_writers_roundtrip(cone_dem, tmp_path):
    cs = cont.generate_contours(cone_dem, interval=5.0, smooth_sigma=0)

    # GeoJSON
    gj = cont.write_geojson(cs, str(tmp_path / "c.geojson"), epsg=32645)
    doc = json.load(open(gj))
    assert doc["type"] == "FeatureCollection" and doc["features"]
    assert "32645" in doc["crs"]["properties"]["name"]
    f0 = doc["features"][0]
    assert f0["geometry"]["type"] == "LineString"
    assert isinstance(f0["properties"]["elevation"], float)

    # Shapefile (read back with pyshp)
    import shapefile
    shp = cont.write_shapefile(cs, str(tmp_path / "c.shp"), epsg=32645)
    with shapefile.Reader(shp) as r:
        assert r.numRecords == sum(len(ls) for _, _, ls in cs)
        rec = r.record(0)
        assert rec["ELEV"] in [lv for lv, _, _ in cs]
        assert r.shape(0).shapeType == shapefile.POLYLINEZ
        # z values carried on the geometry
        assert r.shape(0).z[0] == pytest.approx(rec["ELEV"])
    assert os.path.exists(str(tmp_path / "c.prj"))

    # DXF: structurally sound and elevations present
    dxf = cont.write_dxf(cs, str(tmp_path / "c.dxf"))
    text = open(dxf).read()
    assert text.count("POLYLINE") == sum(len(ls) for _, _, ls in cs)
    assert "CONTOUR_MAJOR" in text and "CONTOUR_MINOR" in text
    assert text.rstrip().endswith("EOF")


def test_export_contours_orchestrator(cone_dem, tmp_path):
    ch = Chunk()
    ch.crs_mode = "epsg"
    ch.epsg = 32645
    ch.outputs.dtm = cone_dem
    out = str(tmp_path / "out")
    written = cont.export_contours(ch, out, 5.0, "dtm",
                                   ("shp", "dxf", "geojson"))
    assert len(written) == 3
    assert all(os.path.exists(p) for p in written)
    with pytest.raises(FileNotFoundError):
        cont.export_contours(ch, out, 5.0, "dsm")
