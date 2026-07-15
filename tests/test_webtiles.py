"""Tests for the XYZ tile / KML superoverlay export."""
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aerosurvey.core import webmercator as wm
from aerosurvey.core import webtiles


@pytest.fixture
def rgba_ortho(tmp_path):
    """A 512x512 RGBA ortho at 0.2 m GSD in EPSG:32645 with a hole."""
    import rasterio
    from rasterio.transform import from_origin
    n = 512
    rgba = np.zeros((4, n, n), np.uint8)
    rgba[0] = 200; rgba[1] = 120; rgba[2] = 60; rgba[3] = 255
    rgba[3, :64, :64] = 0                       # transparent corner
    path = str(tmp_path / "ortho.tif")
    with rasterio.open(path, "w", driver="GTiff", width=n, height=n, count=4,
                       dtype="uint8", crs="EPSG:32645",
                       transform=from_origin(500000, 2500000, 0.2, 0.2)) as dst:
        dst.write(rgba)
    return path


def test_export_tiles_structure(rgba_ortho, tmp_path):
    out = str(tmp_path / "tiles")
    logs = []
    meta = webtiles.export_web_tiles(rgba_ortho, out, kml=True,
                                     log=lambda m, lvl="info": logs.append(m))
    assert meta["tiles"] > 0
    assert meta["min_zoom"] <= meta["max_zoom"]
    # ~0.2 m GSD -> max zoom around 19-20 in mercator
    assert 18 <= meta["max_zoom"] <= 21

    # metadata file round-trips
    meta2 = json.load(open(os.path.join(out, "tiles.json")))
    assert meta2["tiles"] == meta["tiles"]

    # the max-zoom tile containing the raster centre exists and is a PNG
    from PIL import Image
    left, bottom, right, top = meta["bounds_3857"]
    cx, cy = (left + right) / 2, (bottom + top) / 2
    z = meta["max_zoom"]
    tx, ty = wm.mercator_to_tile(cx, cy, z)
    tile_png = os.path.join(out, str(z), str(tx), f"{ty}.png")
    assert os.path.exists(tile_png)
    img = Image.open(tile_png)
    assert img.size == (256, 256) and img.mode == "RGBA"
    arr = np.asarray(img)
    # the pixel at the raster centre is opaque and carries our colour
    # (the tile itself may extend past the raster edge -> transparent rim)
    tw, ts, te, tn = wm.tile_bounds(tx, ty, z)
    px = int((cx - tw) / (te - tw) * 255)
    py = int((tn - cy) / (tn - ts) * 255)
    assert arr[py, px, 3] == 255
    assert abs(int(arr[py, px, 0]) - 200) <= 2


def test_kml_superoverlay(rgba_ortho, tmp_path):
    out = str(tmp_path / "tiles")
    meta = webtiles.export_web_tiles(rgba_ortho, out, kml=True)
    doc = os.path.join(out, "doc.kml")
    assert os.path.exists(doc)
    root = ET.parse(doc).getroot()          # valid XML
    ns = {"k": "http://www.opengis.net/kml/2.2"}
    links = root.findall(".//k:NetworkLink", ns)
    assert links
    # every linked child KML file exists
    for link in links:
        href = link.find(".//k:href", ns).text
        assert os.path.exists(os.path.join(out, href))
    # spot-check one per-tile KML: overlay box matches the tile bounds
    z = meta["min_zoom"]
    zdir = os.path.join(out, str(z))
    tx = sorted(os.listdir(zdir))[0]
    ty = sorted(f for f in os.listdir(os.path.join(zdir, tx))
                if f.endswith(".kml"))[0]
    tkml = ET.parse(os.path.join(zdir, tx, ty)).getroot()
    box = tkml.find(".//k:GroundOverlay/k:LatLonBox", ns)
    w, s, e, n = webtiles._lonlat_box(int(tx), int(ty.split(".")[0]), z)
    assert float(box.find("k:west", ns).text) == pytest.approx(w, abs=1e-9)
    assert float(box.find("k:north", ns).text) == pytest.approx(n, abs=1e-9)
    # icon reference points at the sibling PNG
    href = tkml.find(".//k:GroundOverlay//k:href", ns).text
    assert os.path.exists(os.path.join(zdir, tx, href))


def test_custom_zoom_range_and_no_kml(rgba_ortho, tmp_path):
    out = str(tmp_path / "tiles")
    meta = webtiles.export_web_tiles(rgba_ortho, out, min_zoom=16, max_zoom=17,
                                     kml=False)
    assert (meta["min_zoom"], meta["max_zoom"]) == (16, 17)
    assert not os.path.exists(os.path.join(out, "doc.kml"))
    zooms = sorted(d for d in os.listdir(out)
                   if os.path.isdir(os.path.join(out, d)))
    assert zooms == ["16", "17"]


def test_no_crs_raises(tmp_path):
    import rasterio
    path = str(tmp_path / "nocrs.tif")
    with rasterio.open(path, "w", driver="GTiff", width=8, height=8, count=3,
                       dtype="uint8") as dst:
        dst.write(np.zeros((3, 8, 8), np.uint8))
    with pytest.raises(ValueError):
        webtiles.export_web_tiles(path, str(tmp_path / "t"))
