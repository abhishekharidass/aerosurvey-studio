"""Tests for the web-mercator basemap math."""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aerosurvey.core import webmercator as wm


def test_lonlat_mercator_roundtrip():
    for lon, lat in [(0, 0), (88.49, 22.58), (-122.4, 37.8), (179.9, -84.0)]:
        mx, my = wm.lonlat_to_mercator(lon, lat)
        lon2, lat2 = wm.mercator_to_lonlat(mx, my)
        assert lon2 == pytest.approx(lon, abs=1e-9)
        assert lat2 == pytest.approx(lat, abs=1e-9)


def test_mercator_known_values():
    # null island is the world origin
    assert wm.lonlat_to_mercator(0, 0) == pytest.approx((0.0, 0.0), abs=1e-6)
    # antimeridian hits the mercator edge
    mx, _ = wm.lonlat_to_mercator(180, 0)
    assert mx == pytest.approx(wm.ORIGIN, rel=1e-9)
    # polar clamp keeps the projection finite
    _, my = wm.lonlat_to_mercator(0, 89.9)
    assert math.isfinite(my)


def test_resolution_halves_per_zoom():
    assert wm.resolution(0) == pytest.approx(2 * wm.ORIGIN / 256)
    assert wm.resolution(5) == pytest.approx(wm.resolution(4) / 2)


def test_zoom_for_resolution():
    for z in (3, 10, 16):
        assert wm.zoom_for_resolution(wm.resolution(z)) == z
    assert wm.zoom_for_resolution(1e9) == 1          # coarser than world -> zmin
    assert wm.zoom_for_resolution(1e-6, zmax=19) == 19


def test_tile_indices_and_bounds():
    # zoom 1 has a 2x2 grid; the NE quadrant is tile (1, 0)
    mx, my = wm.lonlat_to_mercator(88.49, 22.58)
    assert wm.mercator_to_tile(mx, my, 1) == (1, 0)
    west, south, east, north = wm.tile_bounds(1, 0, 1)
    assert west == pytest.approx(0.0)
    assert north == pytest.approx(wm.ORIGIN)
    assert east - west == pytest.approx(wm.tile_size_m(1))
    assert west <= mx <= east and south <= my <= north


def test_tile_bounds_cover_world():
    w, s, e, n = wm.tile_bounds(0, 0, 0)
    assert (w, s, e, n) == pytest.approx(
        (-wm.ORIGIN, -wm.ORIGIN, wm.ORIGIN, wm.ORIGIN))


def test_tiles_in_bounds():
    # a small rect around Kolkata at z=12 yields a compact tile set
    mx, my = wm.lonlat_to_mercator(88.49, 22.58)
    tiles = wm.tiles_in_bounds(mx - 2000, my - 2000, mx + 2000, my + 2000, 12)
    assert 1 <= len(tiles) <= 9
    assert wm.mercator_to_tile(mx, my, 12) in tiles
    # cap respected
    world = wm.tiles_in_bounds(-wm.ORIGIN, -wm.ORIGIN, wm.ORIGIN, wm.ORIGIN,
                               10, cap=50)
    assert len(world) == 50
