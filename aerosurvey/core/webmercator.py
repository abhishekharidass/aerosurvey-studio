"""Web-mercator (EPSG:3857) slippy-map math for the basemap layer.

Pure functions, no Qt — the map view builds on these and the tests exercise
them headlessly. Conventions follow the OSM tile scheme: tile (0,0) is the
north-west corner, 256 px tiles, zoom z has 2^z tiles per axis.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Tuple

R = 6378137.0                       # WGS84 spherical radius used by EPSG:3857
ORIGIN = math.pi * R                # half the mercator world width, ~20037508 m
TILE_PX = 256
MAX_LAT = 85.05112878               # mercator singularity clamp


def lonlat_to_mercator(lon: float, lat: float) -> Tuple[float, float]:
    """WGS84 degrees -> EPSG:3857 metres."""
    lat = min(max(lat, -MAX_LAT), MAX_LAT)
    mx = math.radians(lon) * R
    my = math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0)) * R
    return mx, my


def mercator_to_lonlat(mx: float, my: float) -> Tuple[float, float]:
    """EPSG:3857 metres -> WGS84 degrees."""
    lon = math.degrees(mx / R)
    lat = math.degrees(2.0 * math.atan(math.exp(my / R)) - math.pi / 2.0)
    return lon, lat


def resolution(zoom: int) -> float:
    """Metres per tile pixel at a zoom level (at the equator)."""
    return 2.0 * ORIGIN / (TILE_PX * (1 << zoom))


def zoom_for_resolution(metres_per_px: float, zmin: int = 1, zmax: int = 19) -> int:
    """The zoom whose tile resolution best matches a target metres/pixel."""
    if metres_per_px <= 0:
        return zmax
    z = math.log2(2.0 * ORIGIN / (TILE_PX * metres_per_px))
    return min(max(int(round(z)), zmin), zmax)


def tile_size_m(zoom: int) -> float:
    """Side length of one tile in mercator metres."""
    return 2.0 * ORIGIN / (1 << zoom)


def mercator_to_tile(mx: float, my: float, zoom: int) -> Tuple[int, int]:
    """Mercator metres -> (tx, ty) tile indices (OSM scheme, y from north)."""
    size = tile_size_m(zoom)
    tx = int(math.floor((mx + ORIGIN) / size))
    ty = int(math.floor((ORIGIN - my) / size))
    n = (1 << zoom) - 1
    return min(max(tx, 0), n), min(max(ty, 0), n)


def tile_bounds(tx: int, ty: int, zoom: int) -> Tuple[float, float, float, float]:
    """Tile indices -> mercator (west, south, east, north)."""
    size = tile_size_m(zoom)
    west = -ORIGIN + tx * size
    north = ORIGIN - ty * size
    return west, north - size, west + size, north


def tiles_in_bounds(west: float, south: float, east: float, north: float,
                    zoom: int, cap: int = 256) -> List[Tuple[int, int]]:
    """All tiles intersecting a mercator rect, capped to a sane count."""
    x0, y0 = mercator_to_tile(west, north, zoom)
    x1, y1 = mercator_to_tile(east, south, zoom)
    tiles = []
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            tiles.append((tx, ty))
            if len(tiles) >= cap:
                return tiles
    return tiles
