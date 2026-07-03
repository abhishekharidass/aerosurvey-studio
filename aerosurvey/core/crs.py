"""Coordinate reference handling: local / UTM / EPSG, backed by pyproj."""
from __future__ import annotations

from typing import Optional, Tuple

try:
    from pyproj import CRS, Transformer
    _HAS_PYPROJ = True
except Exception:  # pragma: no cover
    _HAS_PYPROJ = False

WGS84 = 4326


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the WGS84 UTM zone containing (lon, lat)."""
    zone = int((lon + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


def describe(epsg: Optional[int]) -> str:
    if epsg is None:
        return "Local coordinates (arbitrary)"
    if not _HAS_PYPROJ:
        return f"EPSG:{epsg}"
    try:
        return f"EPSG:{epsg} - {CRS.from_epsg(epsg).name}"
    except Exception:
        return f"EPSG:{epsg}"


def is_valid_epsg(epsg: int) -> bool:
    if not _HAS_PYPROJ:
        return True
    try:
        CRS.from_epsg(epsg)
        return True
    except Exception:
        return False


class CrsTransform:
    """Transform geotags (WGS84 lon/lat/alt) into a target project CRS."""

    def __init__(self, target_epsg: Optional[int]):
        self.target_epsg = target_epsg
        self._t = None
        if _HAS_PYPROJ and target_epsg and target_epsg != WGS84:
            try:
                self._t = Transformer.from_crs(
                    CRS.from_epsg(WGS84), CRS.from_epsg(target_epsg), always_xy=True
                )
            except Exception:
                self._t = None

    def forward(self, lon: float, lat: float, alt: float = 0.0) -> Tuple[float, float, float]:
        """(lon, lat, alt) WGS84 -> (X, Y, Z) in target CRS. Identity if local."""
        if self._t is None:
            return (lon, lat, alt)
        x, y = self._t.transform(lon, lat)
        return (x, y, alt)


def pyproj_available() -> bool:
    return _HAS_PYPROJ
