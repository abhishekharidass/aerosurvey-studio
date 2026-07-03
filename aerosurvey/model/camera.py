"""Camera (photo) data model."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Camera:
    """A single source photo and everything we know / estimate about it."""

    id: int
    path: str
    width: int = 0
    height: int = 0

    # From EXIF / XMP
    make: str = ""
    model: str = ""
    focal_mm: Optional[float] = None
    datetime: str = ""
    # Geotag (WGS84) as recorded by the drone
    lat: Optional[float] = None
    lon: Optional[float] = None
    alt: Optional[float] = None
    # Gimbal / platform attitude in degrees (from XMP, if present)
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None

    # Solved by aerial triangulation (project CRS). None until aligned.
    est_x: Optional[float] = None
    est_y: Optional[float] = None
    est_z: Optional[float] = None
    aligned: bool = False
    enabled: bool = True

    @property
    def filename(self) -> str:
        return os.path.basename(self.path)

    @property
    def has_geotag(self) -> bool:
        return self.lat is not None and self.lon is not None

    def summary(self) -> str:
        parts = [self.filename]
        if self.width and self.height:
            parts.append(f"{self.width}x{self.height}")
        if self.has_geotag:
            parts.append(f"{self.lat:.6f}, {self.lon:.6f}")
        return "  |  ".join(parts)
