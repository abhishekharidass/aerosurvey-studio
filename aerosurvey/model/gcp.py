"""Ground Control Point model plus its per-image observations (markers)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class Observation:
    """A GCP marked on a specific image, stored in *pixel* coordinates.

    Pixels are kept as floats so sub-pixel marker placement survives round-trips.
    """

    camera_id: int
    px: float
    py: float


@dataclass
class GCP:
    """A surveyed control point expressed in the project's coordinate system."""

    id: int
    label: str
    x: float = 0.0          # Easting / X / local X
    y: float = 0.0          # Northing / Y / local Y
    z: float = 0.0          # Elevation
    is_check: bool = False  # check point (not used to constrain the solve)
    enabled: bool = True
    # Accuracy (m) — how tightly this anchors the bundle adjustment.
    accuracy: float = 0.02

    observations: Dict[int, Observation] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return "Check" if self.is_check else "Control"

    @property
    def marked_count(self) -> int:
        return len(self.observations)

    def mark(self, camera_id: int, px: float, py: float) -> Observation:
        obs = Observation(camera_id, float(px), float(py))
        self.observations[camera_id] = obs
        return obs

    def unmark(self, camera_id: int) -> None:
        self.observations.pop(camera_id, None)

    def observation(self, camera_id: int) -> Optional[Observation]:
        return self.observations.get(camera_id)

    def world(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)
