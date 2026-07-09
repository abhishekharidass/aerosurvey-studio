"""Project / Chunk containers and JSON (de)serialisation."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional

from .camera import Camera
from .gcp import GCP, Observation


@dataclass
class ProcessingSettings:
    """User-adjustable pipeline parameters (Tools ▸ Processing Settings)."""

    # Output resolution. "auto" = the estimated image GSD; "custom" = the
    # explicit value in metres/pixel.
    ortho_gsd_mode: str = "auto"
    ortho_gsd: float = 0.05
    surface_gsd_mode: str = "auto"      # DSM + DTM
    surface_gsd: float = 0.05
    # Safety cap on raster width/height; the cell size is coarsened to fit.
    max_raster_dim: int = 20000

    # Dense matching quality -> OpenMVS resolution level
    # (ultra = full res, high = 1/2, medium = 1/4, low = 1/8).
    dense_quality: str = "high"
    # Target dense-cloud density in points/m² (0 = keep native density).
    dense_target_density: float = 0.0

    # Point classifier: "rules" (morphological+geometric) or "ml" (Random Forest).
    classifier: str = "rules"

    # SfM feature budget per image (more = denser sparse cloud, slower).
    sfm_max_features: int = 8192

    @classmethod
    def from_dict(cls, d: dict) -> "ProcessingSettings":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


@dataclass
class Outputs:
    """Filesystem locations of processing products for a chunk."""

    sparse_cloud: str = ""
    dense_cloud: str = ""
    classified_cloud: str = ""
    dsm: str = ""
    dtm: str = ""
    orthomosaic: str = ""


@dataclass
class Chunk:
    """A block of photos processed together (Metashape terminology)."""

    name: str = "Chunk 1"
    cameras: List[Camera] = field(default_factory=list)
    gcps: List[GCP] = field(default_factory=list)

    # Coordinate reference. crs_mode in {"local", "utm", "epsg"}.
    crs_mode: str = "local"
    epsg: Optional[int] = None          # explicit EPSG when crs_mode == "epsg"/"utm"
    crs_label: str = "Local coordinates (arbitrary)"
    # Vertical datum: "ellipsoidal" (raw GPS height) or "orthometric" (MSL).
    # Orthometric = ellipsoidal - geoid_separation (N).
    vertical_datum: str = "ellipsoidal"
    geoid_separation: float = 0.0

    # Pipeline status flags
    aligned: bool = False
    optimized: bool = False

    outputs: Outputs = field(default_factory=Outputs)
    stats: dict = field(default_factory=dict)   # processing metrics for the report
    settings: ProcessingSettings = field(default_factory=ProcessingSettings)

    _next_cam_id: int = 1
    _next_gcp_id: int = 1

    # -- cameras ---------------------------------------------------------
    def add_camera(self, path: str) -> Camera:
        cam = Camera(id=self._next_cam_id, path=path)
        self._next_cam_id += 1
        self.cameras.append(cam)
        return cam

    def camera(self, cam_id: int) -> Optional[Camera]:
        return next((c for c in self.cameras if c.id == cam_id), None)

    # -- gcps ------------------------------------------------------------
    def add_gcp(self, label: str, x=0.0, y=0.0, z=0.0, is_check=False) -> GCP:
        gcp = GCP(id=self._next_gcp_id, label=label, x=x, y=y, z=z, is_check=is_check)
        self._next_gcp_id += 1
        self.gcps.append(gcp)
        return gcp

    def gcp(self, gcp_id: int) -> Optional[GCP]:
        return next((g for g in self.gcps if g.id == gcp_id), None)

    def remove_gcp(self, gcp_id: int) -> None:
        self.gcps = [g for g in self.gcps if g.id != gcp_id]

    @property
    def total_observations(self) -> int:
        return sum(g.marked_count for g in self.gcps)

    # -- serialisation ---------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "crs_mode": self.crs_mode,
            "epsg": self.epsg,
            "crs_label": self.crs_label,
            "vertical_datum": self.vertical_datum,
            "geoid_separation": self.geoid_separation,
            "aligned": self.aligned,
            "optimized": self.optimized,
            "next_cam_id": self._next_cam_id,
            "next_gcp_id": self._next_gcp_id,
            "outputs": self.outputs.__dict__,
            "stats": self.stats,
            "settings": self.settings.__dict__,
            "cameras": [c.__dict__ for c in self.cameras],
            "gcps": [
                {
                    **{k: v for k, v in g.__dict__.items() if k != "observations"},
                    "observations": [o.__dict__ for o in g.observations.values()],
                }
                for g in self.gcps
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        ch = cls(name=d.get("name", "Chunk 1"))
        ch.crs_mode = d.get("crs_mode", "local")
        ch.epsg = d.get("epsg")
        ch.crs_label = d.get("crs_label", "Local coordinates (arbitrary)")
        ch.vertical_datum = d.get("vertical_datum", "ellipsoidal")
        ch.geoid_separation = d.get("geoid_separation", 0.0)
        ch.aligned = d.get("aligned", False)
        ch.optimized = d.get("optimized", False)
        ch._next_cam_id = d.get("next_cam_id", 1)
        ch._next_gcp_id = d.get("next_gcp_id", 1)
        ch.outputs = Outputs(**d.get("outputs", {}))
        ch.stats = d.get("stats", {})
        ch.settings = ProcessingSettings.from_dict(d.get("settings", {}))
        for cd in d.get("cameras", []):
            ch.cameras.append(Camera(**cd))
        for gd in d.get("gcps", []):
            obs = gd.pop("observations", [])
            gcp = GCP(**gd)
            for o in obs:
                gcp.observations[o["camera_id"]] = Observation(**o)
            ch.gcps.append(gcp)
        return ch


@dataclass
class Project:
    """Top-level document. Holds one or more chunks; scaffold uses one active."""

    path: str = ""
    chunks: List[Chunk] = field(default_factory=lambda: [Chunk()])
    active_index: int = 0

    @property
    def active(self) -> Chunk:
        return self.chunks[self.active_index]

    @property
    def name(self) -> str:
        import os
        return os.path.splitext(os.path.basename(self.path))[0] if self.path else "Untitled"

    def save(self, path: str) -> None:
        self.path = path
        data = {
            "app": "AeroSurvey Studio",
            "format": 1,
            "active_index": self.active_index,
            "chunks": [c.to_dict() for c in self.chunks],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "Project":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        proj = cls(path=path)
        proj.chunks = [Chunk.from_dict(c) for c in data.get("chunks", [])] or [Chunk()]
        proj.active_index = data.get("active_index", 0)
        return proj
