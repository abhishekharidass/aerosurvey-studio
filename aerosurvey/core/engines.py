"""Detect the external photogrammetry engines the pipeline can orchestrate.

Nothing here *runs* the engines yet — the scaffold reports availability so the
UI can show which real backends are present. Each stage falls back to a
simulated implementation when its engine is missing.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import List


@dataclass
class Engine:
    key: str
    name: str
    role: str
    executables: List[str]

    @property
    def path(self) -> str:
        for exe in self.executables:
            found = shutil.which(exe)
            if found:
                return found
        return ""

    @property
    def available(self) -> bool:
        return bool(self.path)


ENGINES: List[Engine] = [
    Engine("colmap", "COLMAP", "Feature matching + Structure-from-Motion / AT",
           ["colmap", "colmap.exe"]),
    Engine("openmvs", "OpenMVS", "Multi-View Stereo dense reconstruction",
           ["DensifyPointCloud", "DensifyPointCloud.exe"]),
    Engine("pdal", "PDAL", "Point-cloud classification & filtering",
           ["pdal", "pdal.exe"]),
    Engine("gdal", "GDAL", "DSM/DTM rasterisation & orthomosaic export",
           ["gdal_translate", "gdalwarp", "gdal_translate.exe"]),
    Engine("odm", "OpenDroneMap", "All-in-one alternative pipeline",
           ["odm", "run.py"]),
]


def engine_status() -> List[dict]:
    return [
        {"key": e.key, "name": e.name, "role": e.role,
         "available": e.available, "path": e.path}
        for e in ENGINES
    ]


def any_available() -> bool:
    return any(e.available for e in ENGINES)
