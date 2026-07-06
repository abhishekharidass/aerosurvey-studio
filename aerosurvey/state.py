"""Central application state + signals shared by every panel.

Panels never talk to each other directly; they mutate state through this
object and react to its signals. This keeps the UI loosely coupled.
"""
from __future__ import annotations

import os
import tempfile
from typing import List, Optional

from PySide6.QtCore import QObject, Signal

from .core import crs as crsmod
from .core import exif as eximod
from .model import Camera, GCP, Project


class AppState(QObject):
    project_changed = Signal()                 # new / opened / crs changed
    cameras_changed = Signal()                 # photo set changed
    gcps_changed = Signal()                    # gcp set or values changed
    observations_changed = Signal()            # a marker was added/moved/removed
    active_camera_changed = Signal(int)        # camera id, or -1
    active_gcp_changed = Signal(int)           # gcp id, or -1
    outputs_changed = Signal()                 # pipeline products changed
    log = Signal(str, str)                     # message, level
    dirty_changed = Signal(bool)

    def __init__(self):
        super().__init__()
        self.project = Project()
        self._active_camera_id = -1
        self._active_gcp_id = -1
        self._dirty = False

    # -- convenience ----------------------------------------------------
    @property
    def chunk(self):
        return self.project.active

    @property
    def active_camera(self) -> Optional[Camera]:
        return self.chunk.camera(self._active_camera_id)

    @property
    def active_gcp(self) -> Optional[GCP]:
        return self.chunk.gcp(self._active_gcp_id)

    def workdir(self) -> str:
        if self.project.path:
            base = os.path.splitext(self.project.path)[0] + "_data"
        else:
            base = os.path.join(tempfile.gettempdir(), "aerosurvey_untitled")
        os.makedirs(base, exist_ok=True)
        return base

    def set_dirty(self, val: bool = True) -> None:
        if val != self._dirty:
            self._dirty = val
            self.dirty_changed.emit(val)

    # -- project lifecycle ----------------------------------------------
    def new_project(self) -> None:
        self.project = Project()
        self._active_camera_id = -1
        self._active_gcp_id = -1
        self.set_dirty(False)
        self.project_changed.emit()
        self.cameras_changed.emit()
        self.gcps_changed.emit()
        self.log.emit("New project created.", "info")

    def open_project(self, path: str) -> None:
        self.project = Project.load(path)
        self._active_camera_id = -1
        self._active_gcp_id = -1
        self.set_dirty(False)
        self.project_changed.emit()
        self.cameras_changed.emit()
        self.gcps_changed.emit()
        self.log.emit(f"Opened project: {path}", "ok")

    def save_project(self, path: str) -> None:
        self.project.save(path)
        self.set_dirty(False)
        self.project_changed.emit()
        self.log.emit(f"Saved project: {path}", "ok")

    # -- coordinate system ----------------------------------------------
    def set_crs(self, mode: str, epsg: Optional[int],
                vertical_datum: str = "ellipsoidal", geoid_separation: float = 0.0) -> None:
        ch = self.chunk
        ch.crs_mode = mode
        ch.epsg = epsg
        ch.crs_label = crsmod.describe(epsg) if mode != "local" else "Local coordinates (arbitrary)"
        ch.vertical_datum = vertical_datum
        ch.geoid_separation = geoid_separation
        self._reproject_cameras()
        self.set_dirty()
        self.project_changed.emit()
        self.log.emit(f"Coordinate system: {ch.crs_label}", "info")
        if vertical_datum == "orthometric":
            self.log.emit(f"Vertical datum: orthometric (geoid N = {geoid_separation:.3f} m)", "info")

    def auto_utm_from_photos(self) -> Optional[int]:
        for c in self.chunk.cameras:
            if c.has_geotag:
                return crsmod.utm_epsg(c.lon, c.lat)
        return None

    def _reproject_cameras(self) -> None:
        ch = self.chunk
        tf = crsmod.CrsTransform(ch.epsg if ch.crs_mode != "local" else None)
        for c in ch.cameras:
            if c.has_geotag:
                c.est_x, c.est_y, c.est_z = tf.forward(c.lon, c.lat, c.alt or 0.0)

    # -- photos ---------------------------------------------------------
    def add_photos(self, paths: List[str]) -> int:
        ch = self.chunk
        added = 0
        for p in paths:
            meta = eximod.read_metadata(p)
            cam = ch.add_camera(p)
            cam.width = meta.get("width", 0)
            cam.height = meta.get("height", 0)
            cam.make = meta.get("make", "")
            cam.model = meta.get("model", "")
            cam.focal_mm = meta.get("focal_mm")
            cam.datetime = meta.get("datetime", "")
            cam.lat = meta.get("lat")
            cam.lon = meta.get("lon")
            cam.alt = meta.get("alt")
            cam.yaw = meta.get("yaw")
            cam.pitch = meta.get("pitch")
            cam.roll = meta.get("roll")
            added += 1
        self._reproject_cameras()
        if added:
            self.set_dirty()
            self.cameras_changed.emit()
            geotagged = sum(1 for c in ch.cameras if c.has_geotag)
            self.log.emit(f"Imported {added} photo(s); {geotagged} geotagged.", "ok")
        return added

    def remove_cameras(self, cam_ids: List[int]) -> None:
        ch = self.chunk
        idset = set(cam_ids)
        ch.cameras = [c for c in ch.cameras if c.id not in idset]
        for g in ch.gcps:
            for cid in list(g.observations):
                if cid in idset:
                    g.unmark(cid)
        self.set_dirty()
        self.cameras_changed.emit()
        self.observations_changed.emit()

    def set_active_camera(self, cam_id: int) -> None:
        self._active_camera_id = cam_id
        self.active_camera_changed.emit(cam_id)

    # -- gcps -----------------------------------------------------------
    def add_gcp(self, label: str, x=0.0, y=0.0, z=0.0, is_check=False) -> GCP:
        g = self.chunk.add_gcp(label, x, y, z, is_check)
        self.set_dirty()
        self.gcps_changed.emit()
        return g

    def remove_gcps(self, gcp_ids: List[int]) -> None:
        for gid in gcp_ids:
            self.chunk.remove_gcp(gid)
        if self._active_gcp_id in gcp_ids:
            self._active_gcp_id = -1
            self.active_gcp_changed.emit(-1)
        self.set_dirty()
        self.gcps_changed.emit()

    def import_gcps_csv(self, path: str, has_header: bool = True) -> int:
        """CSV columns: label, x, y, z[, type]. Delimiter auto-detected."""
        import csv
        added = 0
        with open(path, newline="", encoding="utf-8-sig") as fh:
            sample = fh.read(2048)
            fh.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t ")
            except csv.Error:
                dialect = csv.excel
            reader = csv.reader(fh, dialect)
            rows = list(reader)
        start = 1 if has_header and rows else 0
        for row in rows[start:]:
            row = [c for c in row if c.strip() != ""]
            if len(row) < 4:
                continue
            try:
                label = row[0].strip()
                x, y, z = float(row[1]), float(row[2]), float(row[3])
            except ValueError:
                continue
            is_check = len(row) >= 5 and row[4].strip().lower().startswith(("check", "c"))
            self.chunk.add_gcp(label, x, y, z, is_check)
            added += 1
        if added:
            self.set_dirty()
            self.gcps_changed.emit()
            self.log.emit(f"Imported {added} GCP(s) from {os.path.basename(path)}.", "ok")
        return added

    def set_active_gcp(self, gcp_id: int) -> None:
        self._active_gcp_id = gcp_id
        self.active_gcp_changed.emit(gcp_id)

    # -- observations (markers) -----------------------------------------
    def mark(self, gcp_id: int, cam_id: int, px: float, py: float) -> None:
        g = self.chunk.gcp(gcp_id)
        if g:
            g.mark(cam_id, px, py)
            self.set_dirty()
            self.observations_changed.emit()

    def unmark(self, gcp_id: int, cam_id: int) -> None:
        g = self.chunk.gcp(gcp_id)
        if g:
            g.unmark(cam_id)
            self.set_dirty()
            self.observations_changed.emit()
