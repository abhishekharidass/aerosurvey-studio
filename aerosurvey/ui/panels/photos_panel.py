"""Photos panel — list of source images with sorting, including by proximity
to a selected GCP (the 'sort images according to points' requirement)."""
from __future__ import annotations

import math

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QVBoxLayout, QWidget)

from ...theme import FG_MUTED, OK_GREEN


class PhotosPanel(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        row = QHBoxLayout()
        row.addWidget(QLabel("Sort:"))
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Name", "Capture time", "Nearest to GCP…"])
        self.sort_combo.currentIndexChanged.connect(self.refresh)
        row.addWidget(self.sort_combo, 1)
        self.gcp_combo = QComboBox()
        self.gcp_combo.setVisible(False)
        self.gcp_combo.currentIndexChanged.connect(self.refresh)
        row.addWidget(self.gcp_combo, 1)
        lay.addLayout(row)

        self.list = QListWidget()
        self.list.currentItemChanged.connect(self._on_select)
        lay.addWidget(self.list, 1)

        self.count_label = QLabel("0 photos")
        self.count_label.setStyleSheet(f"color: {FG_MUTED};")
        lay.addWidget(self.count_label)

        state.cameras_changed.connect(self.refresh)
        state.gcps_changed.connect(self._refresh_gcps)
        state.observations_changed.connect(self.refresh)
        state.active_gcp_changed.connect(self.refresh)
        self._refresh_gcps()

    def _refresh_gcps(self) -> None:
        cur = self.gcp_combo.currentData()
        self.gcp_combo.blockSignals(True)
        self.gcp_combo.clear()
        for g in self.state.chunk.gcps:
            self.gcp_combo.addItem(g.label, g.id)
        if cur is not None:
            idx = self.gcp_combo.findData(cur)
            if idx >= 0:
                self.gcp_combo.setCurrentIndex(idx)
        self.gcp_combo.blockSignals(False)
        self.refresh()

    def _sorted_cameras(self):
        cams = list(self.state.chunk.cameras)
        mode = self.sort_combo.currentIndex()
        self.gcp_combo.setVisible(mode == 2)
        if mode == 0:
            cams.sort(key=lambda c: c.filename.lower())
        elif mode == 1:
            cams.sort(key=lambda c: (c.datetime or "", c.filename.lower()))
        elif mode == 2:
            gid = self.gcp_combo.currentData()
            g = self.state.chunk.gcp(gid) if gid is not None else None
            if g is not None:
                def dist(c):
                    if c.est_x is None:
                        return math.inf
                    return math.hypot(c.est_x - g.x, c.est_y - g.y)
                cams.sort(key=dist)
        return cams

    def refresh(self) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        active_gcp = self.state.active_gcp
        for c in self._sorted_cameras():
            tags = []
            if c.has_geotag:
                tags.append("\U0001F4CD")
            if active_gcp and active_gcp.observation(c.id):
                tags.append("◉")  # marked for current GCP
            label = f"{c.filename}  {' '.join(tags)}".rstrip()
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, c.id)
            if active_gcp and active_gcp.observation(c.id):
                item.setForeground(Qt.green)
            self.list.addItem(item)
        self.list.blockSignals(False)
        n = len(self.state.chunk.cameras)
        geo = sum(1 for c in self.state.chunk.cameras if c.has_geotag)
        self.count_label.setText(f"{n} photos · {geo} geotagged")

    def _on_select(self, cur, _prev) -> None:
        if cur is None:
            return
        cam_id = cur.data(Qt.UserRole)
        if cam_id is not None:
            self.state.set_active_camera(cam_id)

    def select_camera(self, cam_id: int) -> None:
        for i in range(self.list.count()):
            if self.list.item(i).data(Qt.UserRole) == cam_id:
                self.list.setCurrentRow(i)
                return
