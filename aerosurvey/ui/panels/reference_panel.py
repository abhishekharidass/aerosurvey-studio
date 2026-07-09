"""Reference panel — GCP table (editable) and camera geotag table.

Mirrors Metashape's Reference pane. Editing a cell writes back to the model;
selecting a GCP row makes it the active point for image marking.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QHeaderView,
                               QInputDialog, QMessageBox, QPushButton,
                               QTableWidget, QTableWidgetItem, QTabWidget,
                               QVBoxLayout, QWidget)

from ...theme import FG_MUTED

GCP_COLS = ["Label", "X / East", "Y / North", "Z", "Type", "Acc (m)", "Images", "Error (m)"]
CAM_COLS = ["Image", "Lat", "Lon", "Alt", "Yaw", "Est X", "Est Y", "Est Z"]


class ReferencePanel(QWidget):
    request_mark_help = Signal()

    def __init__(self, state):
        super().__init__()
        self.state = state
        self._updating = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        tabs = QTabWidget()
        outer.addWidget(tabs)

        # -- GCP tab --
        gcp_tab = QWidget()
        gl = QVBoxLayout(gcp_tab)
        gl.setContentsMargins(0, 0, 0, 0)
        self.gcp_table = QTableWidget(0, len(GCP_COLS))
        self.gcp_table.setHorizontalHeaderLabels(GCP_COLS)
        self.gcp_table.verticalHeader().setVisible(False)
        self.gcp_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.gcp_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.gcp_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.gcp_table.itemChanged.connect(self._on_gcp_edited)
        self.gcp_table.itemSelectionChanged.connect(self._on_gcp_selected)
        gl.addWidget(self.gcp_table)

        btns = QHBoxLayout()
        for text, slot, tip in [
                ("Add GCP", self._add_gcp, ""),
                ("Import CSV…", self._import_csv, ""),
                ("Auto-Mark", self._auto_mark,
                 "Project GCP coordinates into the photos to place predicted "
                 "marks (uses the solved alignment, or EXIF geotags before "
                 "alignment). Existing marks are kept."),
                ("Remove", self._remove_gcp, "")]:
            b = QPushButton(text)
            b.clicked.connect(slot)
            if tip:
                b.setToolTip(tip)
            btns.addWidget(b)
        btns.addStretch(1)
        gl.addLayout(btns)
        tabs.addTab(gcp_tab, "Ground Control")

        # -- Camera tab --
        cam_tab = QWidget()
        cl = QVBoxLayout(cam_tab)
        cl.setContentsMargins(0, 0, 0, 0)
        self.cam_table = QTableWidget(0, len(CAM_COLS))
        self.cam_table.setHorizontalHeaderLabels(CAM_COLS)
        self.cam_table.verticalHeader().setVisible(False)
        self.cam_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.cam_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        cl.addWidget(self.cam_table)
        tabs.addTab(cam_tab, "Cameras")

        state.gcps_changed.connect(self.refresh_gcps)
        state.observations_changed.connect(self.refresh_gcps)
        state.cameras_changed.connect(self.refresh_cameras)
        state.active_gcp_changed.connect(self._sync_selection)
        self.refresh_gcps()
        self.refresh_cameras()

    # -- GCP table -------------------------------------------------------
    def refresh_gcps(self) -> None:
        self._updating = True
        t = self.gcp_table
        t.setRowCount(0)
        for g in self.state.chunk.gcps:
            r = t.rowCount()
            t.insertRow(r)
            err = "" if g.error is None else f"{g.error:.3f}"
            vals = [g.label, f"{g.x:.3f}", f"{g.y:.3f}", f"{g.z:.3f}",
                    g.kind, f"{g.accuracy:.3f}", str(g.marked_count), err]
            for c, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setData(Qt.UserRole, g.id)
                if c == 4:  # type toggles via double-click, not free text
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    item.setForeground(Qt.cyan if g.is_check else Qt.white)
                if c == 6:  # images count read-only
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    item.setForeground(Qt.gray)
                if c == 7:  # georef residual read-only, colour-graded
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    if g.error is not None:
                        item.setForeground(Qt.green if g.error < 0.05
                                           else (Qt.yellow if g.error < 0.2 else Qt.red))
                if c in (1, 2, 3, 5, 7):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                t.setItem(r, c, item)
        self._updating = False

    def _on_gcp_edited(self, item) -> None:
        if self._updating:
            return
        gid = item.data(Qt.UserRole)
        g = self.state.chunk.gcp(gid)
        if not g:
            return
        col = item.column()
        text = item.text().strip()
        try:
            if col == 0:
                g.label = text
            elif col == 1:
                g.x = float(text)
            elif col == 2:
                g.y = float(text)
            elif col == 3:
                g.z = float(text)
            elif col == 5:
                g.accuracy = float(text)
        except ValueError:
            self.refresh_gcps()
            return
        self.state.set_dirty()
        self.state.gcps_changed.emit()

    def _on_gcp_selected(self) -> None:
        if self._updating:
            return
        items = self.gcp_table.selectedItems()
        if items:
            gid = items[0].data(Qt.UserRole)
            self.state.set_active_gcp(gid)

    def _sync_selection(self, gcp_id: int) -> None:
        for r in range(self.gcp_table.rowCount()):
            it = self.gcp_table.item(r, 0)
            if it and it.data(Qt.UserRole) == gcp_id:
                self.gcp_table.blockSignals(True)
                self.gcp_table.selectRow(r)
                self.gcp_table.blockSignals(False)
                return

    def _add_gcp(self) -> None:
        label, ok = QInputDialog.getText(self, "Add GCP", "Point label:",
                                         text=f"GCP{len(self.state.chunk.gcps) + 1}")
        if ok and label.strip():
            g = self.state.add_gcp(label.strip())
            self.state.set_active_gcp(g.id)

    def _remove_gcp(self) -> None:
        items = self.gcp_table.selectedItems()
        if not items:
            return
        gid = items[0].data(Qt.UserRole)
        g = self.state.chunk.gcp(gid)
        if g and QMessageBox.question(self, "Remove GCP",
                                      f"Remove '{g.label}' and its {g.marked_count} marks?") \
                == QMessageBox.Yes:
            self.state.remove_gcps([gid])

    def _auto_mark(self) -> None:
        items = self.gcp_table.selectedItems()
        ids = [items[0].data(Qt.UserRole)] if items else None
        self.state.auto_mark_gcps(ids)

    def _import_csv(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(self, "Import GCPs", "",
                                              "CSV / text (*.csv *.txt *.tsv);;All files (*.*)")
        if path:
            n = self.state.import_gcps_csv(path)
            if n == 0:
                QMessageBox.warning(self, "Import GCPs",
                                    "No rows parsed. Expected: label, X, Y, Z[, type].")

    # -- Camera table ----------------------------------------------------
    def refresh_cameras(self) -> None:
        t = self.cam_table
        t.setRowCount(0)

        def fmt(v, p=6):
            return "" if v is None else f"{v:.{p}f}"

        for c in self.state.chunk.cameras:
            r = t.rowCount()
            t.insertRow(r)
            vals = [c.filename, fmt(c.lat), fmt(c.lon), fmt(c.alt, 2),
                    fmt(c.yaw, 1), fmt(c.est_x, 2), fmt(c.est_y, 2), fmt(c.est_z, 2)]
            for col, v in enumerate(vals):
                item = QTableWidgetItem(v)
                if col > 0:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                t.setItem(r, col, item)
