"""3D products view: a quick top-down render of the point cloud plus a button
to open it in an interactive Open3D window. Also previews DSM/DTM/ortho rasters.
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QPushButton,
                               QVBoxLayout, QWidget)

from ...config import CLASS_COLORS, CLASS_NAMES
from ...theme import BG_INPUT, FG_MUTED


def _np_to_qimage(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = rgb.shape
    return QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format_RGB888).copy()


def render_topdown(P: np.ndarray, C: np.ndarray, size: int = 720) -> QImage:
    img = np.empty((size, size, 3), np.uint8)
    img[:] = (30, 33, 36)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    rx, ry = x.max() - x.min(), y.max() - y.min()
    span = max(rx, ry, 1e-6)
    sx = (x - x.min()) / span
    sy = (y - y.min()) / span
    order = np.argsort(z)  # draw high points last (on top)
    px = (sx[order] * (size - 1) * (rx / span)).astype(int)
    py = ((1 - sy[order]) * (size - 1)).astype(int)
    px = np.clip(px, 0, size - 1)
    py = np.clip(py, 0, size - 1)
    img[py, px] = C[order]
    return _np_to_qimage(img)


def render_raster(path: str, size: int = 720) -> QImage:
    import rasterio
    with rasterio.open(path) as src:
        data = src.read()
    if data.shape[0] >= 3:  # RGB ortho
        rgb = np.transpose(data[:3], (1, 2, 0))
        rgb = np.nan_to_num(rgb).clip(0, 255).astype(np.uint8)
    else:  # single-band elevation -> colourise
        band = data[0].astype(np.float32)
        finite = np.isfinite(band)
        lo, hi = (np.percentile(band[finite], [2, 98]) if finite.any() else (0, 1))
        norm = np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1)
        norm = np.nan_to_num(norm)
        rgb = np.zeros((*band.shape, 3), np.uint8)
        rgb[..., 0] = (norm * 255)
        rgb[..., 1] = (np.abs(np.sin(norm * np.pi)) * 200 + 40)
        rgb[..., 2] = ((1 - norm) * 255)
    return _np_to_qimage(np.ascontiguousarray(rgb))


class ModelView(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        bar = QHBoxLayout()
        bar.setContentsMargins(6, 4, 6, 4)
        bar.addWidget(QLabel("View:"))
        self.source = QComboBox()
        self.source.currentIndexChanged.connect(self.refresh)
        bar.addWidget(self.source)
        bar.addWidget(QLabel("Colour:"))
        self.color_mode = QComboBox()
        self.color_mode.addItems(["RGB", "Classification"])
        self.color_mode.currentIndexChanged.connect(self.refresh)
        bar.addWidget(self.color_mode)
        bar.addStretch(1)
        self.open3d_btn = QPushButton("Open in 3D viewer ↗")
        self.open3d_btn.clicked.connect(self._open_3d)
        bar.addWidget(self.open3d_btn)
        lay.addLayout(bar)

        self.canvas = QLabel("No 3D products yet.\n\nRun Workflow ▸ Build Dense Cloud.")
        self.canvas.setAlignment(Qt.AlignCenter)
        self.canvas.setStyleSheet(f"background: {BG_INPUT}; color: {FG_MUTED};")
        self.canvas.setMinimumHeight(300)
        lay.addWidget(self.canvas, 1)

        self.legend = QLabel("")
        self.legend.setStyleSheet(f"color: {FG_MUTED}; padding: 4px 8px;")
        self.legend.setWordWrap(True)
        lay.addWidget(self.legend)

        self._pixmap = None
        state.outputs_changed.connect(self._refresh_sources)
        state.project_changed.connect(self._refresh_sources)
        self._refresh_sources()

    def _available_sources(self):
        o = self.state.chunk.outputs
        items = []
        if o.classified_cloud:
            items.append(("Classified cloud", o.classified_cloud, "cloud"))
        if o.dense_cloud:
            items.append(("Dense cloud", o.dense_cloud, "cloud"))
        if o.sparse_cloud:
            items.append(("Sparse cloud (SfM)", o.sparse_cloud, "cloud"))
        if o.dsm:
            items.append(("DSM", o.dsm, "raster"))
        if o.dtm:
            items.append(("DTM", o.dtm, "raster"))
        if o.orthomosaic:
            items.append(("Orthomosaic", o.orthomosaic, "raster"))
        return items

    def _refresh_sources(self):
        cur = self.source.currentData()
        self.source.blockSignals(True)
        self.source.clear()
        for name, path, kind in self._available_sources():
            self.source.addItem(name, (path, kind))
        idx = 0
        if cur:
            found = next((i for i in range(self.source.count())
                          if self.source.itemData(i) == cur), -1)
            idx = found if found >= 0 else 0
        self.source.setCurrentIndex(idx)
        self.source.blockSignals(False)
        self.refresh()

    def refresh(self):
        data = self.source.currentData()
        self.color_mode.setEnabled(bool(data) and data[1] == "cloud")
        self.open3d_btn.setEnabled(bool(data) and data[1] == "cloud")
        if not data:
            self.canvas.setText("No 3D products yet.\n\nRun Workflow ▸ Build Dense Cloud.")
            self.canvas.setPixmap(QPixmap())
            self.legend.setText("")
            return
        path, kind = data
        try:
            if kind == "cloud":
                img = self._render_cloud(path)
            else:
                img = render_raster(path)
        except Exception as exc:
            self.canvas.setText(f"Could not render:\n{exc}")
            return
        self._pixmap = QPixmap.fromImage(img)
        self._apply_pixmap()

    def _render_cloud(self, path):
        import laspy
        las = laspy.read(path)
        P = np.column_stack([las.x, las.y, las.z])
        cls = np.array(las.classification)
        if self.color_mode.currentIndex() == 1:
            C = np.array([CLASS_COLORS.get(int(c), (200, 200, 200)) for c in cls], np.uint8)
            present = sorted(set(int(c) for c in cls))
            self.legend.setText("  ".join(
                f"■ {CLASS_NAMES.get(c, c)}" for c in present))
        else:
            C = (np.column_stack([las.red, las.green, las.blue]) // 257).astype(np.uint8)
            self.legend.setText("Top-down view · true colour · higher points drawn on top")
        return render_topdown(P, C)

    def _apply_pixmap(self):
        if self._pixmap:
            self.canvas.setPixmap(self._pixmap.scaled(
                self.canvas.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_pixmap()

    def _open_3d(self):
        data = self.source.currentData()
        if not data:
            return
        path, _ = data
        mode = "class" if self.color_mode.currentIndex() == 1 else "rgb"
        try:
            subprocess.Popen([sys.executable, "-m", "aerosurvey.viewer3d", path, mode],
                             cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            self.state.log.emit("Launched interactive Open3D viewer.", "info")
        except Exception as exc:
            self.state.log.emit(f"Could not launch 3D viewer: {exc}", "error")
