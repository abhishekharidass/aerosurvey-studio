"""Pix4D-style map view: an OSM/satellite basemap with the project overlaid.

Slippy-map tiles are fetched asynchronously (QtNetwork) into a QGraphicsScene
whose coordinates are EPSG:3857 metres (y negated for Qt's y-down). Camera
geotags, GCPs and the orthomosaic are draped on top. Tiles are cached on disk
under ~/.aerosurvey/tilecache so repeat sessions work offline.
"""
from __future__ import annotations

import os

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (QBrush, QColor, QFont, QImage, QPainter, QPainterPath,
                           QPen, QPixmap, QTransform)
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkRequest
from PySide6.QtWidgets import (QComboBox, QGraphicsItem, QGraphicsPathItem,
                               QGraphicsPixmapItem, QGraphicsScene,
                               QGraphicsView, QHBoxLayout, QLabel, QMessageBox,
                               QPushButton, QSlider, QVBoxLayout, QWidget)

from ...config import APP_NAME, APP_VERSION
from ...core import webmercator as wm
from ...theme import ACCENT, BG_INPUT, BORDER, FG_MUTED, OK_GREEN, WARN_AMBER

PROVIDERS = {
    "OpenStreetMap": {
        "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attribution": "© OpenStreetMap contributors",
        "max_zoom": 19,
    },
    "Satellite (Esri)": {
        "url": ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Imagery/MapServer/tile/{z}/{y}/{x}"),
        "attribution": "Imagery © Esri, Maxar, Earthstar Geographics",
        "max_zoom": 19,
    },
    "OpenTopoMap": {
        "url": "https://a.tile.opentopomap.org/{z}/{x}/{y}.png",
        "attribution": "© OpenStreetMap contributors, SRTM · style © OpenTopoMap",
        "max_zoom": 17,
    },
    "No basemap": {"url": "", "attribution": "", "max_zoom": 19},
}

_USER_AGENT = f"{APP_NAME.replace(' ', '')}/{APP_VERSION} (+https://github.com/abhishekharidass/aerosurvey-studio)"
_TILE_CACHE = os.path.join(os.path.expanduser("~"), ".aerosurvey", "tilecache")


def _scene_pt(mx: float, my: float) -> QPointF:
    return QPointF(mx, -my)


class CameraDot(QGraphicsItem):
    """Zoom-invariant dot marking a photo position."""

    def __init__(self, label: str, aligned: bool):
        super().__init__()
        self.color = QColor(OK_GREEN if aligned else ACCENT)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setToolTip(label)
        self.setZValue(20)

    def boundingRect(self) -> QRectF:
        return QRectF(-5, -5, 10, 10)

    def paint(self, p: QPainter, *_):
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(QPen(QColor(0, 0, 0, 180), 1))
        p.setBrush(QBrush(self.color))
        p.drawEllipse(QPointF(0, 0), 3.4, 3.4)


class GcpFlag(QGraphicsItem):
    """Zoom-invariant triangle + label chip marking a GCP."""

    def __init__(self, label: str, is_check: bool):
        super().__init__()
        self.label = label
        self.color = QColor(WARN_AMBER if is_check else "#e8e13a")
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setZValue(21)

    def boundingRect(self) -> QRectF:
        return QRectF(-8, -22, 110, 32)

    def paint(self, p: QPainter, *_):
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(QPen(QColor(0, 0, 0, 200), 1.2))
        p.setBrush(QBrush(self.color))
        pts = [QPointF(0, 0), QPointF(-6, -10), QPointF(6, -10)]
        p.drawPolygon(pts)
        p.setFont(QFont("Segoe UI", 8, QFont.DemiBold))
        tw = p.fontMetrics().horizontalAdvance(self.label) + 8
        chip = QRectF(8, -20, tw, 15)
        p.setBrush(QColor(0, 0, 0, 150))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(chip, 3, 3)
        p.setPen(QPen(self.color))
        p.drawText(chip.adjusted(4, 0, 0, 0), Qt.AlignVCenter | Qt.AlignLeft, self.label)


class MapCanvas(QGraphicsView):
    """Pan/zoom canvas over the mercator scene; owns the tile layer."""

    measure_finished = Signal(list)   # [(mx, my), ...] mercator metres

    def __init__(self, log):
        super().__init__()
        self.log = log
        # timer first: setSceneRect below re-enters via scrollContentsBy
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(120)
        self._refresh_timer.timeout.connect(self._refresh_tiles)

        self.setScene(QGraphicsScene(self))
        self.setSceneRect(-wm.ORIGIN, -wm.ORIGIN, 2 * wm.ORIGIN, 2 * wm.ORIGIN)
        self.setBackgroundBrush(QColor(BG_INPUT))
        self.setRenderHints(QPainter.SmoothPixmapTransform | QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.provider_key = "OpenStreetMap"
        self._net = QNetworkAccessManager(self)
        self._tiles = {}          # (provider, z, x, y) -> QGraphicsPixmapItem
        self._pending = set()
        self._net_warned = False

        self.measure_mode = False
        self._measure_pts = []    # scene coords
        self._measure_item = None

        # world view until the project provides a location
        self.scale(3e-5, 3e-5)

    # -- volume measurement (polygon sketching) ---------------------------
    def set_measure_mode(self, on: bool):
        self.measure_mode = on
        self._clear_measure()
        self.setDragMode(QGraphicsView.NoDrag if on
                         else QGraphicsView.ScrollHandDrag)
        self.viewport().setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)

    def _clear_measure(self):
        self._measure_pts = []
        if self._measure_item is not None:
            self.scene().removeItem(self._measure_item)
            self._measure_item = None

    def _redraw_measure(self):
        if self._measure_item is None:
            self._measure_item = QGraphicsPathItem()
            pen = QPen(QColor(ACCENT), 0)          # cosmetic width-0 pen
            pen.setStyle(Qt.DashLine)
            self._measure_item.setPen(pen)
            self._measure_item.setBrush(QColor(61, 142, 201, 60))
            self._measure_item.setZValue(30)
            self.scene().addItem(self._measure_item)
        path = QPainterPath()
        if self._measure_pts:
            path.moveTo(self._measure_pts[0])
            for p in self._measure_pts[1:]:
                path.lineTo(p)
            path.closeSubpath()
        self._measure_item.setPath(path)

    def mousePressEvent(self, event):
        if self.measure_mode and event.button() == Qt.LeftButton:
            self._measure_pts.append(self.mapToScene(event.position().toPoint()))
            self._redraw_measure()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.measure_mode:
            if len(self._measure_pts) >= 3:
                pts = [(p.x(), -p.y()) for p in self._measure_pts]  # scene->mercator
                self.measure_finished.emit(pts)
            else:
                self.log("A volume polygon needs at least 3 corners.", "warn")
            self._clear_measure()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        if self.measure_mode and event.key() == Qt.Key_Escape:
            self._clear_measure()
            event.accept()
            return
        super().keyPressEvent(event)

    # -- navigation -------------------------------------------------------
    def wheelEvent(self, event):
        factor = 1.3 if event.angleDelta().y() > 0 else 1 / 1.3
        m11 = self.transform().m11() * factor
        if 1e-6 < m11 < 40.0:  # whole world .. ~2.5 cm/px
            self.scale(factor, factor)
            self.schedule_refresh()

    def scrollContentsBy(self, dx, dy):
        super().scrollContentsBy(dx, dy)
        self.schedule_refresh()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.schedule_refresh()

    def zoom_to(self, rect: QRectF):
        if rect.isNull() or not rect.isValid():
            return
        pad = max(rect.width(), rect.height(), 50.0) * 0.15
        self.fitInView(rect.adjusted(-pad, -pad, pad, pad), Qt.KeepAspectRatio)
        self.schedule_refresh()

    # -- tile layer ---------------------------------------------------------
    def set_provider(self, key: str):
        self.provider_key = key
        for item in self._tiles.values():
            self.scene().removeItem(item)
        self._tiles.clear()
        self.schedule_refresh()

    def schedule_refresh(self):
        self._refresh_timer.start()

    def _visible_mercator(self):
        r = self.mapToScene(self.viewport().rect()).boundingRect()
        return r.left(), -r.bottom(), r.right(), -r.top()  # w, s, e, n

    def _refresh_tiles(self):
        prov = PROVIDERS[self.provider_key]
        if not prov["url"]:
            return
        mpp = 1.0 / max(self.transform().m11(), 1e-12)
        z = wm.zoom_for_resolution(mpp, 1, prov["max_zoom"])
        west, south, east, north = self._visible_mercator()
        needed = set(wm.tiles_in_bounds(west, south, east, north, z, cap=80))
        for tx, ty in needed:
            self._request_tile(z, tx, ty)
        self._prune(z, needed)

    def _prune(self, z: int, needed: set):
        loaded_all = all((self.provider_key, z, tx, ty) in self._tiles
                         for tx, ty in needed)
        for key in list(self._tiles):
            pkey, tz, tx, ty = key
            stale = (pkey != self.provider_key
                     or (tz == z and (tx, ty) not in needed and len(needed) > 4)
                     or (tz != z and loaded_all))
            if stale:
                self.scene().removeItem(self._tiles.pop(key))

    def _request_tile(self, z: int, tx: int, ty: int):
        key = (self.provider_key, z, tx, ty)
        if key in self._tiles or key in self._pending:
            return
        cache = os.path.join(_TILE_CACHE, self.provider_key.replace(" ", "_"),
                             str(z), f"{tx}_{ty}.png")
        if os.path.exists(cache):
            pm = QPixmap(cache)
            if not pm.isNull():
                self._place_tile(key, pm)
                return
        url = PROVIDERS[self.provider_key]["url"].format(z=z, x=tx, y=ty)
        req = QNetworkRequest(QUrl(url))
        req.setHeader(QNetworkRequest.UserAgentHeader, _USER_AGENT)
        reply = self._net.get(req)
        self._pending.add(key)
        reply.finished.connect(lambda k=key, r=reply, c=cache: self._on_tile(k, r, c))

    def _on_tile(self, key, reply, cache):
        self._pending.discard(key)
        data = bytes(reply.readAll())
        reply.deleteLater()
        if reply.error() != reply.NetworkError.NoError or not data:
            if not self._net_warned:
                self._net_warned = True
                self.log(f"Basemap tiles unavailable ({reply.errorString()}) — "
                         "check the network; cached tiles still display.", "warn")
            return
        pm = QPixmap()
        if not pm.loadFromData(data):
            return
        try:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            pm.save(cache, "PNG")
        except OSError:
            pass
        self._place_tile(key, pm)

    def _place_tile(self, key, pm: QPixmap):
        pkey, z, tx, ty = key
        if pkey != self.provider_key or key in self._tiles:
            return
        west, south, east, north = wm.tile_bounds(tx, ty, z)
        item = QGraphicsPixmapItem(pm)
        item.setTransformationMode(Qt.SmoothTransformation)
        scale = (east - west) / pm.width()
        item.setTransform(QTransform(scale, 0, 0, scale, 0, 0))
        item.setPos(_scene_pt(west, north))
        item.setZValue(z - 100)  # tiles always under overlays; deeper zoom on top
        self.scene().addItem(item)
        self._tiles[key] = item


class MapView(QWidget):
    """Toolbar + canvas + attribution; syncs overlays with the app state."""

    def __init__(self, state):
        super().__init__()
        self.state = state
        self._overlay_items = []
        self._ortho_item = None
        self._ortho_sig = None
        self._zoomed_once = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        bar = QHBoxLayout()
        bar.setContentsMargins(6, 4, 6, 4)
        bar.addWidget(QLabel("Basemap:"))
        self.provider = QComboBox()
        self.provider.addItems(list(PROVIDERS))
        self.provider.currentTextChanged.connect(self._on_provider)
        bar.addWidget(self.provider)
        bar.addSpacing(12)
        bar.addWidget(QLabel("Ortho opacity:"))
        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(90)
        self.opacity.setFixedWidth(120)
        self.opacity.valueChanged.connect(self._on_opacity)
        bar.addWidget(self.opacity)
        bar.addStretch(1)
        self.measure_btn = QPushButton("Measure volume")
        self.measure_btn.setCheckable(True)
        self.measure_btn.setToolTip("Click the stockpile corners on the map; "
                                    "double-click to close the polygon. Esc restarts.")
        self.measure_btn.toggled.connect(self._on_measure_toggled)
        bar.addWidget(self.measure_btn)
        fit = QPushButton("Zoom to project")
        fit.clicked.connect(self.zoom_to_project)
        bar.addWidget(fit)
        lay.addLayout(bar)

        self.canvas = MapCanvas(state.log.emit)
        self.canvas.measure_finished.connect(self._on_measure_finished)
        lay.addWidget(self.canvas, 1)

        self.attribution = QLabel(PROVIDERS[self.canvas.provider_key]["attribution"])
        self.attribution.setStyleSheet(
            f"color: {FG_MUTED}; padding: 2px 8px; border-top: 1px solid {BORDER};")
        lay.addWidget(self.attribution)

        state.cameras_changed.connect(self.rebuild)
        state.gcps_changed.connect(self.rebuild)
        state.outputs_changed.connect(self.rebuild)
        state.project_changed.connect(self.rebuild)
        self.rebuild()

    # -- toolbar ----------------------------------------------------------
    def _on_provider(self, key: str):
        self.canvas.set_provider(key)
        self.attribution.setText(PROVIDERS[key]["attribution"])

    def _on_opacity(self, val: int):
        if self._ortho_item:
            self._ortho_item.setOpacity(val / 100.0)

    # -- volume measurement -------------------------------------------------
    def _on_measure_toggled(self, on: bool):
        if on:
            o = self.state.chunk.outputs
            if not (o.dsm and os.path.exists(o.dsm)):
                QMessageBox.information(self, "Measure volume",
                                        "No DSM yet — run 'Build DSM' first.")
                self.measure_btn.setChecked(False)
                return
            if self.state.chunk.crs_mode == "local" or not self.state.chunk.epsg:
                QMessageBox.information(self, "Measure volume",
                                        "Volumes need a georeferenced project "
                                        "(set a CRS and re-run georeferencing).")
                self.measure_btn.setChecked(False)
                return
            self.state.log.emit("Volume tool: click the outline on the map, "
                                "double-click to finish.", "info")
        self.canvas.set_measure_mode(self.measure_btn.isChecked())

    def _on_measure_finished(self, merc_pts):
        from ...core import volumes as volmod
        from ..dialogs import VolumeBaseDialog
        self.measure_btn.setChecked(False)
        try:
            from pyproj import Transformer
            tf = Transformer.from_crs("EPSG:3857",
                                      f"EPSG:{self.state.chunk.epsg}",
                                      always_xy=True)
            poly = [tf.transform(mx, my) for mx, my in merc_pts]
        except Exception as exc:
            QMessageBox.critical(self, "Measure volume",
                                 f"Coordinate transform failed:\n{exc}")
            return
        dlg = VolumeBaseDialog(self)
        if not dlg.exec():
            return
        mode, custom_z = dlg.result_options()
        try:
            r = volmod.measure_volume(self.state.chunk.outputs.dsm, poly,
                                      mode, custom_z)
        except Exception as exc:
            QMessageBox.critical(self, "Measure volume", str(exc))
            return
        self.state.log.emit(f"Volume: {r.summary()}", "ok")
        for w in r.warnings:
            self.state.log.emit(f"Volume: {w}", "warn")
        lines = [f"Cut (above base):  {r.cut_m3:,.1f} m³",
                 f"Fill (below base):  {r.fill_m3:,.1f} m³",
                 f"Net:  {r.net_m3:+,.1f} m³", "",
                 f"Polygon area: {r.area_m2:,.1f} m² "
                 f"({r.coverage * 100:.0f}% has DSM data)",
                 f"Base [{r.base_mode}]: {r.base_z_min:.2f}"
                 + ("" if r.base_z_min == r.base_z_max
                    else f" – {r.base_z_max:.2f}") + " m",
                 f"Cell size: {r.cell_m:.2f} m ({r.n_cells:,} cells)"]
        if r.warnings:
            lines += [""] + ["⚠ " + w for w in r.warnings]
        QMessageBox.information(self, "Volume measurement", "\n".join(lines))

    # -- coordinate helpers -------------------------------------------------
    def _project_to_mercator(self):
        """Transformer from the chunk CRS to EPSG:3857, or None for local CRS."""
        ch = self.state.chunk
        if ch.crs_mode == "local" or not ch.epsg:
            return None
        try:
            from pyproj import Transformer
            return Transformer.from_crs(f"EPSG:{ch.epsg}", "EPSG:3857",
                                        always_xy=True)
        except Exception:
            return None

    # -- overlays ------------------------------------------------------------
    def rebuild(self):
        scene = self.canvas.scene()
        for item in self._overlay_items:
            scene.removeItem(item)
        self._overlay_items = []

        bounds = QRectF()
        for cam in self.state.chunk.cameras:
            if not cam.has_geotag:
                continue
            mx, my = wm.lonlat_to_mercator(cam.lon, cam.lat)
            dot = CameraDot(cam.filename, cam.aligned)
            dot.setPos(_scene_pt(mx, my))
            scene.addItem(dot)
            self._overlay_items.append(dot)
            bounds = bounds.united(QRectF(mx - 1, -my - 1, 2, 2))

        tf = self._project_to_mercator()
        if tf:
            for g in self.state.chunk.gcps:
                if g.x == 0 and g.y == 0:
                    continue
                try:
                    mx, my = tf.transform(g.x, g.y)
                except Exception:
                    continue
                flag = GcpFlag(g.label, g.is_check)
                flag.setPos(_scene_pt(mx, my))
                scene.addItem(flag)
                self._overlay_items.append(flag)
                bounds = bounds.united(QRectF(mx - 1, -my - 1, 2, 2))

        rect = self._rebuild_ortho()
        if rect is not None:
            bounds = bounds.united(rect)

        if not self._zoomed_once and not bounds.isNull():
            self._zoomed_once = True
            self.canvas.zoom_to(bounds)

    def _rebuild_ortho(self):
        """Drape a decimated, mercator-warped ortho preview; returns its rect."""
        path = self.state.chunk.outputs.orthomosaic
        scene = self.canvas.scene()
        sig = (path, os.path.getmtime(path)) if path and os.path.exists(path) else None
        if sig == self._ortho_sig:
            if self._ortho_item is not None:
                r = self._ortho_item.sceneBoundingRect()
                return QRectF(r.left(), r.top(), r.width(), r.height())
            return None
        if self._ortho_item is not None:
            scene.removeItem(self._ortho_item)
            self._ortho_item = None
        self._ortho_sig = sig
        if sig is None:
            return None
        try:
            img, west, north, px_x, px_y = _warp_ortho_preview(path)
        except Exception as exc:
            self.state.log.emit(f"Map: could not drape orthomosaic: {exc}", "warn")
            return None
        item = QGraphicsPixmapItem(QPixmap.fromImage(img))
        item.setTransformationMode(Qt.SmoothTransformation)
        item.setTransform(QTransform(px_x, 0, 0, px_y, 0, 0))
        item.setPos(_scene_pt(west, north))
        item.setZValue(5)
        item.setOpacity(self.opacity.value() / 100.0)
        scene.addItem(item)
        self._ortho_item = item
        return item.sceneBoundingRect()

    def zoom_to_project(self):
        bounds = QRectF()
        for item in self._overlay_items:
            p = item.pos()
            bounds = bounds.united(QRectF(p.x() - 1, p.y() - 1, 2, 2))
        if self._ortho_item is not None:
            bounds = bounds.united(self._ortho_item.sceneBoundingRect())
        if bounds.isNull():
            self.state.log.emit(
                "Nothing to zoom to yet — import geotagged photos or GCPs.", "info")
            return
        self.canvas.zoom_to(bounds)


def _warp_ortho_preview(path: str, max_dim: int = 1400):
    """Decimate + reproject the ortho to EPSG:3857; return QImage and placement."""
    import numpy as np
    import rasterio
    from rasterio.transform import Affine
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError("orthomosaic has no CRS")
        scale = max(max(src.width, src.height) / float(max_dim), 1.0)
        dw, dh = max(int(src.width / scale), 1), max(int(src.height / scale), 1)
        data = src.read(out_shape=(src.count, dh, dw))
        src_tr = src.transform * Affine.scale(src.width / dw, src.height / dh)
        dst_tr, w, h = calculate_default_transform(
            src.crs, "EPSG:3857", dw, dh, *src.bounds)
        bands = min(src.count, 4)
        dst = np.zeros((bands, h, w), np.uint8)
        for b in range(bands):
            reproject(data[b], dst[b], src_transform=src_tr, src_crs=src.crs,
                      dst_transform=dst_tr, dst_crs="EPSG:3857",
                      resampling=Resampling.bilinear)
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., :3] = np.transpose(dst[:3] if bands >= 3 else
                                 np.repeat(dst[:1], 3, axis=0), (1, 2, 0))
    rgba[..., 3] = dst[3] if bands == 4 else 255
    img = QImage(np.ascontiguousarray(rgba).tobytes(), w, h, 4 * w,
                 QImage.Format_RGBA8888).copy()
    return img, dst_tr.c, dst_tr.f, dst_tr.a, -dst_tr.e
