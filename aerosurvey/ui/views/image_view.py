"""Interactive image canvas for marking Ground Control Points.

Capabilities (the core GCP-marking requirement):
  * high-res zoom (wheel, to cursor) and pan (left-drag on empty / middle-drag)
  * click to place the active GCP's marker at sub-pixel precision
  * drag an existing marker to reposition it
  * right-click a marker (or select + Delete) to remove it
  * every GCP observed on the current image is shown; the active one is highlighted
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (QBrush, QColor, QFont, QPainter, QPen, QPixmap)
from PySide6.QtWidgets import (QGraphicsItem, QGraphicsPixmapItem, QGraphicsScene,
                               QGraphicsView, QHBoxLayout, QLabel, QMenu,
                               QToolButton, QVBoxLayout, QWidget)

from ...theme import ACCENT, BG_INPUT, FG_MUTED, OK_GREEN

_PAN_THRESHOLD = 6  # px of movement before a left-drag becomes a pan


class MarkerItem(QGraphicsItem):
    """A draggable, zoom-invariant crosshair marking one GCP on one image."""

    def __init__(self, gcp, cam_id: int, active: bool, canvas):
        super().__init__()
        self.gcp = gcp
        self.cam_id = cam_id
        self.active = active
        self.canvas = canvas
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setZValue(10)
        self.setCursor(Qt.SizeAllCursor)
        self._suppress = False

    def boundingRect(self) -> QRectF:
        return QRectF(-16, -28, 120, 44)

    def paint(self, p: QPainter, *_):
        p.setRenderHint(QPainter.Antialiasing, True)
        col = QColor(ACCENT) if self.active else QColor(OK_GREEN)
        pen = QPen(col, 1.6)
        p.setPen(pen)
        r = 9
        # crosshair with gap at centre
        p.drawLine(-r - 5, 0, -3, 0)
        p.drawLine(3, 0, r + 5, 0)
        p.drawLine(0, -r - 5, 0, -3)
        p.drawLine(0, 3, 0, r + 5)
        p.drawEllipse(QPointF(0, 0), r, r)
        p.setBrush(QBrush(col))
        p.drawEllipse(QPointF(0, 0), 1.6, 1.6)
        # label chip
        p.setFont(QFont("Segoe UI", 8, QFont.DemiBold))
        text = self.gcp.label
        tw = p.fontMetrics().horizontalAdvance(text) + 8
        chip = QRectF(r + 4, -20, tw, 15)
        p.setBrush(QColor(0, 0, 0, 150))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(chip, 3, 3)
        p.setPen(QPen(col))
        p.drawText(chip.adjusted(4, 0, 0, 0), Qt.AlignVCenter | Qt.AlignLeft, text)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and not self._suppress:
            # clamp within image bounds (child coords == image pixels)
            pm = self.canvas.pixmap_size
            x = min(max(value.x(), 0.0), pm.width())
            y = min(max(value.y(), 0.0), pm.height())
            value = QPointF(x, y)
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.canvas.commit_marker(self)

    def contextMenuEvent(self, event):
        menu = QMenu()
        act = menu.addAction(f"Remove mark of '{self.gcp.label}'")
        if menu.exec(event.screenPos()) == act:
            self.canvas.remove_marker(self)


class ImageCanvas(QGraphicsView):
    marker_info = Signal(str)   # readout for the selected marker

    def __init__(self, state):
        super().__init__()
        self.state = state
        self.scene_obj = QGraphicsScene(self)
        self.setScene(self.scene_obj)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setBackgroundBrush(QColor(BG_INPUT))
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.scene_obj.selectionChanged.connect(self._on_selection)

        self.pix_item: QGraphicsPixmapItem | None = None
        self.pixmap_size = QPixmap().size()
        self._cam_id = -1
        self._panning = False
        self._maybe_pan = False
        self._press_view = None
        self._press_scene = None

        state.active_camera_changed.connect(self.load_camera)
        state.active_gcp_changed.connect(lambda _: self.rebuild_markers())
        state.observations_changed.connect(self.rebuild_markers)

    # -- image loading ---------------------------------------------------
    def load_camera(self, cam_id: int) -> None:
        self._cam_id = cam_id
        cam = self.state.chunk.camera(cam_id)
        self.scene_obj.clear()
        self.pix_item = None
        if not cam:
            return
        pm = QPixmap(cam.path)
        if pm.isNull():
            self.pixmap_size = pm.size()
            return
        self.pixmap_size = pm.size()
        self.pix_item = self.scene_obj.addPixmap(pm)
        self.pix_item.setTransformationMode(Qt.SmoothTransformation)
        self.scene_obj.setSceneRect(QRectF(pm.rect()).adjusted(-200, -200, 200, 200))
        self.fit()
        self.rebuild_markers()

    def fit(self) -> None:
        if self.pix_item:
            self.resetTransform()
            self.fitInView(self.pix_item, Qt.KeepAspectRatio)

    def zoom(self, factor: float, center=None) -> None:
        if center is None:
            self.scale(factor, factor)
            return
        old = self.mapToScene(center)
        self.scale(factor, factor)
        new = self.mapToScene(center)
        delta = new - old
        self.translate(delta.x(), delta.y())

    # -- markers ---------------------------------------------------------
    def rebuild_markers(self) -> None:
        if self.pix_item is None:
            return
        for it in list(self.pix_item.childItems()):
            if isinstance(it, MarkerItem):
                self.scene_obj.removeItem(it)
        active = self.state.active_gcp
        for g in self.state.chunk.gcps:
            obs = g.observation(self._cam_id)
            if obs is None:
                continue
            m = MarkerItem(g, self._cam_id, active is not None and g.id == active.id, self)
            m.setParentItem(self.pix_item)
            m._suppress = True
            m.setPos(obs.px, obs.py)
            m._suppress = False

    def commit_marker(self, marker: MarkerItem) -> None:
        pos = marker.pos()
        self.state.mark(marker.gcp.id, marker.cam_id, pos.x(), pos.y())

    def remove_marker(self, marker: MarkerItem) -> None:
        self.state.unmark(marker.gcp.id, marker.cam_id)

    def _place_active_marker(self, scene_pos) -> None:
        gcp = self.state.active_gcp
        if gcp is None or self.pix_item is None:
            self.window().statusBar().showMessage(
                "Select a GCP in the Reference panel before marking.", 4000)
            return
        local = self.pix_item.mapFromScene(scene_pos)
        w, h = self.pixmap_size.width(), self.pixmap_size.height()
        if not (0 <= local.x() <= w and 0 <= local.y() <= h):
            return
        self.state.mark(gcp.id, self._cam_id, local.x(), local.y())

    # -- interaction -----------------------------------------------------
    def wheelEvent(self, event):
        if self.pix_item is None:
            return
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.zoom(factor, event.position().toPoint())

    def mousePressEvent(self, event):
        item = self.itemAt(event.pos())
        if event.button() == Qt.LeftButton and isinstance(item, MarkerItem):
            super().mousePressEvent(event)  # let the marker handle its own drag
            return
        if event.button() == Qt.MiddleButton or \
           (event.button() == Qt.LeftButton and self.state.active_gcp is None):
            self._panning = True
            self._press_view = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button() == Qt.LeftButton:
            self._maybe_pan = True
            self._press_view = event.pos()
            self._press_scene = self.mapToScene(event.pos())
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning or self._maybe_pan:
            if self._maybe_pan and (event.pos() - self._press_view).manhattanLength() > _PAN_THRESHOLD:
                self._panning = True
                self._maybe_pan = False
                self.setCursor(Qt.ClosedHandCursor)
            if self._panning:
                delta = event.pos() - self._press_view
                self._press_view = event.pos()
                h, v = self.horizontalScrollBar(), self.verticalScrollBar()
                h.setValue(h.value() - delta.x())
                v.setValue(v.value() - delta.y())
                return
        super().mouseMoveEvent(event)
        pos = self.mapToScene(event.pos())
        if self.pix_item:
            local = self.pix_item.mapFromScene(pos)
            self.window().statusBar().showMessage(
                f"pixel  x={local.x():.1f}  y={local.y():.1f}")

    def mouseReleaseEvent(self, event):
        if self._panning:
            self._panning = False
            self.unsetCursor()
            return
        if self._maybe_pan:
            self._maybe_pan = False
            self._place_active_marker(self._press_scene)
            return
        super().mouseReleaseEvent(event)

    def _selected_marker(self):
        for it in self.scene_obj.selectedItems():
            if isinstance(it, MarkerItem):
                return it
        return None

    def _on_selection(self):
        m = self._selected_marker()
        if m is not None:
            p = m.pos()
            self.marker_info.emit(f"{m.gcp.label}: x={p.x():.2f}  y={p.y():.2f}  "
                                  "(arrows nudge · Shift = 0.2 px)")
        else:
            self.marker_info.emit("")

    def nudge_marker(self, m: MarkerItem, dx: float, dy: float) -> None:
        w, h = self.pixmap_size.width(), self.pixmap_size.height()
        nx = min(max(m.pos().x() + dx, 0.0), float(w))
        ny = min(max(m.pos().y() + dy, 0.0), float(h))
        m._suppress = True
        m.setPos(nx, ny)
        m._suppress = False
        # update the model in place without a rebuild (keeps the marker selected)
        g = self.state.chunk.gcp(m.gcp.id)
        if g is not None:
            g.mark(m.cam_id, nx, ny)
            self.state.set_dirty()
        self.marker_info.emit(f"{m.gcp.label}: x={nx:.2f}  y={ny:.2f}  "
                              "(arrows nudge · Shift = 0.2 px)")

    def keyPressEvent(self, event):
        m = self._selected_marker()
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if m is not None:
                self.remove_marker(m)
                return
        elif event.key() in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down) and m is not None:
            step = 0.2 if (event.modifiers() & Qt.ShiftModifier) else 1.0
            dx = -step if event.key() == Qt.Key_Left else step if event.key() == Qt.Key_Right else 0.0
            dy = -step if event.key() == Qt.Key_Up else step if event.key() == Qt.Key_Down else 0.0
            self.nudge_marker(m, dx, dy)
            return
        elif event.key() == Qt.Key_F:
            self.fit()
        super().keyPressEvent(event)


class ImageView(QWidget):
    """Toolbar + canvas wrapper shown in the central tab area."""

    def __init__(self, state):
        super().__init__()
        self.state = state
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        bar = QHBoxLayout()
        bar.setContentsMargins(6, 4, 6, 4)
        self.title = QLabel("No image selected")
        self.title.setStyleSheet("font-weight: 600;")
        bar.addWidget(self.title)
        bar.addStretch(1)
        self.gcp_label = QLabel("Active GCP: —")
        self.gcp_label.setStyleSheet(f"color: {ACCENT};")
        bar.addWidget(self.gcp_label)
        bar.addSpacing(12)
        for text, tip, slot in [("−", "Zoom out", lambda: self.canvas.zoom(1 / 1.25)),
                                ("+", "Zoom in", lambda: self.canvas.zoom(1.25)),
                                ("⤢", "Fit to view (F)", lambda: self.canvas.fit())]:
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            bar.addWidget(b)
        lay.addLayout(bar)

        self.canvas = ImageCanvas(state)
        lay.addWidget(self.canvas, 1)

        readout_row = QHBoxLayout()
        hint = QLabel("  Left-click: place · drag: move · arrows: nudge (Shift = 0.2 px) · "
                      "right-click / Delete: remove · wheel: zoom · middle-drag: pan")
        hint.setStyleSheet(f"color: {FG_MUTED}; padding: 3px 6px;")
        readout_row.addWidget(hint, 1)
        self.marker_readout = QLabel("")
        self.marker_readout.setStyleSheet(f"color: {ACCENT}; padding: 3px 8px; "
                                          "font-family: Consolas, monospace;")
        readout_row.addWidget(self.marker_readout)
        lay.addLayout(readout_row)
        self.canvas.marker_info.connect(self.marker_readout.setText)

        state.active_camera_changed.connect(self._update_title)
        state.active_gcp_changed.connect(self._update_gcp_label)

    def _update_title(self, cam_id: int) -> None:
        cam = self.state.chunk.camera(cam_id)
        self.title.setText(cam.filename if cam else "No image selected")

    def _update_gcp_label(self, _gid: int) -> None:
        g = self.state.active_gcp
        self.gcp_label.setText(f"Active GCP: {g.label}" if g else "Active GCP: —")
