"""Small modal dialogs: coordinate-system picker and engine status."""
from __future__ import annotations

from typing import Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
                               QMessageBox, QPushButton, QRadioButton, QSpinBox,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ..core import crs as crsmod
from ..core import engines as enginemod
from ..theme import ERR_RED, FG_MUTED, OK_GREEN


class CrsDialog(QDialog):
    """Pick Local / UTM / explicit EPSG for the active chunk."""

    def __init__(self, parent, suggested_utm: Optional[int], center_lonlat=None,
                 current_vertical=("ellipsoidal", 0.0)):
        super().__init__(parent)
        self.setWindowTitle("Coordinate System")
        self.setMinimumWidth(440)
        self._result: Tuple[str, Optional[int]] = ("local", None)
        self._center = center_lonlat
        self._vertical = tuple(current_vertical)

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Choose how project coordinates and GCPs are interpreted:"))

        self.group = QButtonGroup(self)
        self.rb_local = QRadioButton("Local / arbitrary coordinates (no georeferencing)")
        self.rb_utm = QRadioButton("WGS84 / UTM")
        self.rb_epsg = QRadioButton("Explicit EPSG code")
        for rb in (self.rb_local, self.rb_utm, self.rb_epsg):
            self.group.addButton(rb)
            lay.addWidget(rb)

        form = QFormLayout()
        self.utm_spin = QSpinBox()
        self.utm_spin.setRange(32601, 32760)
        if suggested_utm:
            self.utm_spin.setValue(suggested_utm)
            self.rb_utm.setChecked(True)
        else:
            self.rb_local.setChecked(True)
        utm_row = QWidget()
        ur = QHBoxLayout(utm_row)
        ur.setContentsMargins(0, 0, 0, 0)
        ur.addWidget(self.utm_spin)
        self.utm_hint = QLabel("")
        self.utm_hint.setStyleSheet(f"color: {FG_MUTED};")
        ur.addWidget(self.utm_hint, 1)
        form.addRow("UTM EPSG:", utm_row)

        self.epsg_spin = QSpinBox()
        self.epsg_spin.setRange(1024, 999999)
        self.epsg_spin.setValue(suggested_utm or 32633)
        self.epsg_hint = QLabel("")
        self.epsg_hint.setStyleSheet(f"color: {FG_MUTED};")
        erow = QWidget()
        er = QHBoxLayout(erow)
        er.setContentsMargins(0, 0, 0, 0)
        er.addWidget(self.epsg_spin)
        er.addWidget(self.epsg_hint, 1)
        form.addRow("EPSG code:", erow)
        lay.addLayout(form)

        lay.addWidget(QLabel("Vertical datum (height reference):"))
        vform = QFormLayout()
        self.vdatum = QComboBox()
        self.vdatum.addItems(["Ellipsoidal (raw GPS height)",
                              "Orthometric — mean sea level (geoid)"])
        self.vdatum.setCurrentIndex(1 if self._vertical[0] == "orthometric" else 0)
        vform.addRow("Datum:", self.vdatum)
        self.geoid_spin = QDoubleSpinBox()
        self.geoid_spin.setRange(-200.0, 200.0)
        self.geoid_spin.setDecimals(3)
        self.geoid_spin.setSuffix(" m")
        self.geoid_spin.setValue(float(self._vertical[1]))
        auto_btn = QPushButton("Auto (EGM2008)")
        auto_btn.clicked.connect(self._auto_geoid)
        grow = QWidget()
        gl = QHBoxLayout(grow)
        gl.setContentsMargins(0, 0, 0, 0)
        gl.addWidget(self.geoid_spin)
        gl.addWidget(auto_btn)
        vform.addRow("Geoid separation N:", grow)
        lay.addLayout(vform)
        self.vdatum.currentIndexChanged.connect(self._update_vertical)
        self._update_vertical()

        if suggested_utm:
            lay.addWidget(self._note(f"Suggested UTM zone from photo geotags: "
                                     f"EPSG:{suggested_utm}"))

        self.utm_spin.valueChanged.connect(self._update_hints)
        self.epsg_spin.valueChanged.connect(self._update_hints)
        self._update_hints()

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _note(self, text):
        lab = QLabel(text)
        lab.setStyleSheet(f"color: {OK_GREEN};")
        return lab

    def _update_hints(self):
        self.utm_hint.setText(crsmod.describe(self.utm_spin.value()))
        ok = crsmod.is_valid_epsg(self.epsg_spin.value())
        self.epsg_hint.setText(crsmod.describe(self.epsg_spin.value()) if ok
                               else "unknown EPSG")
        self.epsg_hint.setStyleSheet(f"color: {OK_GREEN if ok else ERR_RED};")

    def _accept(self):
        if self.rb_local.isChecked():
            self._result = ("local", None)
        elif self.rb_utm.isChecked():
            self._result = ("utm", self.utm_spin.value())
        else:
            if not crsmod.is_valid_epsg(self.epsg_spin.value()):
                QMessageBox.warning(self, "Invalid EPSG", "That EPSG code was not recognised.")
                return
            self._result = ("epsg", self.epsg_spin.value())
        datum = "orthometric" if self.vdatum.currentIndex() == 1 else "ellipsoidal"
        self._vertical = (datum, self.geoid_spin.value() if datum == "orthometric" else 0.0)
        self.accept()

    def _update_vertical(self):
        self.geoid_spin.setEnabled(self.vdatum.currentIndex() == 1)

    def _auto_geoid(self):
        if not self._center:
            QMessageBox.information(self, "Geoid", "No photo geotags to locate the survey area.")
            return
        lon, lat = self._center
        n = crsmod.geoid_separation(lon, lat)
        if n is None:
            QMessageBox.information(self, "Geoid separation",
                                    "EGM2008 geoid grid is unavailable offline. Enter N manually "
                                    "(e.g. from an NGS/UNAVCO geoid calculator for your area).")
        else:
            self.geoid_spin.setValue(n)
            self.vdatum.setCurrentIndex(1)

    def result_crs(self) -> Tuple[str, Optional[int]]:
        return self._result

    def result_vertical(self):
        return self._vertical


class ProcessingSettingsDialog(QDialog):
    """Edit the chunk's ProcessingSettings (resolution, density, quality...)."""

    def __init__(self, parent, chunk, estimated_gsd=None, ml_available=False):
        super().__init__(parent)
        self.setWindowTitle("Processing Settings")
        self.setMinimumWidth(480)
        self.chunk = chunk
        s = chunk.settings

        lay = QVBoxLayout(self)
        est = (f"Estimated image GSD: {estimated_gsd*100:.1f} cm/px"
               if estimated_gsd else
               "Image GSD not estimable yet (needs EXIF focal + altitude, "
               "or a solved alignment).")
        est_lab = QLabel(est)
        est_lab.setStyleSheet(f"color: {OK_GREEN if estimated_gsd else FG_MUTED};")
        lay.addWidget(est_lab)

        form = QFormLayout()

        def gsd_row(mode, value):
            combo = QComboBox()
            combo.addItems(["Auto (image GSD)", "Custom"])
            combo.setCurrentIndex(1 if mode == "custom" else 0)
            spin = QDoubleSpinBox()
            spin.setRange(0.005, 10.0)
            spin.setDecimals(3)
            spin.setSingleStep(0.01)
            spin.setSuffix(" m/px")
            spin.setValue(value)
            spin.setEnabled(mode == "custom")
            combo.currentIndexChanged.connect(lambda i, sp=spin: sp.setEnabled(i == 1))
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(combo)
            h.addWidget(spin)
            return row, combo, spin

        row, self.ortho_mode, self.ortho_spin = gsd_row(s.ortho_gsd_mode, s.ortho_gsd)
        form.addRow("Orthomosaic resolution:", row)
        row, self.surf_mode, self.surf_spin = gsd_row(s.surface_gsd_mode, s.surface_gsd)
        form.addRow("DSM / DTM resolution:", row)

        self.max_dim = QSpinBox()
        self.max_dim.setRange(2000, 100000)
        self.max_dim.setSingleStep(1000)
        self.max_dim.setSuffix(" px")
        self.max_dim.setValue(int(s.max_raster_dim))
        self.max_dim.setToolTip("Safety cap on raster width/height; finer GSDs "
                                "are coarsened to fit.")
        form.addRow("Max raster dimension:", self.max_dim)

        self.quality = QComboBox()
        self.quality.addItems(["Ultra (full resolution)", "High (1/2)",
                               "Medium (1/4)", "Low (1/8)"])
        self.quality.setCurrentIndex(
            {"ultra": 0, "high": 1, "medium": 2, "low": 3}.get(s.dense_quality, 1))
        form.addRow("Dense matching quality:", self.quality)

        self.density = QDoubleSpinBox()
        self.density.setRange(0.0, 100000.0)
        self.density.setDecimals(0)
        self.density.setSuffix(" pts/m²")
        self.density.setSpecialValueText("Native (no limit)")
        self.density.setValue(s.dense_target_density)
        self.density.setToolTip("Thin the dense cloud to this density after "
                                "matching. 0 keeps every point.")
        form.addRow("Dense cloud density:", self.density)

        self.classifier = QComboBox()
        self.classifier.addItems(["Rule-based (morphology + geometry)",
                                  "Machine learning (Random Forest)"])
        self.classifier.setCurrentIndex(1 if s.classifier == "ml" else 0)
        if not ml_available:
            self.classifier.setCurrentIndex(0)
            self.classifier.model().item(1).setEnabled(False)
            self.classifier.setToolTip("No trained model found "
                                       "(aerosurvey/models/pointcloud_rf.joblib).")
        form.addRow("Point classifier:", self.classifier)

        self.max_feat = QSpinBox()
        self.max_feat.setRange(1024, 65536)
        self.max_feat.setSingleStep(1024)
        self.max_feat.setValue(int(s.sfm_max_features))
        self.max_feat.setToolTip("SIFT features per image. More = denser sparse "
                                 "cloud and better matching, slower alignment.")
        form.addRow("SfM features per image:", self.max_feat)

        lay.addLayout(form)
        note = QLabel("Auto resolution matches the outputs to what the sensor "
                      "captured. Values apply to the next processing run.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {FG_MUTED};")
        lay.addWidget(note)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _accept(self):
        s = self.chunk.settings
        s.ortho_gsd_mode = "custom" if self.ortho_mode.currentIndex() == 1 else "auto"
        s.ortho_gsd = float(self.ortho_spin.value())
        s.surface_gsd_mode = "custom" if self.surf_mode.currentIndex() == 1 else "auto"
        s.surface_gsd = float(self.surf_spin.value())
        s.max_raster_dim = int(self.max_dim.value())
        s.dense_quality = ["ultra", "high", "medium", "low"][self.quality.currentIndex()]
        s.dense_target_density = float(self.density.value())
        s.classifier = "ml" if self.classifier.currentIndex() == 1 else "rules"
        s.sfm_max_features = int(self.max_feat.value())
        self.accept()


class EngineStatusDialog(QDialog):
    """Report which external engines are detected on PATH."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Processing Engines")
        self.setMinimumWidth(560)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("External engines the pipeline can orchestrate. Missing engines "
                             "fall back to the built-in simulation."))
        rows = enginemod.engine_status()
        table = QTableWidget(len(rows), 3)
        table.setHorizontalHeaderLabels(["Engine", "Role", "Status"])
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        table.setColumnWidth(0, 120)
        table.setColumnWidth(1, 300)
        for r, e in enumerate(rows):
            table.setItem(r, 0, QTableWidgetItem(e["name"]))
            table.setItem(r, 1, QTableWidgetItem(e["role"]))
            status = QTableWidgetItem("● detected" if e["available"] else "○ not found (simulated)")
            status.setForeground(Qt.green if e["available"] else Qt.gray)
            status.setToolTip(e["path"] or "")
            table.setItem(r, 2, status)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        lay.addWidget(table)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        bb.accepted.connect(self.accept)
        bb.button(QDialogButtonBox.Close).clicked.connect(self.accept)
        lay.addWidget(bb)


class ContourDialog(QDialog):
    """Options for the contour-lines export (source, interval, formats)."""

    def __init__(self, parent, has_dtm: bool, has_dsm: bool):
        super().__init__(parent)
        from PySide6.QtWidgets import QCheckBox
        self.setWindowTitle("Export Contour Lines")
        self.setMinimumWidth(380)

        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.source = QComboBox()
        if has_dtm:
            self.source.addItem("DTM (terrain — recommended)", "dtm")
        if has_dsm:
            self.source.addItem("DSM (surface incl. buildings)", "dsm")
        form.addRow("Elevation source:", self.source)

        self.interval = QDoubleSpinBox()
        self.interval.setRange(0.1, 100.0)
        self.interval.setDecimals(2)
        self.interval.setSingleStep(0.5)
        self.interval.setValue(1.0)
        self.interval.setSuffix(" m")
        form.addRow("Contour interval:", self.interval)
        lay.addLayout(form)

        lay.addWidget(QLabel("Output formats:"))
        self.cb_shp = QCheckBox("Shapefile (.shp — GIS)")
        self.cb_dxf = QCheckBox("DXF (.dxf — CAD)")
        self.cb_geojson = QCheckBox("GeoJSON (.geojson — web)")
        for cb in (self.cb_shp, self.cb_dxf, self.cb_geojson):
            cb.setChecked(True)
            lay.addWidget(cb)

        hint = QLabel("Every 5th level is flagged as an index contour "
                      "(CONTOUR_MAJOR layer / INDEX attribute).")
        hint.setStyleSheet(f"color: {FG_MUTED};")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def result_options(self):
        formats = tuple(f for f, cb in (("shp", self.cb_shp),
                                        ("dxf", self.cb_dxf),
                                        ("geojson", self.cb_geojson))
                        if cb.isChecked())
        return (self.source.currentData(), float(self.interval.value()), formats)
