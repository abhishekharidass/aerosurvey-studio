"""Main application window: docked panels, menus, toolbar, pipeline runner."""
from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QDockWidget, QFileDialog, QLabel, QMainWindow,
                               QMessageBox, QProgressBar, QTabWidget, QToolBar,
                               QWidget)

from ..config import APP_NAME, APP_VERSION, IMAGE_EXTENSIONS, PROJECT_EXT
from ..core import engines as enginemod
from ..pipeline import PIPELINE, PipelineWorker
from ..state import AppState
from .dialogs import CrsDialog, EngineStatusDialog, ProcessingSettingsDialog
from .panels.console_panel import ConsolePanel
from .panels.photos_panel import PhotosPanel
from .panels.reference_panel import ReferencePanel
from .panels.workspace_panel import WorkspacePanel
from .views.image_view import ImageView
from .views.model_view import ModelView

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.state = AppState()
        self._worker = None
        self.resize(1400, 880)

        self._build_central()
        self._build_docks()
        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._build_statusbar()

        self.state.dirty_changed.connect(lambda _: self._update_title())
        self.state.project_changed.connect(self._update_title)
        self.state.active_camera_changed.connect(self._on_camera_selected)
        self._update_title()

        self.state.log.emit(f"{APP_NAME} {APP_VERSION} started.", "ok")
        n = sum(1 for e in enginemod.engine_status() if e["available"])
        self.state.log.emit(
            f"{n} external engine(s) detected; missing stages use the built-in simulation.",
            "info" if n else "warn")

    # -- construction ----------------------------------------------------
    def _build_central(self):
        self.tabs = QTabWidget()
        self.model_view = ModelView(self.state)
        self.image_view = ImageView(self.state)
        self.tabs.addTab(self.model_view, "  3D Model & Maps  ")
        self.tabs.addTab(self.image_view, "  Photo / GCP Marking  ")
        self.setCentralWidget(self.tabs)

    def _dock(self, title, widget, area):
        d = QDockWidget(title, self)
        d.setObjectName(title)
        d.setWidget(widget)
        d.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(area, d)
        return d

    def _build_docks(self):
        self.workspace = WorkspacePanel(self.state)
        self.photos = PhotosPanel(self.state)
        self.reference = ReferencePanel(self.state)
        self.console = ConsolePanel(self.state)

        self.dock_workspace = self._dock("Workspace", self.workspace, Qt.LeftDockWidgetArea)
        self.dock_photos = self._dock("Photos", self.photos, Qt.LeftDockWidgetArea)
        self.dock_reference = self._dock("Reference", self.reference, Qt.RightDockWidgetArea)
        self.dock_console = self._dock("Console", self.console, Qt.BottomDockWidgetArea)
        self.resizeDocks([self.dock_workspace, self.dock_photos], [340, 460], Qt.Vertical)
        self.resizeDocks([self.dock_reference], [420], Qt.Horizontal)
        self.resizeDocks([self.dock_console], [180], Qt.Vertical)

    def _build_actions(self):
        self.workflow_actions = []
        mk = lambda text, slot, sc=None, tip="": self._action(text, slot, sc, tip)

        self.act_new = mk("New Project", self.new_project, QKeySequence.New)
        self.act_open = mk("Open Project…", self.open_project, QKeySequence.Open)
        self.act_save = mk("Save Project", self.save_project, QKeySequence.Save)
        self.act_save_as = mk("Save Project As…", self.save_project_as, QKeySequence.SaveAs)
        self.act_import_photos = mk("Import Photos…", self.import_photos, "Ctrl+Shift+P")
        self.act_import_gcps = mk("Import GCPs…", self.import_gcps, "Ctrl+Shift+G")
        self.act_export = mk("Export Products…", self.export_products, "Ctrl+E")
        self.act_report = mk("Generate Processing Report…", self.generate_report_action, "Ctrl+Shift+R")
        self.act_quit = mk("Exit", self.close, "Ctrl+Q")

        self.act_crs = mk("Set Coordinate System…", self.set_crs)
        self.act_settings = mk("Processing Settings…", self.show_settings, "Ctrl+,")
        self.act_engines = mk("Processing Engines…", self.show_engines)
        self.act_sample = mk("Generate Sample Dataset", self.generate_sample)
        self.act_about = mk("About", self.about)

        # workflow stages
        self.stage_actions = {}
        for stage in PIPELINE:
            a = self._action(stage.name, lambda _=False, k=stage.key: self.run_stages([k]))
            self.stage_actions[stage.key] = a
            self.workflow_actions.append(a)
        self.act_run_all = self._action("Run Full Pipeline", self.run_all, "Ctrl+R")
        self.workflow_actions.append(self.act_run_all)

        self.act_cancel = self._action("Cancel", self.cancel_run)
        self.act_cancel.setEnabled(False)

    def _action(self, text, slot, shortcut=None, tip=""):
        a = QAction(text, self)
        a.triggered.connect(slot)
        if shortcut:
            a.setShortcut(shortcut)
        if tip:
            a.setToolTip(tip)
        return a

    def _build_menus(self):
        mb = self.menuBar()
        m_file = mb.addMenu("&File")
        for a in [self.act_new, self.act_open, self.act_save, self.act_save_as]:
            m_file.addAction(a)
        m_file.addSeparator()
        for a in [self.act_import_photos, self.act_import_gcps]:
            m_file.addAction(a)
        m_file.addSeparator()
        m_file.addAction(self.act_export)
        m_file.addAction(self.act_report)
        m_file.addSeparator()
        m_file.addAction(self.act_quit)

        m_flow = mb.addMenu("&Workflow")
        for stage in PIPELINE:
            m_flow.addAction(self.stage_actions[stage.key])
        m_flow.addSeparator()
        m_flow.addAction(self.act_run_all)
        m_flow.addAction(self.act_cancel)

        m_tools = mb.addMenu("&Tools")
        m_tools.addAction(self.act_crs)
        m_tools.addAction(self.act_settings)
        m_tools.addAction(self.act_engines)
        m_tools.addSeparator()
        m_tools.addAction(self.act_sample)

        m_view = mb.addMenu("&View")
        for d in [self.dock_workspace, self.dock_photos, self.dock_reference, self.dock_console]:
            m_view.addAction(d.toggleViewAction())

        mb.addMenu("&Help").addAction(self.act_about)

    def _build_toolbar(self):
        tb = QToolBar("Main")
        tb.setObjectName("MainToolbar")
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        tb.setMovable(False)
        self.addToolBar(tb)
        for a in [self.act_new, self.act_open, self.act_save]:
            tb.addAction(a)
        tb.addSeparator()
        tb.addAction(self.act_import_photos)
        tb.addAction(self.act_import_gcps)
        tb.addAction(self.act_crs)
        tb.addSeparator()
        for stage in PIPELINE:
            tb.addAction(self.stage_actions[stage.key])
        tb.addSeparator()
        tb.addAction(self.act_run_all)
        tb.addAction(self.act_cancel)
        tb.addSeparator()
        tb.addAction(self.act_export)

    def _build_statusbar(self):
        sb = self.statusBar()
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(240)
        self.progress.setVisible(False)
        self.crs_label = QLabel()
        self.engine_label = QLabel()
        n = sum(1 for e in enginemod.engine_status() if e["available"])
        self.engine_label.setText(f"Engines: {n} detected")
        sb.addPermanentWidget(self.crs_label)
        sb.addPermanentWidget(self.progress)
        sb.addPermanentWidget(self.engine_label)
        self.state.project_changed.connect(self._update_crs_label)
        self._update_crs_label()

    # -- title / status --------------------------------------------------
    def _update_title(self):
        star = "•" if self.state._dirty else ""
        self.setWindowTitle(f"{star}{self.state.project.name} — {APP_NAME}")

    def _update_crs_label(self):
        self.crs_label.setText(f"CRS: {self.state.chunk.crs_label}")

    def _on_camera_selected(self, cam_id):
        if cam_id >= 0:
            self.tabs.setCurrentWidget(self.image_view)
        self.photos.select_camera(cam_id)

    # -- file ops --------------------------------------------------------
    def _confirm_discard(self) -> bool:
        if not self.state._dirty:
            return True
        r = QMessageBox.question(self, "Unsaved changes",
                                 "Save changes to the current project?",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Cancel:
            return False
        if r == QMessageBox.Save:
            return self.save_project()
        return True

    def new_project(self):
        if self._confirm_discard():
            self.state.new_project()

    def open_project(self):
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", f"AeroSurvey project (*{PROJECT_EXT})")
        if path:
            try:
                self.state.open_project(path)
            except Exception as exc:
                QMessageBox.critical(self, "Open failed", str(exc))

    def save_project(self) -> bool:
        if not self.state.project.path:
            return self.save_project_as()
        self.state.save_project(self.state.project.path)
        return True

    def save_project_as(self) -> bool:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", self.state.project.name + PROJECT_EXT,
            f"AeroSurvey project (*{PROJECT_EXT})")
        if not path:
            return False
        if not path.endswith(PROJECT_EXT):
            path += PROJECT_EXT
        self.state.save_project(path)
        return True

    def import_photos(self):
        patt = "Images (" + " ".join(f"*{e}" for e in IMAGE_EXTENSIONS) + ")"
        paths, _ = QFileDialog.getOpenFileNames(self, "Import Photos", "", patt)
        if paths:
            self.state.add_photos(paths)
            self._maybe_suggest_crs()

    def import_gcps(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import GCPs", "", "CSV / text (*.csv *.txt *.tsv);;All files (*.*)")
        if path:
            n = self.state.import_gcps_csv(path)
            if n == 0:
                QMessageBox.warning(self, "Import GCPs",
                                    "No rows parsed. Expected columns: label, X, Y, Z[, type].")

    def export_products(self):
        from ..core import export as exportmod
        o = self.state.chunk.outputs
        products = [(k, v) for k, v in o.__dict__.items() if v and os.path.exists(v)]
        if not products:
            QMessageBox.information(self, "Export", "No products to export yet. Run the workflow first.")
            return
        box = QMessageBox(self)
        box.setWindowTitle("Export Products")
        box.setText("Choose the export format:")
        box.setInformativeText(
            "Cloud-ready converts rasters to Cloud-Optimized GeoTIFF (streamable "
            "on web platforms) and point clouds to compressed LAZ.\n"
            "Original copies the files as they are.")
        cloud_btn = box.addButton("Cloud-ready (COG + LAZ)", QMessageBox.AcceptRole)
        preset = exportmod.load_portal_preset()
        preset_btn = None
        if preset is not None:
            preset_btn = box.addButton(preset.PRESET_LABEL, QMessageBox.AcceptRole)
            preset_btn.setToolTip(getattr(preset, "PRESET_TOOLTIP", ""))
        orig_btn = box.addButton("Original files", QMessageBox.AcceptRole)
        box.addButton(QMessageBox.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked not in (cloud_btn, preset_btn, orig_btn) or clicked is None:
            return
        out = QFileDialog.getExistingDirectory(self, "Export products to folder")
        if not out:
            return
        emit = lambda m, lvl="info": self.state.log.emit(m, lvl)
        if preset_btn is not None and clicked is preset_btn:
            done = preset.export_package(
                self.state.chunk, out, self.state.project.name, log=emit)
        else:
            done = exportmod.export_products(
                self.state.chunk, out, cloud_ready=clicked is cloud_btn, log=emit)
        self.state.log.emit(f"Exported {len(done)} product(s) to {out}.", "ok")
        QMessageBox.information(self, "Export",
                                f"Exported {len(done)} product(s) to:\n{out}")

    def generate_report_action(self):
        from ..report import generate_report
        ch = self.state.chunk
        if not ch.stats and not ch.aligned:
            QMessageBox.information(self, "Report",
                                    "Run the workflow first — there is nothing to report yet.")
            return
        default = os.path.join(self.state.workdir(), f"{self.state.project.name}_report.html")
        path, _ = QFileDialog.getSaveFileName(self, "Save Processing Report", default,
                                              "HTML report (*.html)")
        if not path:
            return
        try:
            generate_report(ch, path)
        except Exception as exc:
            QMessageBox.critical(self, "Report", f"Could not generate report:\n{exc}")
            return
        self.state.log.emit(f"Processing report written: {path}", "ok")
        try:
            os.startfile(path)  # open in default browser
        except Exception:
            QMessageBox.information(self, "Report", f"Report saved to:\n{path}")

    # -- tools -----------------------------------------------------------
    def _maybe_suggest_crs(self):
        if self.state.chunk.crs_mode == "local":
            utm = self.state.auto_utm_from_photos()
            if utm:
                self.state.log.emit(
                    f"Geotags detected. Tools ▸ Set Coordinate System suggests EPSG:{utm}.", "info")

    def set_crs(self):
        ch = self.state.chunk
        center = next(((c.lon, c.lat) for c in ch.cameras if c.has_geotag), None)
        dlg = CrsDialog(self, self.state.auto_utm_from_photos(), center_lonlat=center,
                        current_vertical=(ch.vertical_datum, ch.geoid_separation))
        if dlg.exec():
            mode, epsg = dlg.result_crs()
            vdatum, geoid = dlg.result_vertical()
            self.state.set_crs(mode, epsg, vdatum, geoid)
            self._update_crs_label()

    def show_settings(self):
        from ..core import gsd as gsdmod
        from ..pipeline import ml_classify
        ch = self.state.chunk
        est = ch.stats.get("image_gsd_m") or gsdmod.estimate_gsd(ch)
        dlg = ProcessingSettingsDialog(self, ch, estimated_gsd=est,
                                       ml_available=ml_classify.available())
        if dlg.exec():
            self.state.set_dirty()
            self.state.log.emit("Processing settings updated.", "info")

    def show_engines(self):
        EngineStatusDialog(self).exec()

    def generate_sample(self):
        from ..sample import generate
        out_dir = os.path.join(PROJECT_ROOT, "sample_data")
        self.state.log.emit("Generating sample dataset…", "info")
        try:
            images, csv = generate(out_dir)
        except Exception as exc:
            QMessageBox.critical(self, "Sample data", f"Generation failed:\n{exc}")
            return
        self.state.add_photos(images)
        self.state.import_gcps_csv(csv)
        if self.state.chunk.cameras:
            self.state.set_active_camera(self.state.chunk.cameras[0].id)
        if self.state.chunk.gcps:
            self.state.set_active_gcp(self.state.chunk.gcps[0].id)
        self._maybe_suggest_crs()
        QMessageBox.information(
            self, "Sample dataset",
            f"Created {len(images)} photos + {os.path.basename(csv)} in:\n{out_dir}\n\n"
            "Pick a GCP in the Reference panel, then click its target in the image to mark it.")

    def about(self):
        QMessageBox.about(
            self, f"About {APP_NAME}",
            f"<h3>{APP_NAME} {APP_VERSION}</h3>"
            "<p>A photogrammetry workspace for drone/aerial imagery — feature matching, "
            "aerial triangulation, dense cloud, classification, DSM/DTM and orthomosaic.</p>"
            "<p>Orchestrates COLMAP / OpenMVS / PDAL / GDAL where available, with a built-in "
            "simulation fallback. Geospatial stack: pyproj, rasterio, laspy, Open3D.</p>")

    # -- pipeline --------------------------------------------------------
    def run_all(self):
        self.run_stages([s.key for s in PIPELINE])

    def run_stages(self, keys):
        if self._worker and self._worker.isRunning():
            QMessageBox.information(self, "Busy", "A processing run is already in progress.")
            return
        if not self.state.chunk.cameras:
            QMessageBox.information(self, "No photos",
                                    "Import photos before running the workflow.")
            return
        self._set_running(True)
        worker = PipelineWorker(self.state.chunk, keys, self.state.workdir())
        worker.log.connect(self.state.log)
        worker.overall_progress.connect(self.progress.setValue)
        worker.stage_started.connect(lambda name: self.statusBar().showMessage(f"Running: {name}"))
        worker.outputs_changed.connect(self.state.outputs_changed)
        worker.run_finished.connect(self._on_run_finished)
        self._worker = worker
        worker.start()

    def cancel_run(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self.statusBar().showMessage("Cancelling…")

    def _set_running(self, running: bool):
        for a in self.workflow_actions:
            a.setEnabled(not running)
        self.act_cancel.setEnabled(running)
        self.progress.setVisible(running)
        self.progress.setValue(0)

    def _on_run_finished(self, ok: bool):
        self._set_running(False)
        self.statusBar().showMessage("Done." if ok else "Stopped.", 4000)
        # refresh panels that reflect stage side-effects
        self.state.cameras_changed.emit()
        self.state.gcps_changed.emit()
        self.state.project_changed.emit()
        self.state.outputs_changed.emit()
        self._worker = None

    # -- close -----------------------------------------------------------
    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(2000)
        if self._confirm_discard():
            event.accept()
        else:
            event.ignore()
