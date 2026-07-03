"""Workspace tree — project / chunk / cameras / GCPs / products overview."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem

from ...theme import FG_MUTED


class WorkspacePanel(QTreeWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.setHeaderHidden(True)
        self.setIndentation(14)
        for sig in (state.project_changed, state.cameras_changed,
                    state.gcps_changed, state.outputs_changed):
            sig.connect(self.rebuild)
        self.itemDoubleClicked.connect(self._on_double_click)
        self.rebuild()

    def rebuild(self) -> None:
        self.clear()
        proj = self.state.project
        ch = proj.active
        root = QTreeWidgetItem([f"\U0001F4C1  {proj.name}"])
        self.addTopLevelItem(root)

        chunk_item = QTreeWidgetItem([f"\U0001F4E6  {ch.name}"])
        root.addChild(chunk_item)

        crs_item = QTreeWidgetItem([f"\U0001F30D  CRS: {ch.crs_label}"])
        crs_item.setForeground(0, Qt.gray)
        chunk_item.addChild(crs_item)

        cams = QTreeWidgetItem([f"\U0001F4F7  Cameras ({len(ch.cameras)})"])
        cams.setData(0, Qt.UserRole, ("cameras", 0))
        chunk_item.addChild(cams)
        aligned = sum(1 for c in ch.cameras if c.aligned)
        if ch.cameras:
            info = QTreeWidgetItem([f"aligned {aligned}/{len(ch.cameras)}"])
            info.setForeground(0, Qt.gray)
            cams.addChild(info)

        gcps = QTreeWidgetItem([f"\U0001F4CD  GCPs ({len(ch.gcps)})"])
        chunk_item.addChild(gcps)
        for g in ch.gcps:
            gi = QTreeWidgetItem([f"{g.label} — {g.marked_count} img"])
            gi.setData(0, Qt.UserRole, ("gcp", g.id))
            gcps.addChild(gi)

        prod = QTreeWidgetItem(["\U0001F5FA  Products"])
        chunk_item.addChild(prod)
        o = ch.outputs
        for label, path in [("Dense cloud", o.dense_cloud), ("Classified", o.classified_cloud),
                            ("DSM", o.dsm), ("DTM", o.dtm), ("Orthomosaic", o.orthomosaic)]:
            done = "✓" if path else "—"
            pi = QTreeWidgetItem([f"{done}  {label}"])
            pi.setForeground(0, Qt.gray if not path else Qt.white)
            prod.addChild(pi)

        self.expandToDepth(2)

    def _on_double_click(self, item, _col) -> None:
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        kind, ident = data
        if kind == "gcp":
            self.state.set_active_gcp(ident)
