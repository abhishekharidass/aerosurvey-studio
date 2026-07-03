"""Log console — colour-coded messages, like the Metashape console."""
from __future__ import annotations

from datetime import datetime

from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QPlainTextEdit

from ...theme import ACCENT, ERR_RED, FG_MUTED, FG_TEXT, OK_GREEN, WARN_AMBER

_COLORS = {
    "info": FG_TEXT,
    "ok": OK_GREEN,
    "warn": WARN_AMBER,
    "error": ERR_RED,
    "stage": ACCENT,
}


class ConsolePanel(QPlainTextEdit):
    def __init__(self, state):
        super().__init__()
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)
        self.setStyleSheet("QPlainTextEdit { font-family: Consolas, 'Courier New', monospace;"
                           " font-size: 11px; }")
        state.log.connect(self.append_message)
        self.append_message("AeroSurvey Studio console ready.", "info")

    def append_message(self, message: str, level: str = "info") -> None:
        color = _COLORS.get(level, FG_TEXT)
        ts = datetime.now().strftime("%H:%M:%S")
        html = (f'<span style="color:{FG_MUTED}">[{ts}]</span> '
                f'<span style="color:{color}">{self._escape(message)}</span>')
        self.appendHtml(html)
        self.moveCursor(QTextCursor.End)

    @staticmethod
    def _escape(text: str) -> str:
        return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
