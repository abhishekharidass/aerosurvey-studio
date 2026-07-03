"""Dark Fusion theme approximating the Metashape / Pix4D desktop look."""
from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

# Core palette
BG_DARK = "#232629"      # window background
BG_PANEL = "#2b2e31"     # docks / panels
BG_INPUT = "#1e2124"     # editable fields / views
FG_TEXT = "#dfe3e6"
FG_MUTED = "#9aa0a6"
ACCENT = "#3d8ec9"       # selection / highlight (survey blue)
ACCENT_DIM = "#2f6e9e"
BORDER = "#3a3d41"
OK_GREEN = "#5cb85c"
WARN_AMBER = "#e0a53b"
ERR_RED = "#d9534f"


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")

    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(BG_DARK))
    pal.setColor(QPalette.WindowText, QColor(FG_TEXT))
    pal.setColor(QPalette.Base, QColor(BG_INPUT))
    pal.setColor(QPalette.AlternateBase, QColor(BG_PANEL))
    pal.setColor(QPalette.ToolTipBase, QColor(BG_PANEL))
    pal.setColor(QPalette.ToolTipText, QColor(FG_TEXT))
    pal.setColor(QPalette.Text, QColor(FG_TEXT))
    pal.setColor(QPalette.Button, QColor(BG_PANEL))
    pal.setColor(QPalette.ButtonText, QColor(FG_TEXT))
    pal.setColor(QPalette.Highlight, QColor(ACCENT))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.Link, QColor(ACCENT))
    pal.setColor(QPalette.PlaceholderText, QColor(FG_MUTED))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor(FG_MUTED))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(FG_MUTED))
    app.setPalette(pal)

    app.setStyleSheet(STYLESHEET)


STYLESHEET = f"""
* {{ outline: 0; }}
QWidget {{ color: {FG_TEXT}; font-size: 12px; }}
QMainWindow, QDialog {{ background: {BG_DARK}; }}

QMenuBar {{ background: {BG_PANEL}; border-bottom: 1px solid {BORDER}; }}
QMenuBar::item {{ padding: 5px 10px; background: transparent; }}
QMenuBar::item:selected {{ background: {ACCENT_DIM}; }}
QMenu {{ background: {BG_PANEL}; border: 1px solid {BORDER}; padding: 4px; }}
QMenu::item {{ padding: 5px 26px 5px 22px; }}
QMenu::item:selected {{ background: {ACCENT_DIM}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 8px; }}

QToolBar {{ background: {BG_PANEL}; border: 0; border-bottom: 1px solid {BORDER}; spacing: 3px; padding: 3px; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 4px; padding: 5px 8px; }}
QToolButton:hover {{ background: {BG_INPUT}; border: 1px solid {BORDER}; }}
QToolButton:pressed, QToolButton:checked {{ background: {ACCENT_DIM}; border: 1px solid {ACCENT}; }}
QToolButton:disabled {{ color: {FG_MUTED}; }}

QDockWidget {{ titlebar-close-icon: none; font-weight: 600; color: {FG_MUTED}; }}
QDockWidget::title {{ background: {BG_DARK}; padding: 6px 8px; border-bottom: 1px solid {BORDER};
    text-transform: uppercase; letter-spacing: 1px; font-size: 11px; }}

QTreeView, QListView, QTableView, QTableWidget, QTreeWidget, QListWidget {{
    background: {BG_INPUT}; border: 1px solid {BORDER}; alternate-background-color: #26292c;
    selection-background-color: {ACCENT_DIM}; gridline-color: {BORDER};
}}
QHeaderView::section {{ background: {BG_PANEL}; color: {FG_MUTED}; padding: 4px 6px;
    border: 0; border-right: 1px solid {BORDER}; border-bottom: 1px solid {BORDER}; }}
QTableView::item:selected, QTreeView::item:selected, QListView::item:selected {{ color: #fff; }}

QPlainTextEdit, QTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {BG_INPUT}; border: 1px solid {BORDER}; border-radius: 4px; padding: 3px 5px;
    selection-background-color: {ACCENT};
}}
QComboBox::drop-down {{ border: 0; width: 18px; }}
QComboBox QAbstractItemView {{ background: {BG_PANEL}; border: 1px solid {BORDER};
    selection-background-color: {ACCENT_DIM}; }}

QPushButton {{ background: {BG_PANEL}; border: 1px solid {BORDER}; border-radius: 4px; padding: 6px 14px; }}
QPushButton:hover {{ border: 1px solid {ACCENT}; }}
QPushButton:pressed {{ background: {ACCENT_DIM}; }}
QPushButton:default {{ border: 1px solid {ACCENT}; }}
QPushButton:disabled {{ color: {FG_MUTED}; }}

QProgressBar {{ background: {BG_INPUT}; border: 1px solid {BORDER}; border-radius: 3px;
    text-align: center; height: 16px; color: {FG_TEXT}; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 2px; }}

QTabBar::tab {{ background: {BG_DARK}; color: {FG_MUTED}; padding: 6px 14px;
    border: 1px solid {BORDER}; border-bottom: 0; }}
QTabBar::tab:selected {{ background: {BG_PANEL}; color: {FG_TEXT}; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; top: -1px; }}

QStatusBar {{ background: {BG_PANEL}; border-top: 1px solid {BORDER}; color: {FG_MUTED}; }}
QSplitter::handle {{ background: {BORDER}; }}
QScrollBar:vertical {{ background: {BG_DARK}; width: 12px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #45494d; border-radius: 5px; min-height: 24px; margin: 2px; }}
QScrollBar::handle:vertical:hover {{ background: #55595d; }}
QScrollBar:horizontal {{ background: {BG_DARK}; height: 12px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: #45494d; border-radius: 5px; min-width: 24px; margin: 2px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QToolTip {{ background: {BG_PANEL}; color: {FG_TEXT}; border: 1px solid {ACCENT}; padding: 4px; }}
"""
