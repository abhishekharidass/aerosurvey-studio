"""Application entry point."""
from __future__ import annotations

import os
import sys

from PySide6.QtWidgets import QApplication

from .config import APP_NAME, ORG_NAME
from .theme import apply_theme


def is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def app_dir() -> str:
    """Folder containing the executable (frozen) or the project root (source)."""
    if is_frozen():
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _register_bundled_engines() -> None:
    """Add engines shipped alongside a portable build to PATH so COLMAP/OpenMVS
    are found without a system install."""
    base = os.path.join(app_dir(), "engines")
    for sub in (os.path.join("colmap", "bin"),
                os.path.join("openmvs", "vc17", "x64", "Release"),
                "openmvs", "colmap"):
        p = os.path.join(base, sub)
        if os.path.isdir(p):
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")


def main() -> int:
    _register_bundled_engines()
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    apply_theme(app)

    # Import here so the theme/palette is set before widgets are built.
    from .ui import MainWindow
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
