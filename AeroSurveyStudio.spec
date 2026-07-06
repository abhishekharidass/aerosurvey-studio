# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build for a portable AeroSurvey Studio (Windows, onedir).

Build:  pyinstaller --noconfirm AeroSurveyStudio.spec
Output: dist/AeroSurveyStudio/  (copy this whole folder anywhere and run the .exe)
"""
from PyInstaller.utils.hooks import (collect_data_files, collect_submodules,
                                     collect_dynamic_libs)

datas, binaries, hiddenimports = [], [], []

# Geospatial libs bundle their own data (GDAL / PROJ) + native DLLs.
for pkg in ("rasterio", "pyproj", "laspy"):
    datas += collect_data_files(pkg)
    binaries += collect_dynamic_libs(pkg)
    hiddenimports += collect_submodules(pkg)

# scikit-learn / scipy have many dynamically-loaded submodules.
hiddenimports += collect_submodules("sklearn")
hiddenimports += ["scipy._lib.array_api_compat.numpy.fft", "scipy.special._cdflib",
                  "sklearn.utils._typedefs", "sklearn.neighbors._partition_nodes",
                  "sklearn.tree._utils"]

# Ship the trained Random Forest model and the app's own package data.
datas += [("aerosurvey/models/pointcloud_rf.joblib", "aerosurvey/models")]

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["open3d", "matplotlib", "tkinter", "PyQt5", "PyQt6", "pandas",
              "IPython", "notebook", "pytest", "PySide6.QtWebEngineCore",
              "PySide6.Qt3DCore", "PySide6.QtCharts", "PySide6.QtQuick"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="AeroSurveyStudio",
    console=False,         # windowed app (no console window on launch)
    icon=None,
)
coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False, upx=False,
    name="AeroSurveyStudio",
)
