"""Tests for the processing/quality report generator."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aerosurvey.model.project import Chunk
from aerosurvey.report import _hms, generate_report


def test_hms():
    assert _hms(42) == "42 s"
    assert _hms(125) == "2 min 05 s"
    assert _hms(3725) == "1 h 02 min"


def test_report_includes_quality_sections(tmp_path):
    ch = Chunk()
    ch.stats = {
        "cameras_total": 56, "cameras_aligned": 56, "align_engine": "COLMAP",
        "mean_reproj_px": 0.812, "sparse_points": 43210, "ba_rmse_px": 0.64,
        "georef_method": "Camera GPS", "georef_rmse_m": 1.532,
        "georef_rmse_xyz_m": [0.61, 0.72, 1.18],
        "calibration": {"focal_px": 3702.15, "cx": 2640.5, "cy": 1978.1,
                        "k1": -0.00123, "source": "self-calibrated (bundle adjustment)"},
        "dense_points": 4400000, "dense_density_ppm2": 210.0,
        "mesh_vertices": 1200000, "mesh_faces": 2400000,
        "stage_seconds": {"align": 300.0, "dense": 3900.0, "mesh": 2500.0},
    }
    out = str(tmp_path / "report.html")
    generate_report(ch, out)
    html = open(out, encoding="utf-8").read()

    # per-axis accuracy
    assert "X 0.610 m" in html and "Z 1.180 m" in html
    # calibration table
    assert "Camera Calibration" in html
    assert "3702.15 px" in html and "-0.001230" in html
    # outputs rows for cloud + mesh
    assert "4,400,000 points" in html and "210 pts/m²" in html.replace("&sup2;", "²")
    assert "1,200,000 vertices" in html and "2,400,000 faces" in html
    # timings
    assert "Processing Time" in html
    assert "5 min 00 s" in html                # align
    assert "1 h 05 min" in html                # dense
    assert "1 h 51 min" in html or "Total" in html


def test_report_degrades_without_new_stats(tmp_path):
    ch = Chunk()
    ch.stats = {"cameras_total": 5, "cameras_aligned": 5,
                "align_engine": "simulation", "mean_reproj_px": 1.0}
    out = str(tmp_path / "report.html")
    generate_report(ch, out)
    html = open(out, encoding="utf-8").read()
    assert "Camera Calibration" not in html
    assert "Processing Time" not in html
    assert "Photogrammetry Processing Report" in html
