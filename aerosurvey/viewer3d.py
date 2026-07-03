"""Standalone interactive Open3D point-cloud viewer.

Run in its own process so the Qt UI stays responsive:
    python -m aerosurvey.viewer3d <cloud.las> [rgb|class]
"""
from __future__ import annotations

import sys

import numpy as np

from .config import CLASS_COLORS


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    if not argv:
        print("usage: python -m aerosurvey.viewer3d <cloud.las> [rgb|class]")
        return 2
    path = argv[0]
    mode = argv[1] if len(argv) > 1 else "rgb"

    import laspy
    import open3d as o3d

    las = laspy.read(path)
    P = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
    if mode == "class":
        cls = np.array(las.classification)
        C = np.array([CLASS_COLORS.get(int(c), (200, 200, 200)) for c in cls], np.float64) / 255.0
    else:
        C = np.column_stack([las.red, las.green, las.blue]).astype(np.float64) / 65535.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P - P.mean(axis=0))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(C, 0, 1))
    o3d.visualization.draw_geometries([pcd], window_name=f"AeroSurvey — {path} [{mode}]",
                                      width=1100, height=750)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
