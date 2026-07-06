#!/usr/bin/env python
"""Launch AeroSurvey Studio.

    python main.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aerosurvey.app import main


def _selftest() -> int:
    """Exercise the bundled geospatial/ML stack + engines (verifies a portable build).
    Results are printed and written to selftest_result.txt next to the executable."""
    import tempfile
    import numpy as np
    from aerosurvey.app import _register_bundled_engines, app_dir
    lines = []

    def out(m):
        lines.append(m)
        print(m)

    out("python + numpy OK")
    from pyproj import Transformer
    x, y = Transformer.from_crs(4326, 32645, always_xy=True).transform(88.49, 22.58)
    out(f"pyproj (PROJ data) OK -> {x:.1f}, {y:.1f}")
    import rasterio
    from rasterio.transform import from_origin
    p = tempfile.mktemp(suffix=".tif")
    with rasterio.open(p, "w", driver="GTiff", height=4, width=4, count=1, dtype="float32",
                       crs=rasterio.crs.CRS.from_epsg(32645), transform=from_origin(0, 4, 1, 1)) as d:
        d.write(np.ones((1, 4, 4), "float32"))
    with rasterio.open(p) as s:
        out(f"rasterio (GDAL) OK -> {s.crs}")
    import laspy  # noqa: F401
    out("laspy OK")
    from aerosurvey.pipeline import ml_classify, colmap, openmvs
    out("Random Forest model loaded: " + type(ml_classify.load()).__name__)
    _register_bundled_engines()
    out(f"COLMAP detected: {colmap.available()}  ({colmap.exe() or 'not found'})")
    out(f"OpenMVS detected: {openmvs.available()}")
    out("SELFTEST PASSED")
    try:
        with open(os.path.join(app_dir(), "selftest_result.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    raise SystemExit(main())
