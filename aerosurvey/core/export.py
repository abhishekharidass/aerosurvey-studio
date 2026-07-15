"""Cloud-ready product export.

Web platforms (drone-data portals, GIS servers, digital-twin sites) ingest
the same standard formats:
  * rasters  -> Cloud-Optimized GeoTIFF (COG: tiled + overview pyramid, so
    viewers stream any zoom level over HTTP without reading the whole file);
  * point clouds -> LAZ (compressed LAS, typically 5-10x smaller).

`export_products` converts every available product of a chunk into a target
folder; plain copies remain available for tools that want the originals.
"""
from __future__ import annotations

import os
import shutil
from typing import Callable, List, Tuple

PRODUCT_LABELS = {
    "sparse_cloud": "Sparse cloud",
    "dense_cloud": "Dense cloud",
    "classified_cloud": "Classified cloud",
    "mesh": "Textured mesh",
    "dsm": "DSM",
    "dtm": "DTM",
    "orthomosaic": "Orthomosaic",
}


def _copy_mesh_bundle(obj_path: str, out_dir: str) -> str:
    """Copy an OBJ together with its .mtl / texture / offset sidecars."""
    src_dir = os.path.dirname(obj_path)
    dst_dir = os.path.join(out_dir, "mesh")
    os.makedirs(dst_dir, exist_ok=True)
    for fn in os.listdir(src_dir):
        shutil.copy2(os.path.join(src_dir, fn), os.path.join(dst_dir, fn))
    return os.path.join(dst_dir, os.path.basename(obj_path))


def to_cog(src: str, dst: str) -> None:
    """Convert a GeoTIFF to a Cloud-Optimized GeoTIFF (GDAL >= 3.1)."""
    import rasterio
    from rasterio.shutil import copy as rio_copy
    with rasterio.open(src) as ds:
        is_int = ds.dtypes[0].startswith(("uint", "int"))
    # DEFLATE is lossless and universally readable; predictor helps floats.
    rio_copy(src, dst, driver="COG", compress="DEFLATE",
             predictor="2" if is_int else "3", BIGTIFF="IF_SAFER")


def to_laz(src: str, dst: str) -> None:
    """Rewrite a LAS file as compressed LAZ (needs the lazrs backend)."""
    import laspy
    las = laspy.read(src)
    las.write(dst)          # .laz extension selects LAZ compression


def laz_available() -> bool:
    try:
        import lazrs  # noqa: F401
        return True
    except ImportError:
        try:
            import laszip  # noqa: F401
            return True
        except ImportError:
            return False


def load_portal_preset():
    """Optional local-only export preset (aerosurvey/core/portal_export.py,
    not part of the repository). Returns the module or None.

    A preset module defines PRESET_LABEL, PRESET_TOOLTIP and
    export_package(chunk, out_dir, project_name, log=None)."""
    try:
        from . import portal_export
        return portal_export
    except ImportError:
        return None


def export_products(chunk, out_dir: str, cloud_ready: bool = True,
                    log: Callable[[str, str], None] = None) -> List[Tuple[str, str]]:
    """Export every existing product. Returns [(label, exported_path)].

    cloud_ready: rasters -> COG, clouds -> LAZ; otherwise plain copies.
    """
    log = log or (lambda m, lvl="info": None)
    done: List[Tuple[str, str]] = []
    use_laz = cloud_ready and laz_available()
    if cloud_ready and not use_laz:
        log("LAZ backend (lazrs) not installed — point clouds exported as LAS.",
            "warn")
    for key, src in chunk.outputs.__dict__.items():
        if not src or not os.path.exists(src):
            continue
        label = PRODUCT_LABELS.get(key, key)
        base = os.path.splitext(os.path.basename(src))[0]
        try:
            if key == "mesh":
                dst = _copy_mesh_bundle(src, out_dir)
            elif cloud_ready and src.lower().endswith((".tif", ".tiff")):
                dst = os.path.join(out_dir, base + "_cog.tif")
                to_cog(src, dst)
            elif use_laz and src.lower().endswith(".las"):
                dst = os.path.join(out_dir, base + ".laz")
                to_laz(src, dst)
            else:
                dst = os.path.join(out_dir, os.path.basename(src))
                shutil.copy2(src, dst)
            done.append((label, dst))
            mb = os.path.getsize(dst) / 1e6
            log(f"Exported {label}: {os.path.basename(dst)} ({mb:.1f} MB)", "ok")
        except Exception as exc:
            log(f"Export failed for {label}: {exc}", "error")
    return done
