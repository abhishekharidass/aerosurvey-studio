# AeroSurvey Studio

A desktop **photogrammetry workspace** for drone / aerial imagery, in the spirit of
Agisoft Metashape and Pix4D. Built with PySide6 (Qt) and a Python geospatial stack.

The application is architected to **orchestrate proven open-source engines**
(COLMAP, OpenMVS, PDAL, GDAL). Where an engine is not installed, the matching
pipeline stage falls back to a built-in **simulation** that produces real,
well-formed outputs (a LAS point cloud and float32 GeoTIFF DSM/DTM/ortho) so the
entire application is usable and testable today.

## Status: scaffold (v0.1)

This is the **full application scaffold** — every panel and the end-to-end
workflow are in place and working. Real engine calls replace the simulated
stages one module at a time (see *Wiring real engines* below).

## What works now

| Area | Capability |
|------|-----------|
| **Project** | New / Open / Save (`.asproj` JSON), multi-panel dockable UI, dark theme |
| **Data ingestion** | Import photos, read EXIF geotags + DJI/XMP gimbal attitude, image sizes |
| **Coordinate systems** | Local / arbitrary, WGS84 UTM (auto zone from geotags), explicit EPSG — via `pyproj` |
| **Photos** | List, sort by name / capture time / **proximity to a selected GCP** |
| **GCPs** | Add / edit / remove, import from CSV (`label,x,y,z[,type]`), control vs check points |
| **GCP marking** | Zoom, pan, **click to place**, **drag to move**, **right-click / Delete to remove** markers; every GCP shown per image, active one highlighted; live pixel readout |
| **Pipeline** | Align → Dense cloud → Classify → DSM → DTM → Orthomosaic, on a background thread with progress + cancel + console log |
| **Point cloud** | Classified LAS (ground / vegetation / building), top-down preview, **interactive Open3D viewer**, colour by RGB or class |
| **Surfaces** | DSM (max-Z), DTM (ground-only), Orthomosaic (RGB) as **LZW GeoTIFF with EPSG + geotransform** — opens natively in QGIS |
| **Export** | Copy all products (LAS + GeoTIFFs) to a folder |

## Requirements

```
pip install -r requirements.txt
```

Core deps: PySide6, Pillow, numpy, pyproj, rasterio, laspy, open3d
(optional: piexif — only used to embed GPS into the generated sample dataset).

## Run

```
python main.py
```

Then either **Tools ▸ Generate Sample Dataset** (creates 6 photos with survey
targets + a GCP CSV and loads them) or **File ▸ Import Photos**.

### Try the GCP-marking workflow in 30 seconds
1. `Tools ▸ Generate Sample Dataset`
2. Pick a GCP row in the **Reference** panel (right).
3. In the **Photo / GCP Marking** tab, click the matching numbered target in the image.
4. Wheel to zoom, drag the marker to nudge it, right-click to remove.
5. Select other photos in the **Photos** panel and mark the same GCP across them.
6. `Workflow ▸ Run Full Pipeline` to build the cloud, DSM/DTM and orthomosaic.

## Architecture

```
aerosurvey/
  app.py            QApplication bootstrap + theme
  config.py         constants, classification classes/colours
  theme.py          dark Fusion stylesheet
  state.py          AppState: single source of truth + Qt signals
  core/
    exif.py         EXIF geotag + XMP attitude reader
    crs.py          local / UTM / EPSG transforms (pyproj)
    engines.py      detect COLMAP / OpenMVS / PDAL / GDAL on PATH
  model/
    camera.py gcp.py project.py   data model + .asproj (de)serialisation
  pipeline/
    stages.py       6 workflow stages (simulated, real-output)
    worker.py       QThread runner with progress/log/cancel signals
  ui/
    main_window.py  docks, menus, toolbar, pipeline wiring
    panels/         workspace, photos, reference (GCP table), console
    views/          image_view (GCP marking canvas), model_view (3D + rasters)
  viewer3d.py       standalone Open3D subprocess viewer
  sample.py         synthetic dataset generator (targets + geotags + CSV)
```

Panels never talk to each other directly — they mutate `AppState` and react to
its signals, which keeps the UI loosely coupled.

## Wiring real engines

Each stage in `aerosurvey/pipeline/stages.py` is a function `run_x(ctx)` that
today produces simulated output. To go production for a stage, replace its body
with a `subprocess` call to the real engine and parse its result:

- **align** → COLMAP `feature_extractor` / `exhaustive_matcher` / `mapper`, then
  read cameras + sparse cloud; inject GCPs into a second bundle adjustment.
- **dense** → OpenMVS `DensifyPointCloud` (or COLMAP `patch_match_stereo` +
  `stereo_fusion`).
- **classify** → PDAL pipeline (`filters.smrf` / `filters.csf` for ground).
- **dsm / dtm / ortho** → PDAL `writers.gdal` + GDAL `gdal_translate` / `gdalwarp`.

`engines.py` already reports which of these are present on the machine
(**Tools ▸ Processing Engines**), so a stage can choose engine-vs-simulation at
runtime.
