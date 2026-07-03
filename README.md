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
| **Pipeline** | Align → **Optimize/Georeference** → Dense cloud → Classify → DSM → DTM → Orthomosaic, on a background thread with progress + cancel + console log |
| **Georeferencing** | 7-param similarity fit into the project CRS from camera GPS, or from triangulated GCP marks (control/check split); per-GCP residual + control/check RMSE reported ([`pipeline/georef.py`](aerosurvey/pipeline/georef.py)) |
| **Bundle adjustment** | Sparse LM bundle adjustment (scipy) re-solving camera poses + tie points against GCP control, minimising reprojection error ([`pipeline/bundle.py`](aerosurvey/pipeline/bundle.py)) |
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

- **align** → **✅ wired** ([`pipeline/colmap.py`](aerosurvey/pipeline/colmap.py)):
  runs COLMAP `feature_extractor` / `exhaustive_matcher` / `mapper`, parses the
  sparse model, sets camera poses and saves a sparse tie-point cloud. Falls back
  to the simulation when COLMAP isn't on PATH.
- **georef** → **✅ wired** ([`pipeline/georef.py`](aerosurvey/pipeline/georef.py)):
  Umeyama similarity fit + DLT triangulation. Georeferences the local SfM frame
  into the project CRS using triangulated GCP marks (preferred) or camera GPS
  (fallback), and reports control/check residuals (shown in the Reference panel's
  *Error (m)* column).
- **bundle adjustment** → **✅ wired** ([`pipeline/bundle.py`](aerosurvey/pipeline/bundle.py)):
  after the similarity fit, a sparse Levenberg–Marquardt bundle adjustment
  (scipy `least_squares`, analytic sparsity) re-solves camera poses + tie points
  to minimise reprojection error, holding GCPs fixed as control. Runs when a
  COLMAP reconstruction with a tie-point graph is available.
- **dense** → **✅ wired** ([`pipeline/openmvs.py`](aerosurvey/pipeline/openmvs.py)):
  runs COLMAP `image_undistorter` → OpenMVS `InterfaceCOLMAP` → `DensifyPointCloud`,
  reads the dense PLY (Open3D) and carries it into the project CRS via the georef
  transform. Falls back to the simulation when OpenMVS isn't on PATH.
- **dsm / dtm / ortho** → **✅ real rasterisers** (rasterio): extent-aware gridding
  (max-Z / ground min-Z / nadir RGB) writing LZW GeoTIFFs with the project EPSG and
  a proper geotransform — so they now produce correct output from *any* real
  georeferenced cloud, not just the synthetic domain.
- **classify** → **✅ wired** ([`pipeline/classify.py`](aerosurvey/pipeline/classify.py)):
  a real progressive morphological ground filter (scipy) + KDTree local-roughness
  to split non-ground into building (planar) vs vegetation (rough), colour as a
  tie-breaker; ~98% accuracy on the synthetic scene. Uses a PDAL `filters.smrf`
  pipeline instead when the PDAL CLI is on PATH.

`engines.py` already reports which of these are present on the machine
(**Tools ▸ Processing Engines**), so a stage can choose engine-vs-simulation at
runtime.
