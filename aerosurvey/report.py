"""Generate a self-contained HTML quality report after processing.

Mirrors a Pix4D / Metashape processing report: project summary, alignment and
georeferencing quality, GCP accuracy table, point-cloud classification, output
specifications, and embedded orthomosaic / DSM / camera-position thumbnails.
No Qt dependency, so it can be produced head-less or from the GUI.
"""
from __future__ import annotations

import base64
import io
import os
from datetime import datetime

import numpy as np

from .config import APP_NAME, APP_VERSION, CLASS_NAMES, CLASS_COLORS


def _png_b64(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()


def _to_png(arr_rgb: np.ndarray) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(arr_rgb, dtype=np.uint8)).save(buf, "PNG")
    return buf.getvalue()


def _raster_thumb(path: str, max_px: int = 620):
    import rasterio
    with rasterio.open(path) as s:
        scale = min(1.0, max_px / max(s.width, s.height))
        h, w = max(int(s.height * scale), 1), max(int(s.width * scale), 1)
        data = s.read(out_shape=(s.count, h, w))
    if data.shape[0] >= 3:
        rgb = np.nan_to_num(np.transpose(data[:3], (1, 2, 0))).clip(0, 255).astype(np.uint8)
    else:
        band = data[0].astype(np.float32)
        fin = np.isfinite(band)
        lo, hi = (np.percentile(band[fin], [2, 98]) if fin.any() else (0, 1))
        n = np.nan_to_num(np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1))
        rgb = np.zeros((*band.shape, 3), np.uint8)
        rgb[..., 0] = (n * 255)
        rgb[..., 1] = (np.abs(np.sin(n * np.pi)) * 200 + 40)
        rgb[..., 2] = ((1 - n) * 255)
    return _to_png(rgb)


def _camera_map(chunk, size: int = 560):
    from PIL import Image, ImageDraw
    cams = [(c.est_x, c.est_y) for c in chunk.cameras if c.est_x is not None]
    gcps = [(g.x, g.y) for g in chunk.gcps]
    pts = cams + gcps
    if len(pts) < 2:
        return None
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    span = max(maxx - minx, maxy - miny, 1.0)
    pad = 40
    img = Image.new("RGB", (size, size), (255, 255, 255))
    d = ImageDraw.Draw(img)

    def px(x, y):
        sx = pad + (x - minx) / span * (size - 2 * pad)
        sy = size - pad - (y - miny) / span * (size - 2 * pad)
        return sx, sy

    for x, y in cams:
        cx, cy = px(x, y)
        d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(61, 142, 201))
    for x, y in gcps:
        cx, cy = px(x, y)
        d.line([cx - 6, cy, cx + 6, cy], fill=(217, 83, 79), width=2)
        d.line([cx, cy - 6, cx, cy + 6], fill=(217, 83, 79), width=2)
    d.text((pad, size - 24), f"{len(cams)} cameras (blue)  ·  {len(gcps)} GCPs (red)",
           fill=(90, 90, 90))
    return _to_png(np.asarray(img))


def _fmt(v, unit="", nd=3):
    if v is None:
        return "&mdash;"
    if isinstance(v, float):
        return f"{v:.{nd}f}{unit}"
    return f"{v}{unit}"


def generate_report(chunk, out_path: str) -> str:
    s = chunk.stats or {}
    o = chunk.outputs
    geotagged = sum(1 for c in chunk.cameras if c.has_geotag)
    models = sorted({(c.make + " " + c.model).strip() for c in chunk.cameras if c.model})
    vdatum = ("Orthometric (MSL), geoid N = %.3f m" % chunk.geoid_separation
              if chunk.vertical_datum == "orthometric" else "Ellipsoidal (GPS)")

    def stat_card(label, value, sub=""):
        return (f'<div class="card"><div class="num">{value}</div>'
                f'<div class="lab">{label}</div>'
                f'{f"<div class=sub>{sub}</div>" if sub else ""}</div>')

    cards = [
        stat_card("Cameras aligned", f'{s.get("cameras_aligned","&mdash;")}/{s.get("cameras_total","&mdash;")}',
                  s.get("align_engine", "")),
        stat_card("Reprojection error", _fmt(s.get("mean_reproj_px"), " px"), "mean"),
        stat_card("Georeferencing", _fmt(s.get("georef_rmse_m") or s.get("control_rmse_m"), " m"),
                  s.get("georef_method", "")),
        stat_card("Dense points", f'{s.get("dense_points",0):,}' if s.get("dense_points") else "&mdash;"),
    ]

    # GCP table
    gcp_rows = ""
    for g in chunk.gcps:
        err = f"{g.error:.3f}" if g.error is not None else "&mdash;"
        ecls = "good" if (g.error is not None and g.error < 0.05) else \
               ("warn" if (g.error is not None and g.error < 0.2) else "bad") if g.error is not None else ""
        gcp_rows += (f"<tr><td>{g.label}</td><td>{g.kind}</td><td>{g.x:.3f}</td><td>{g.y:.3f}</td>"
                     f"<td>{g.z:.3f}</td><td>{g.marked_count}</td>"
                     f"<td class='{ecls}'>{err}</td></tr>")
    gcp_section = ""
    if chunk.gcps:
        gcp_section = f"""
        <h2>Ground Control Points</h2>
        <table><thead><tr><th>Label</th><th>Type</th><th>X / East</th><th>Y / North</th>
        <th>Z</th><th>Images</th><th>Error (m)</th></tr></thead><tbody>{gcp_rows}</tbody></table>
        <p class="note">Control RMSE {_fmt(s.get('control_rmse_m'),' m')}
        &nbsp;·&nbsp; independent check-point RMSE {_fmt(s.get('check_rmse_m'),' m')}</p>"""

    # classification
    cls_section = ""
    counts = s.get("class_counts") or {}
    if counts:
        total = sum(counts.values()) or 1
        bars = ""
        for k, n in sorted(counts.items()):
            col = "rgb(%d,%d,%d)" % CLASS_COLORS.get(int(k), (150, 150, 150))
            pct = 100 * n / total
            bars += (f"<div class='barrow'><span class='blab'>{CLASS_NAMES.get(int(k), k)}</span>"
                     f"<span class='btrack'><span class='bfill' style='width:{pct:.1f}%;background:{col}'></span></span>"
                     f"<span class='bval'>{n:,} ({pct:.1f}%)</span></div>")
        cls_section = f"<h2>Point-Cloud Classification</h2><div class='bars'>{bars}</div>"

    # outputs table
    def spec(d):
        return f"{d['w']}&times;{d['h']} @ {d['gsd_m']:.3f} m/px" if d else "&mdash;"
    out_rows = (f"<tr><td>Orthomosaic</td><td>{spec(s.get('ortho'))}</td>"
                f"<td>{'✓' if o.orthomosaic else '&mdash;'}</td></tr>"
                f"<tr><td>DSM</td><td>{spec(s.get('dsm'))}</td><td>{'✓' if o.dsm else '&mdash;'}</td></tr>"
                f"<tr><td>DTM</td><td>{spec(s.get('dtm'))}</td><td>{'✓' if o.dtm else '&mdash;'}</td></tr>")

    # thumbnails
    thumbs = ""
    try:
        if o.orthomosaic and os.path.exists(o.orthomosaic):
            thumbs += f"<figure><img src='{_png_b64(_raster_thumb(o.orthomosaic))}'><figcaption>Orthomosaic</figcaption></figure>"
        if o.dsm and os.path.exists(o.dsm):
            thumbs += f"<figure><img src='{_png_b64(_raster_thumb(o.dsm))}'><figcaption>Digital Surface Model</figcaption></figure>"
        cm = _camera_map(chunk)
        if cm:
            thumbs += f"<figure><img src='{_png_b64(cm)}'><figcaption>Camera positions &amp; GCPs</figcaption></figure>"
    except Exception:
        pass

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Processing Report — {chunk.name}</title>
<style>
body{{font-family:'Segoe UI',Arial,sans-serif;color:#222;max-width:1000px;margin:24px auto;padding:0 20px;line-height:1.5}}
h1{{font-size:26px;margin:0}} h2{{font-size:18px;border-bottom:2px solid #eee;padding-bottom:6px;margin-top:34px;color:#1b3a44}}
.head{{border-bottom:3px solid #e7a13b;padding-bottom:14px;margin-bottom:8px}}
.muted{{color:#888;font-size:13px}}
.cards{{display:flex;gap:14px;margin:18px 0}}
.card{{flex:1;background:#f6f8f9;border-radius:10px;padding:16px;text-align:center}}
.num{{font-size:28px;font-weight:700;color:#2c7a54}} .lab{{font-size:12px;color:#666;margin-top:4px}}
.sub{{font-size:11px;color:#999}}
table{{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}}
th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid #eee}} th{{background:#f6f8f9;color:#555}}
td.good{{color:#2c7a54;font-weight:600}} td.warn{{color:#b8860b;font-weight:600}} td.bad{{color:#c0392b;font-weight:600}}
.summary td:first-child{{color:#777;width:200px}}
.bars{{margin-top:8px}} .barrow{{display:flex;align-items:center;gap:10px;margin:5px 0;font-size:13px}}
.blab{{width:130px}} .btrack{{flex:1;background:#eee;border-radius:4px;height:14px;overflow:hidden}}
.bfill{{display:block;height:100%}} .bval{{width:150px;text-align:right;color:#666}}
.gallery{{display:flex;flex-wrap:wrap;gap:16px;margin-top:10px}}
figure{{margin:0}} figure img{{max-width:300px;border-radius:8px;border:1px solid #ddd}}
figcaption{{font-size:12px;color:#777;text-align:center;margin-top:4px}}
.note{{font-size:12px;color:#777}} footer{{margin-top:40px;color:#aaa;font-size:12px;border-top:1px solid #eee;padding-top:10px}}
</style></head><body>
<div class="head"><h1>Photogrammetry Processing Report</h1>
<div class="muted">{chunk.name} &nbsp;·&nbsp; generated {datetime.now():%Y-%m-%d %H:%M} &nbsp;·&nbsp; {APP_NAME} {APP_VERSION}</div></div>

<h2>Project Summary</h2>
<table class="summary">
<tr><td>Images</td><td>{len(chunk.cameras)} ({geotagged} geotagged)</td></tr>
<tr><td>Camera(s)</td><td>{', '.join(models) or '&mdash;'}</td></tr>
<tr><td>Horizontal CRS</td><td>{chunk.crs_label}</td></tr>
<tr><td>Vertical datum</td><td>{vdatum}</td></tr>
<tr><td>Ground Control Points</td><td>{len(chunk.gcps)} ({chunk.total_observations} image marks)</td></tr>
</table>

<h2>Processing Quality</h2>
<div class="cards">{''.join(cards)}</div>
<table class="summary">
<tr><td>Alignment engine</td><td>{s.get('align_engine','&mdash;')}</td></tr>
<tr><td>Sparse tie points</td><td>{f"{s.get('sparse_points',0):,}" if s.get('sparse_points') else '&mdash;'}</td></tr>
<tr><td>Bundle-adjustment RMSE</td><td>{_fmt(s.get('ba_rmse_px'),' px')}</td></tr>
<tr><td>Georeferencing method</td><td>{s.get('georef_method','&mdash;')}</td></tr>
</table>
{gcp_section}
{cls_section}

<h2>Output Products</h2>
<table><thead><tr><th>Product</th><th>Specification</th><th>Generated</th></tr></thead>
<tbody>{out_rows}</tbody></table>

<h2>Previews</h2>
<div class="gallery">{thumbs or '<p class="note">No raster products available.</p>'}</div>

<footer>Generated by {APP_NAME} {APP_VERSION} — orchestrating COLMAP · OpenMVS with a NumPy/SciPy
geospatial &amp; optimisation core.</footer>
</body></html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
