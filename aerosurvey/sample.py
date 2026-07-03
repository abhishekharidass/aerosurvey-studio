"""Generate a small synthetic dataset (photos with survey targets + a GCP CSV)
so the marking workflow can be exercised without real drone imagery."""
from __future__ import annotations

import math
import os
import random
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont

W, H = 1600, 1200
N_IMAGES = 6
# Five GCPs with arbitrary local coordinates (metres).
GCPS = [
    ("GCP1", 12.40, 8.10, 101.20),
    ("GCP2", 88.75, 15.60, 100.85),
    ("GCP3", 45.10, 62.30, 103.40),
    ("GCP4", 120.90, 70.20, 102.10),
    ("GCP5", 70.00, 110.50, 104.75),
]
# Approx WGS84 anchor (for optional geotags) — a field somewhere generic.
BASE_LAT, BASE_LON, BASE_ALT = 47.3980, 8.5460, 520.0


def _font(size: int):
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_target(d: ImageDraw.ImageDraw, x: int, y: int, label: str):
    r = 26
    # checker target
    d.ellipse([x - r, y - r, x + r, y + r], fill=(245, 245, 245), outline=(20, 20, 20), width=2)
    d.pieslice([x - r, y - r, x + r, y + r], 0, 90, fill=(15, 15, 15))
    d.pieslice([x - r, y - r, x + r, y + r], 180, 270, fill=(15, 15, 15))
    d.line([x - r - 8, y, x + r + 8, y], fill=(220, 40, 40), width=1)
    d.line([x, y - r - 8, x, y + r + 8], fill=(220, 40, 40), width=1)
    d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(220, 40, 40))
    f = _font(20)
    d.text((x + r + 6, y - r - 4), label, fill=(255, 230, 120), font=f)


def _background(seed: int) -> Image.Image:
    rng = random.Random(seed)
    img = Image.new("RGB", (W, H), (120, 135, 95))
    d = ImageDraw.Draw(img)
    # patchwork fields
    for _ in range(40):
        x0, y0 = rng.randint(-100, W), rng.randint(-100, H)
        w, h = rng.randint(150, 500), rng.randint(150, 400)
        shade = rng.randint(70, 150)
        col = (shade, shade + rng.randint(0, 40), shade - rng.randint(0, 30))
        d.rectangle([x0, y0, x0 + w, y0 + h], fill=col)
    # a couple of "roads"
    for _ in range(3):
        y = rng.randint(0, H)
        d.line([0, y, W, y + rng.randint(-120, 120)], fill=(90, 90, 92), width=rng.randint(8, 20))
    # speckle
    for _ in range(4000):
        x, y = rng.randint(0, W - 1), rng.randint(0, H - 1)
        d.point((x, y), fill=(rng.randint(60, 200),) * 3)
    return img


def _try_geotag_exif(idx: int):
    try:
        import piexif
    except Exception:
        return None
    lat = BASE_LAT + (idx - N_IMAGES / 2) * 0.00015
    lon = BASE_LON + math.sin(idx) * 0.00012

    def deg_to_dms_rational(deg):
        deg = abs(deg)
        d = int(deg)
        m = int((deg - d) * 60)
        s = round((deg - d - m / 60) * 3600 * 100)
        return ((d, 1), (m, 1), (s, 100))

    gps = {
        piexif.GPSIFD.GPSLatitudeRef: "N" if lat >= 0 else "S",
        piexif.GPSIFD.GPSLatitude: deg_to_dms_rational(lat),
        piexif.GPSIFD.GPSLongitudeRef: "E" if lon >= 0 else "W",
        piexif.GPSIFD.GPSLongitude: deg_to_dms_rational(lon),
        piexif.GPSIFD.GPSAltitudeRef: 0,
        piexif.GPSIFD.GPSAltitude: (int(BASE_ALT * 100), 100),
    }
    return piexif.dump({"GPS": gps, "0th": {piexif.ImageIFD.Make: b"AeroSurvey",
                                            piexif.ImageIFD.Model: b"SampleCam"}})


def generate(out_dir: str) -> Tuple[List[str], str]:
    os.makedirs(out_dir, exist_ok=True)
    image_paths: List[str] = []
    rng = random.Random(99)
    for i in range(N_IMAGES):
        img = _background(seed=i)
        d = ImageDraw.Draw(img)
        # show a rotating subset of GCP targets per image (overlap)
        for (label, *_rest) in GCPS:
            if rng.random() < 0.7:
                x = rng.randint(120, W - 120)
                y = rng.randint(120, H - 120)
                _draw_target(d, x, y, label)
        d.text((20, 20), f"Sample flight — image {i + 1:02d}/{N_IMAGES}",
               fill=(255, 255, 255), font=_font(24))
        path = os.path.join(out_dir, f"IMG_{i + 1:03d}.jpg")
        exif = _try_geotag_exif(i)
        if exif:
            img.save(path, quality=88, exif=exif)
        else:
            img.save(path, quality=88)
        image_paths.append(path)

    csv_path = os.path.join(out_dir, "gcps.csv")
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("label,x,y,z,type\n")
        for (label, x, y, z) in GCPS:
            fh.write(f"{label},{x},{y},{z},control\n")
    return image_paths, csv_path
