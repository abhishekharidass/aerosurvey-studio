"""Read EXIF geotags and DJI/XMP gimbal attitude from drone photos."""
from __future__ import annotations

import re
from typing import Optional

from PIL import Image, ExifTags

_GPS_TAGS = {v: k for k, v in ExifTags.GPSTAGS.items()}  # name -> id (unused, ref only)
_TAGS = ExifTags.TAGS
_GPSTAGS = ExifTags.GPSTAGS


def _ratio(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _dms_to_deg(dms, ref) -> Optional[float]:
    try:
        d = _ratio(dms[0]) or 0.0
        m = _ratio(dms[1]) or 0.0
        s = _ratio(dms[2]) or 0.0
        deg = d + m / 60.0 + s / 3600.0
        if ref in ("S", "W"):
            deg = -deg
        return deg
    except Exception:
        return None


# XMP attitude keys drones commonly write (DJI, Autel, Parrot).
_XMP_PATTERNS = {
    "yaw": re.compile(rb'(?:GimbalYawDegree|FlightYawDegree|Yaw)\s*=\s*"?\s*(-?\d+\.?\d*)'),
    "pitch": re.compile(rb'(?:GimbalPitchDegree|FlightPitchDegree|Pitch)\s*=\s*"?\s*(-?\d+\.?\d*)'),
    "roll": re.compile(rb'(?:GimbalRollDegree|FlightRollDegree|Roll)\s*=\s*"?\s*(-?\d+\.?\d*)'),
    "rel_alt": re.compile(rb'RelativeAltitude\s*=\s*"?\s*(\+?-?\d+\.?\d*)'),
}


def read_xmp_attitude(path: str) -> dict:
    """Scan the first chunk of the file for an XMP packet and pull attitude."""
    out: dict = {}
    try:
        with open(path, "rb") as fh:
            blob = fh.read(128 * 1024)  # XMP lives in the header region
        start = blob.find(b"<x:xmpmeta")
        if start == -1:
            return out
        end = blob.find(b"</x:xmpmeta>", start)
        packet = blob[start: end + 12] if end != -1 else blob[start:]
        for key, pat in _XMP_PATTERNS.items():
            m = pat.search(packet)
            if m:
                try:
                    out[key] = float(m.group(1))
                except ValueError:
                    pass
    except OSError:
        pass
    return out


def read_metadata(path: str) -> dict:
    """Return a flat dict of the metadata we care about for one photo."""
    meta: dict = {"width": 0, "height": 0}
    try:
        with Image.open(path) as img:
            meta["width"], meta["height"] = img.size
            exif = img.getexif()
    except Exception:
        return meta

    if not exif:
        meta.update(read_xmp_attitude(path))
        return meta

    named = {_TAGS.get(k, k): v for k, v in exif.items()}
    # FocalLength / DateTimeOriginal live in the Exif SubIFD, not IFD0.
    try:
        sub = exif.get_ifd(ExifTags.IFD.Exif)
        named.update({_TAGS.get(k, k): v for k, v in sub.items()})
    except Exception:
        pass
    meta["make"] = str(named.get("Make", "")).strip("\x00 ")
    meta["model"] = str(named.get("Model", "")).strip("\x00 ")
    meta["datetime"] = str(named.get("DateTimeOriginal", named.get("DateTime", "")))
    fl = _ratio(named.get("FocalLength"))
    if fl:
        meta["focal_mm"] = fl
    f35 = _ratio(named.get("FocalLengthIn35mmFilm"))
    if f35:
        meta["focal35_mm"] = f35

    # GPS block lives under IFD
    try:
        gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
    except Exception:
        gps_ifd = {}
    if gps_ifd:
        g = {_GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}
        lat = _dms_to_deg(g.get("GPSLatitude"), g.get("GPSLatitudeRef"))
        lon = _dms_to_deg(g.get("GPSLongitude"), g.get("GPSLongitudeRef"))
        alt = _ratio(g.get("GPSAltitude"))
        if alt is not None and g.get("GPSAltitudeRef") in (1, b"\x01"):
            alt = -alt
        if lat is not None:
            meta["lat"] = lat
        if lon is not None:
            meta["lon"] = lon
        if alt is not None:
            meta["alt"] = alt

    meta.update(read_xmp_attitude(path))
    return meta
