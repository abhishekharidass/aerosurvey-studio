"""Application-wide constants and metadata."""
from __future__ import annotations

APP_NAME = "AeroSurvey Studio"
APP_VERSION = "0.1.0"
ORG_NAME = "AeroSurvey"
PROJECT_EXT = ".asproj"

# Supported image extensions for photo import.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".tif", ".tiff", ".png")

# Point-cloud classification classes (ASPRS-style subset).
CLASS_NAMES = {
    0: "Never classified",
    1: "Unclassified",
    2: "Ground",
    3: "Low vegetation",
    4: "Medium vegetation",
    5: "High vegetation",
    6: "Building",
    9: "Water",
    17: "Bridge deck",
}

CLASS_COLORS = {
    0: (170, 170, 170),
    1: (200, 200, 200),
    2: (160, 110, 60),
    3: (120, 200, 120),
    4: (70, 170, 70),
    5: (30, 130, 30),
    6: (210, 120, 90),
    9: (70, 140, 220),
    17: (200, 200, 90),
}
