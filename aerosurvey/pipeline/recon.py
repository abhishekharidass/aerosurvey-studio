"""Reload previously computed engine results from the work directory.

The align/georef stages keep their results on the chunk as transient
attributes (``_colmap_result``, ``_georef_sim``) for the duration of a run.
This module persists / reconstitutes them so later stages (dense, ortho,
GCP auto-marking) also work when run in a fresh session:

  * the COLMAP sparse model lives in   <workdir>/colmap/sparse/<n>/
  * the georeferencing similarity is saved to <workdir>/georef_sim.json
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

from . import colmap
from .georef import Similarity

SIM_FILE = "georef_sim.json"


# -- georef similarity -------------------------------------------------------
def save_sim(workdir: str, sim: Similarity, method: str = "") -> None:
    data = {"s": float(sim.s),
            "R": np.asarray(sim.R, dtype=float).ravel().tolist(),
            "t": np.asarray(sim.t, dtype=float).ravel().tolist(),
            "method": method}
    with open(os.path.join(workdir, SIM_FILE), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)


def load_sim(workdir: str) -> Optional[Similarity]:
    path = os.path.join(workdir, SIM_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return Similarity(float(d["s"]),
                          np.array(d["R"], dtype=float).reshape(3, 3),
                          np.array(d["t"], dtype=float))
    except Exception:
        return None


def get_sim(chunk, workdir: str) -> Optional[Similarity]:
    """In-memory georef transform if present, else the persisted one."""
    sim = getattr(chunk, "_georef_sim", None)
    return sim if sim is not None else load_sim(workdir)


# -- COLMAP reconstruction ---------------------------------------------------
def load_reconstruction(workdir: str, ctx=None) -> Optional[colmap.ColmapResult]:
    """Re-read the largest COLMAP model under workdir/colmap/sparse."""
    sparse = os.path.join(workdir, "colmap", "sparse")
    model = colmap._largest_model(sparse)
    if model is None:
        return None
    if not os.path.exists(os.path.join(model, "images.txt")):
        if not colmap.available():
            return None
        args = ["model_converter", "--input_path", model,
                "--output_path", model, "--output_type", "TXT"]
        if ctx is not None:
            if colmap._run_step("model_converter", args, ctx) is not True:
                return None
        else:
            import subprocess
            r = subprocess.run([colmap.exe()] + args, capture_output=True)
            if r.returncode != 0:
                return None
    res = colmap.read_model_dir(model)
    res.model_dir = model
    img_dir = os.path.join(workdir, "colmap", "images")
    res.image_dir = img_dir if os.path.isdir(img_dir) else ""
    return res if res.poses else None


def get_reconstruction(chunk, workdir: str, ctx=None) -> Optional[colmap.ColmapResult]:
    """In-memory COLMAP result if present, else reconstitute from disk."""
    res = getattr(chunk, "_colmap_result", None)
    if res is not None:
        return res
    res = load_reconstruction(workdir, ctx)
    if res is not None:
        chunk._colmap_result = res  # cache for the rest of the session
    return res
