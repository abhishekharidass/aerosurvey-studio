"""Textured-mesh helpers for the mesh stage.

The OpenMVS mesh lives in the COLMAP local frame; `transform_obj` streams it
into the project CRS. Because UTM coordinates overflow the float32 mantissa
used by most mesh viewers, vertices are shifted by a round local offset that
is recorded in the OBJ header and a sidecar file (the Pix4D/Agisoft
convention). The simulation fallback triangulates a coloured heightfield
straight from the dense cloud.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np


def auto_offset(pt) -> Tuple[float, float, float]:
    """A round-100 m offset near a point, so shifted coords stay small."""
    return (float(np.floor(pt[0] / 100.0) * 100.0),
            float(np.floor(pt[1] / 100.0) * 100.0),
            0.0)


def ply_face_count(ply_path: str) -> int:
    """Face count from a PLY header, without loading the geometry."""
    with open(ply_path, "rb") as fh:
        for _ in range(200):
            line = fh.readline()
            if not line or line.strip() == b"end_header":
                break
            if line.startswith(b"element face"):
                return int(line.split()[-1])
    return 0


def peek_first_vertex(obj_path: str) -> Optional[np.ndarray]:
    """First `v` position in an OBJ, without loading the file."""
    with open(obj_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("v "):
                p = line.split()
                return np.array([float(p[1]), float(p[2]), float(p[3])])
    return None


def transform_obj(src: str, dst: str, sim=None,
                  offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                  epsg: Optional[int] = None) -> Tuple[int, int]:
    """Rewrite an OBJ applying an optional Similarity and a coordinate offset.

    Vertex positions get sim then -offset; vertex normals get rotation only.
    Everything else (uv, faces, mtllib) streams through untouched.
    Returns (n_vertices, n_faces).
    """
    off = np.asarray(offset, np.float64)
    nv = nf = 0
    with open(src, "r", encoding="utf-8", errors="replace") as fin, \
            open(dst, "w", encoding="utf-8") as fout:
        fout.write("# AeroSurvey Studio textured mesh\n")
        fout.write(f"# coordinate offset (add to vertices): "
                   f"{off[0]:.3f} {off[1]:.3f} {off[2]:.3f}\n")
        if epsg:
            fout.write(f"# crs: EPSG:{epsg}\n")
        for line in fin:
            if line.startswith("v "):
                parts = line.split()
                p = np.array([float(parts[1]), float(parts[2]),
                              float(parts[3])], np.float64)
                if sim is not None:
                    p = sim.apply(p)
                p -= off
                rest = " " + " ".join(parts[4:]) if len(parts) > 4 else ""
                fout.write(f"v {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}{rest}\n")
                nv += 1
            elif line.startswith("vn ") and sim is not None:
                parts = line.split()
                n = np.array([float(parts[1]), float(parts[2]),
                              float(parts[3])], np.float64)
                n = sim.R @ n
                n /= max(np.linalg.norm(n), 1e-12)
                fout.write(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")
            else:
                if line.startswith("f "):
                    nf += 1
                fout.write(line)
    _write_offset_sidecar(dst, off, epsg)
    return nv, nf


def _write_offset_sidecar(obj_path: str, off: np.ndarray,
                          epsg: Optional[int]) -> None:
    with open(os.path.splitext(obj_path)[0] + "_offset.txt", "w",
              encoding="utf-8") as fh:
        fh.write(f"{off[0]:.3f} {off[1]:.3f} {off[2]:.3f}\n")
        if epsg:
            fh.write(f"EPSG:{epsg}\n")


def heightfield_mesh(P: np.ndarray, C: np.ndarray, cell: float
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triangulate a point cloud as a regular heightfield.

    Returns (verts (M,3), colors (M,3) uint8, faces (K,3) 0-based int).
    Cells without points are left out; only fully valid quads triangulate.
    """
    x0, y0 = P[:, 0].min(), P[:, 1].min()
    ix = ((P[:, 0] - x0) / cell).astype(int)
    iy = ((P[:, 1] - y0) / cell).astype(int)
    nx, ny = ix.max() + 1, iy.max() + 1
    zgrid = np.full((ny, nx), np.nan)
    cgrid = np.zeros((ny, nx, 3), np.float64)
    cnt = np.zeros((ny, nx), np.int64)
    # mean z / colour per cell (accumulate then divide)
    zsum = np.zeros((ny, nx))
    np.add.at(zsum, (iy, ix), P[:, 2])
    np.add.at(cgrid, (iy, ix), C.astype(np.float64))
    np.add.at(cnt, (iy, ix), 1)
    valid = cnt > 0
    zgrid[valid] = zsum[valid] / cnt[valid]
    zgrid[~valid] = np.nan
    cgrid[valid] /= cnt[valid][:, None]

    vid = np.full((ny, nx), -1, np.int64)
    vy, vx = np.nonzero(valid)
    vid[vy, vx] = np.arange(len(vy))
    verts = np.column_stack([x0 + (vx + 0.5) * cell,
                             y0 + (vy + 0.5) * cell,
                             zgrid[vy, vx]])
    colors = cgrid[vy, vx].clip(0, 255).astype(np.uint8)

    quad = (valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:])
    qy, qx = np.nonzero(quad)
    a = vid[qy, qx]
    b = vid[qy, qx + 1]
    c = vid[qy + 1, qx]
    d = vid[qy + 1, qx + 1]
    faces = np.concatenate([np.column_stack([a, b, c]),
                            np.column_stack([b, d, c])]) if len(qy) else \
        np.zeros((0, 3), np.int64)
    return verts, colors, faces


def write_obj_with_colors(path: str, verts: np.ndarray, colors: np.ndarray,
                          faces: np.ndarray,
                          offset: Tuple[float, float, float] = (0, 0, 0),
                          epsg: Optional[int] = None) -> None:
    """OBJ with per-vertex colours (common viewer extension: v x y z r g b)."""
    off = np.asarray(offset, np.float64)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# AeroSurvey Studio mesh (per-vertex colours)\n")
        fh.write(f"# coordinate offset (add to vertices): "
                 f"{off[0]:.3f} {off[1]:.3f} {off[2]:.3f}\n")
        if epsg:
            fh.write(f"# crs: EPSG:{epsg}\n")
        for (x, y, z), (r, g, b) in zip(verts - off, colors / 255.0):
            fh.write(f"v {x:.4f} {y:.4f} {z:.4f} {r:.4f} {g:.4f} {b:.4f}\n")
        for a, b_, c in faces + 1:      # OBJ is 1-based
            fh.write(f"f {a} {b_} {c}\n")
    _write_offset_sidecar(path, off, epsg)
