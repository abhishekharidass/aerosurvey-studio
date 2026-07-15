"""Tests for the textured-mesh helpers (OBJ georeferencing + heightfield)."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aerosurvey.pipeline import mesh as meshmod
from aerosurvey.pipeline.georef import Similarity


@pytest.fixture
def obj_file(tmp_path):
    p = tmp_path / "in.obj"
    p.write_text(
        "mtllib scene.mtl\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "v 0 0 1 0.5 0.5 0.5\n"     # trailing vertex colour must survive
        "vn 1 0 0\n"
        "vt 0.5 0.5\n"
        "f 1/1/1 2/1/1 3/1/1\n")
    return str(p)


def _rot_z(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0],
                     [math.sin(a), math.cos(a), 0],
                     [0, 0, 1]])


def test_transform_obj_applies_similarity_and_offset(obj_file, tmp_path):
    # 90 deg about z, scale 2, translate to UTM-ish coords
    sim = Similarity(2.0, _rot_z(90), np.array([500000.0, 2500000.0, 10.0]))
    out = str(tmp_path / "out.obj")
    nv, nf = meshmod.transform_obj(obj_file, out, sim,
                                   offset=(500000, 2500000, 0), epsg=32645)
    assert (nv, nf) == (3, 1)
    lines = open(out).read().splitlines()
    vs = [l for l in lines if l.startswith("v ")]
    # v1 (1,0,0): rot90 -> (0,2,0), +t, -offset -> (0, 2, 10)
    x, y, z = map(float, vs[0].split()[1:4])
    assert (x, y, z) == pytest.approx((0.0, 2.0, 10.0), abs=1e-3)
    # vertex colour suffix preserved
    assert vs[2].split()[4:] == ["0.5", "0.5", "0.5"]
    # normal rotated but not scaled/translated: (1,0,0) -> (0,1,0)
    vn = [l for l in lines if l.startswith("vn ")][0]
    assert list(map(float, vn.split()[1:])) == pytest.approx([0, 1, 0], abs=1e-6)
    # faces / uv / mtllib untouched
    assert "f 1/1/1 2/1/1 3/1/1" in lines
    assert "vt 0.5 0.5" in lines
    assert "mtllib scene.mtl" in lines
    # header + sidecar carry the offset and CRS
    head = "\n".join(lines[:4])
    assert "500000.000 2500000.000" in head and "EPSG:32645" in head
    side = open(str(tmp_path / "out_offset.txt")).read()
    assert "500000.000" in side and "EPSG:32645" in side


def test_transform_obj_identity_without_sim(obj_file, tmp_path):
    out = str(tmp_path / "out.obj")
    nv, _ = meshmod.transform_obj(obj_file, out)
    assert nv == 3
    vs = [l for l in open(out) if l.startswith("v ")]
    assert list(map(float, vs[0].split()[1:4])) == pytest.approx([1, 0, 0])


def test_auto_offset_rounds_down_to_100():
    off = meshmod.auto_offset(np.array([500123.4, 2500987.6, 42.0]))
    assert off == (500100.0, 2500900.0, 0.0)


def test_heightfield_mesh_plane():
    # a flat 10x10 m plane sampled on a 0.5 m lattice, z = 5
    xs, ys = np.meshgrid(np.arange(0, 10, 0.5), np.arange(0, 10, 0.5))
    P = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, 5.0)])
    C = np.full((len(P), 3), 128, np.uint8)
    verts, colors, faces = meshmod.heightfield_mesh(P, C, cell=1.0)
    assert len(verts) == 100          # 10x10 occupied cells
    assert len(faces) == 81 * 2       # two triangles per interior quad
    assert verts[:, 2] == pytest.approx(np.full(len(verts), 5.0))
    assert (colors == 128).all()
    # every face references valid vertex ids
    assert faces.min() >= 0 and faces.max() < len(verts)


def test_heightfield_mesh_skips_empty_cells():
    # two far-apart clusters: no faces should bridge the gap
    a = np.column_stack([np.random.rand(50) * 2, np.random.rand(50) * 2,
                         np.zeros(50)])
    b = a + np.array([50.0, 0, 0])
    P = np.vstack([a, b])
    C = np.full((len(P), 3), 200, np.uint8)
    verts, _, faces = meshmod.heightfield_mesh(P, C, cell=1.0)
    for f in faces:
        xs = verts[f, 0]
        assert xs.max() - xs.min() < 5.0     # no triangle spans the void


def test_write_obj_with_colors(tmp_path):
    verts = np.array([[500000.0, 2500000.0, 1.0],
                      [500001.0, 2500000.0, 1.0],
                      [500000.0, 2500001.0, 2.0]])
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], np.uint8)
    faces = np.array([[0, 1, 2]])
    out = str(tmp_path / "m.obj")
    meshmod.write_obj_with_colors(out, verts, colors, faces,
                                  offset=(500000, 2500000, 0), epsg=32645)
    lines = open(out).read().splitlines()
    vs = [l for l in lines if l.startswith("v ")]
    assert len(vs) == 3
    assert list(map(float, vs[0].split()[1:4])) == pytest.approx([0, 0, 1])
    assert list(map(float, vs[0].split()[4:])) == pytest.approx([1, 0, 0])
    assert "f 1 2 3" in lines
    assert os.path.exists(str(tmp_path / "m_offset.txt"))


def test_peek_first_vertex(obj_file):
    v = meshmod.peek_first_vertex(obj_file)
    assert v == pytest.approx([1.0, 0.0, 0.0])


def test_ply_face_count(tmp_path):
    p = tmp_path / "m.ply"
    p.write_bytes(b"ply\nformat binary_little_endian 1.0\n"
                  b"element vertex 8911444\nproperty float x\n"
                  b"element face 17821866\n"
                  b"property list uchar int vertex_indices\nend_header\n")
    assert meshmod.ply_face_count(str(p)) == 17821866
    q = tmp_path / "noface.ply"
    q.write_bytes(b"ply\nformat ascii 1.0\nend_header\n")
    assert meshmod.ply_face_count(str(q)) == 0
