"""Supervised machine-learning point-cloud classification.

A Random Forest trained on per-point geometric + radiometric features. This is
the learned alternative to the rule-based morphological/PCA classifier: instead
of hand-tuned thresholds, the model *learns* the ground / vegetation / building
decision boundary from labelled data.

Features (per point, from a k-NN neighbourhood — the standard Weinmann-style set):
  height-above-ground, planarity, linearity, sphericity, surface-variation,
  verticality, local height range, greenness, brightness.

Vectorised with NumPy + scipy.cKDTree so it scales to millions of points.
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np

FEATURE_NAMES = [
    "height_above_ground", "planarity", "linearity", "sphericity",
    "surface_variation", "verticality", "height_range", "greenness", "brightness",
]

CLASS_LABELS = {2: "Ground", 5: "Vegetation", 6: "Building"}
DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "pointcloud_rf.joblib")


def _height_above_ground(P: np.ndarray, cell: float = 3.0) -> np.ndarray:
    minx, miny = float(P[:, 0].min()), float(P[:, 1].min())
    maxx, maxy = float(P[:, 0].max()), float(P[:, 1].max())
    nx = max(int(np.ceil((maxx - minx) / cell)) + 1, 1)
    ny = max(int(np.ceil((maxy - miny) / cell)) + 1, 1)
    ix = np.clip(((P[:, 0] - minx) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((P[:, 1] - miny) / cell).astype(int), 0, ny - 1)
    key = iy * nx + ix
    ground = np.full(nx * ny, np.inf)
    np.minimum.at(ground, key, P[:, 2])
    return P[:, 2] - ground[key]


def compute_features(P: np.ndarray, C: np.ndarray, k: int = 15) -> np.ndarray:
    """Return an (N, len(FEATURE_NAMES)) feature matrix."""
    from scipy.spatial import cKDTree
    P = np.asarray(P, np.float64)
    C = np.asarray(C, np.float64)
    n = len(P)
    k = min(k, n)

    hag = _height_above_ground(P)

    tree = cKDTree(P)
    _, idx = tree.query(P, k=k)
    nb = P[idx]                                   # (N, k, 3)
    centered = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / k
    evals, evecs = np.linalg.eigh(cov)            # ascending: l3 <= l2 <= l1
    l3, l2, l1 = evals[:, 0], evals[:, 1], evals[:, 2]
    l1 = np.maximum(l1, 1e-9)
    ssum = np.maximum(l1 + l2 + l3, 1e-9)

    planarity = (l2 - l3) / l1
    linearity = (l1 - l2) / l1
    sphericity = l3 / l1
    surf_var = l3 / ssum
    normal_z = np.abs(evecs[:, 2, 0])             # z-component of smallest-eigenvalue vector
    verticality = 1.0 - normal_z
    hrange = nb[:, :, 2].max(axis=1) - nb[:, :, 2].min(axis=1)

    csum = C.sum(axis=1) + 1e-6
    greenness = (2 * C[:, 1] - C[:, 0] - C[:, 2]) / csum
    brightness = csum / 3.0 / 255.0

    return np.column_stack([hag, planarity, linearity, sphericity,
                            surf_var, verticality, hrange, greenness, brightness])


def train(X: np.ndarray, y: np.ndarray, test_size: float = 0.3, seed: int = 42):
    """Train a Random Forest; return (model, metrics dict)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size,
                                          random_state=seed, stratify=y)
    clf = RandomForestClassifier(n_estimators=150, max_depth=14, min_samples_leaf=12,
                                 n_jobs=-1, class_weight="balanced", random_state=seed)
    clf.fit(Xtr, ytr)
    yp = clf.predict(Xte)
    classes = sorted(np.unique(y).tolist())
    metrics = {
        "test_accuracy": float(accuracy_score(yte, yp)),
        "cv_accuracy": float(cross_val_score(clf, X, y, cv=3, n_jobs=-1).mean()),
        "report": classification_report(yte, yp,
                                        target_names=[CLASS_LABELS.get(c, str(c)) for c in classes]),
        "confusion": confusion_matrix(yte, yp, labels=classes),
        "classes": classes,
        "importances": dict(zip(FEATURE_NAMES, clf.feature_importances_.tolist())),
        "n_train": len(Xtr), "n_test": len(Xte),
    }
    return clf, metrics


def save(model, path: str = DEFAULT_MODEL) -> None:
    import joblib
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load(path: str = DEFAULT_MODEL):
    import joblib
    return joblib.load(path)


def available(path: str = DEFAULT_MODEL) -> bool:
    return os.path.exists(path)


def classify(P: np.ndarray, C: np.ndarray, model=None) -> np.ndarray:
    """Predict ASPRS classes for a cloud using a trained model."""
    if model is None:
        model = load()
    X = compute_features(P, C)
    return model.predict(X).astype(np.uint8)
