"""Train the Random Forest point-cloud classifier on labelled synthetic scenes.

Run:  python tools/train_classifier.py
Saves the model to aerosurvey/models/pointcloud_rf.joblib and prints
train/test accuracy, a classification report, the confusion matrix and
feature importances.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aerosurvey.pipeline.stages import _synth_scene
from aerosurvey.pipeline import ml_classify as ml


def build_dataset(seeds):
    """Generate labelled scenes with realistic noise so classes overlap the way
    real Multi-View-Stereo clouds do (otherwise the problem is trivially separable)."""
    Xs, ys = [], []
    for s in seeds:
        P, C, cls = _synth_scene(seed=s)
        rng = np.random.default_rng(s)
        P = P + rng.normal(0, 0.06, P.shape)                       # positional MVS noise
        P[:, 2] += rng.normal(0, 0.20, len(P))                     # vertical noise
        C = np.clip(C.astype(float) + rng.normal(0, 28, C.shape), 0, 255)  # colour scatter
        Xs.append(ml.compute_features(P, C))
        ys.append(cls)
        print(f"  scene seed {s:>3}: {len(P):>7,} points")
    return np.vstack(Xs), np.concatenate(ys)


def main():
    seeds = [7, 11, 23, 42, 99, 128]
    print("Generating labelled training scenes (ground truth = synthetic labels)...")
    X, y = build_dataset(seeds)
    print(f"Total: {len(X):,} points x {X.shape[1]} features\n")

    model, m = ml.train(X, y)
    print(f"Test accuracy   : {m['test_accuracy']:.4f}   (held-out {m['n_test']:,} pts)")
    print(f"3-fold CV score : {m['cv_accuracy']:.4f}\n")
    print("Classification report:")
    print(m["report"])
    print("Confusion matrix  (rows = true, cols = predicted)  classes =", m["classes"])
    print(m["confusion"], "\n")
    print("Feature importances:")
    for name, imp in sorted(m["importances"].items(), key=lambda kv: -kv[1]):
        print(f"  {name:20s} {imp:.4f}")

    ml.save(model)
    size_mb = os.path.getsize(ml.DEFAULT_MODEL) / 1e6
    print(f"\nModel saved: {ml.DEFAULT_MODEL}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
