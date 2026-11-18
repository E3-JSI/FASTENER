"""Helpers shared by the examples.

FASTENER uses flat top-level imports (`from item import ...`), so its
directory must be on `sys.path`. Importing this module puts it there.
"""
import sys
import time
from pathlib import Path

import numpy as np

FASTENER_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FASTENER_DIR))


def run_folder(name: str) -> str:
    """A fresh output folder name. `Config` writes to log/<name> and refuses
    to reuse a folder, so each run gets a timestamped one."""
    return f"{name}_{time.strftime('%Y%m%d-%H%M%S')}_{time.time_ns() % 10**6}"


def front_table(optimizer, feature_names=None) -> str:
    """The Pareto front, one line per subset size."""
    lines = []
    for size, item in sorted(optimizer.pareto_front.items()):
        idx = np.flatnonzero(item.genes)
        names = [feature_names[i] for i in idx] if feature_names is not None \
            else [int(i) for i in idx]
        lines.append(f"  {size:3d} features  score {item.result.score:.4f}  "
                     f"{names}")
    return "\n".join(lines)


def classification_data(seed: int = 0):
    """Breast cancer (569 x 30), split 50/25/25 into train / validation /
    test. Returns (X_train, y_train, X_val, y_val, X_test, y_test, names)."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=1 / 3, stratify=y_rest, random_state=seed)
    return (X_train, y_train, X_val, y_val, X_test, y_test,
            [str(n) for n in data.feature_names])

