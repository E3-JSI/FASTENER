"""Shared fixtures for FASTENER's unit tests.

FASTENER uses flat top-level imports (`from item import ...`), so the repo root
has to be on `sys.path` before any of its modules can be imported.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from item import EvalItem, Item, Result  # noqa: E402


N_FEATURES = 8


def genes_from(indices, n_features=N_FEATURES):
    genes = [False] * n_features
    for i in indices:
        genes[i] = True
    return genes


def item_from(indices, n_features=N_FEATURES, generation=0):
    return Item(genes_from(indices, n_features), generation, None, None)


def eval_item(indices, score, n_features=N_FEATURES, generation=0):
    return EvalItem(genes_from(indices, n_features), generation, None, None,
                    Result(score))


def selected(genes):
    return {i for i, g in enumerate(genes) if g}


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """Config writes to "log/<name>" relative to the cwd."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(scope="session")
def small_data():
    """A small classification problem where only features 0 and 1 matter.

    Returns (X_train, y_train, X_val, y_val). Kept tiny so that
    `mutual_info_classif` and a full mainloop run in well under a second.
    """
    rng = np.random.RandomState(0)
    X = rng.normal(size=(300, N_FEATURES))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X[:200], y[:200], X[200:], y[200:]
