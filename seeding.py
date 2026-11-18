"""
Filter seeds: an initial population built from a filter's feature ranking.

By default FASTENER starts from whatever `initial_genes` it is given. A seed
starts it instead from a filter's top-k nested subsets -- the best feature, the
best two, ..., the best k -- so the search begins on the filter's front rather
than having to find it. Two filters are offered:

* mRMR (MID variant): relevance I(f;y) minus mean redundancy to the features
  already chosen. It fits no model.
* Boruta, run on the optimizer's own model. Every Boruta iteration fits that
  model once, and those fits are charged to the run.
"""
import warnings
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from sklearn.feature_selection import mutual_info_classif

SEED_FILTERS = ("mrmr", "boruta")


def discretize(X: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Equal-width binning of every column into `n_bins` codes."""
    out = np.empty(X.shape, dtype=np.int32)
    for j in range(X.shape[1]):
        col = X[:, j]
        lo, hi = col.min(), col.max()
        if hi <= lo:
            out[:, j] = 0
            continue
        edges = np.linspace(lo, hi, n_bins + 1)[1:-1]
        out[:, j] = np.searchsorted(edges, col, side="right")
    return out


def _entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def rank_mrmr(X: np.ndarray, y: np.ndarray, n_select: int, n_bins: int = 10,
              random_state: Optional[int] = None) -> np.ndarray:
    """The first `n_select` features in mRMR (MID) order.

    Relevance is scikit-learn's `mutual_info_classif` on the raw features;
    redundancy is the mutual information between equal-width discretised
    features. Pairwise terms are computed only between candidates and the
    features already selected, so this is O(n_select * p), not O(p^2).
    """
    n_select = min(n_select, X.shape[1])
    codes = discretize(X, n_bins)
    relevance = mutual_info_classif(X, y, random_state=random_state)

    selected = [int(np.argmax(relevance))]
    remaining = set(range(X.shape[1])) - set(selected)

    def pair_mi(a: int, b: int) -> float:
        joint = np.bincount(codes[:, a] * n_bins + codes[:, b],
                            minlength=n_bins * n_bins)
        return (_entropy(np.bincount(codes[:, a], minlength=n_bins))
                + _entropy(np.bincount(codes[:, b], minlength=n_bins))
                - _entropy(joint))

    redundancy = np.zeros(X.shape[1])
    while len(selected) < n_select:
        last = selected[-1]
        cand = np.fromiter(remaining, dtype=int)
        for c in cand:
            redundancy[c] += pair_mi(int(c), last)
        score = relevance[cand] - redundancy[cand] / len(selected)
        nxt = int(cand[int(np.argmax(score))])
        selected.append(nxt)
        remaining.discard(nxt)
    return np.array(selected, dtype=int)


class _ModelForBoruta:
    """The optimizer's model, shaped the way BorutaPy needs.

    BorutaPy calls `set_params(n_estimators=..., random_state=...)` on its
    estimator every iteration, sizing a forest. This accepts `n_estimators`
    and ignores it, and passes `random_state` on to the model if it has one.
    The model must expose `feature_importances_` after fitting.
    """

    def __init__(self, model_factory: Callable[[], Any],
                 n_estimators: Any = 1, random_state: Any = None):
        self.model_factory = model_factory
        self.n_estimators = n_estimators
        self.random_state = random_state

    def get_params(self, deep: bool = True) -> dict:
        return {"model_factory": self.model_factory,
                "n_estimators": self.n_estimators,
                "random_state": self.random_state}

    def set_params(self, **params) -> "_ModelForBoruta":
        for key, value in params.items():
            if key not in ("model_factory", "n_estimators", "random_state"):
                raise ValueError(f"no parameter {key!r}")
            setattr(self, key, value)
        return self

    def fit(self, X, y) -> "_ModelForBoruta":
        model = self.model_factory()
        if self.random_state is not None and \
                "random_state" in getattr(model, "get_params", dict)():
            model.set_params(random_state=self.random_state)
        self.model_ = model.fit(X, y)
        self.feature_importances_ = self.model_.feature_importances_
        return self

    def predict(self, X):
        return self.model_.predict(X)


def rank_boruta(model_factory: Callable[[], Any], X: np.ndarray, y: np.ndarray,
                random_state: Optional[int] = None,
                max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray, int]:
    """Boruta on `model_factory()`: (feature order, confirmed set, fits).

    Needs the `Boruta` package (`pip install Boruta`). `fits` is the number of
    iterations Boruta ran, one model fit each.

    Boruta ranks every confirmed feature 1 and every tentative one 2, so the
    order breaks ties on mean shadow-relative importance, Boruta's own measure
    of strength; without that, a top-k prefix inside the confirmed block would
    be an arbitrary pick from it.
    """
    try:
        from boruta import BorutaPy
    except ImportError as e:
        raise ImportError("the Boruta seed needs the Boruta package: "
                          "pip install Boruta") from e

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = BorutaPy(_ModelForBoruta(model_factory), n_estimators="auto",
                       random_state=random_state, max_iter=max_iter, verbose=0)
        sel.fit(X, y)
        # Row 0 of the history is BorutaPy's zero-filled starting row.
        imp = np.nanmean(sel.importance_history_[1:], axis=0)
    imp = np.nan_to_num(imp, nan=0.0)
    order = np.lexsort((-imp, sel.ranking_))  # rank asc, then importance desc
    confirmed = np.where(sel.support_)[0]
    fits = int(len(sel.importance_history_) - 1)
    return order, confirmed, fits


def nested_prefixes(order: np.ndarray, k: int) -> List[List[int]]:
    """[[f1], [f1, f2], ..., [f1, ..., fk]] from a feature order."""
    order = [int(i) for i in order[:k]]
    return [order[:n] for n in range(1, len(order) + 1)]
