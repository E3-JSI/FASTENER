"""Mutual information between continuous features and a discrete target,
as `sklearn.feature_selection.mutual_info_classif` estimates it, computed by
sorting each feature instead of fitting nearest-neighbour models per feature.

The estimator is Ross (2014), as in scikit-learn: the data are scaled and
jittered exactly as there (the same random draws, from the same generator),
then for each sample the distance to its k-th nearest neighbour of the same
class, and the number of samples of any class within that distance. In one
dimension both come from the sorted feature: the k-th neighbour lies within k
places either side, and the count is a range found by binary search. On wide
data this is 10-50 times faster than fitting k-NN models per feature and per
class, which is where scikit-learn's time goes.

The arithmetic is scikit-learn's, so the values are identical, not just
close. That matters: on data with repeated values (nci9) the jitter is far
below rounding error, and which neighbour counts as k-th is decided by how a
distance was rounded. scikit-learn picks a brute-force search for a class
with k >= m // 2 (m its size) and computes those distances as
sqrt(max(0, (a^2 - 2ab) + b^2)); otherwise a k-d tree, with sqrt((a - b)^2).
The radius count compares (a - b)^2 with r^2, as the k-d tree does.
`tests/test_fast_mi.py` checks equality against scikit-learn.
"""
from typing import Optional

import numpy as np
from scipy.special import digamma
from sklearn.preprocessing import scale
from sklearn.utils import check_random_state, check_X_y
from sklearn.utils.multiclass import check_classification_targets


def _kth_neighbour_brute(v: np.ndarray, k: int) -> np.ndarray:
    """Each point's distance to its k-th nearest other point, as scikit-learn's
    brute-force `kneighbors()` computes it: k + 1 neighbours by the expanded
    squared distance, then the point itself dropped (or, if rounding pushed it
    out of the k + 1, the first)."""
    m = v.shape[0]
    xx = v * v
    d2 = (xx[:, None] + (-2.0 * v)[:, None] * v[None, :]) + xx[None, :]
    d = np.sqrt(np.maximum(d2, 0.0))
    o = np.argsort(d, axis=1, kind="stable")[:, :k + 1]
    rows = np.arange(m)
    col = np.where(o[:, k] == rows, o[:, k - 1], o[:, k])
    return d[rows, col]


def _kth_neighbour_distance(v: np.ndarray, k: int) -> np.ndarray:
    """For sorted `v`, each point's distance to its k-th nearest other point,
    as scikit-learn's k-d tree computes it."""
    n = v.shape[0]
    idx = np.arange(n)[:, None] + np.concatenate(
        [np.arange(-k, 0), np.arange(1, k + 1)])[None, :]
    valid = (idx >= 0) & (idx < n)
    diff = v[np.clip(idx, 0, n - 1)] - v[:, None]
    d2 = diff * diff
    d2[~valid] = np.inf
    return np.sqrt(np.partition(d2, k - 1, axis=1)[:, k - 1])


def _count_within(s: np.ndarray, c: np.ndarray, r: np.ndarray) -> np.ndarray:
    """For sorted `s`, how many s_j satisfy (s_j - c_i)^2 <= r_i^2, per i.

    `searchsorted` on c +- r finds the range up to rounding; the bounds are
    then moved one place at a time until the test holds just inside them and
    fails just outside. fl((s_j - c_i)^2) is monotone in |s_j - c_i|, so the
    qualifying j form one contiguous range.
    """
    n = s.shape[0]
    r2 = r * r
    lo = np.searchsorted(s, c - r, side="left")
    hi = np.searchsorted(s, c + r, side="right")   # one past the last

    def inside(j):
        jj = np.clip(j, 0, n - 1)
        diff = s[jj] - c
        return (j >= 0) & (j < n) & (diff * diff <= r2)

    for _ in range(n):
        grow = inside(lo - 1)
        shrink = ~inside(lo) & (lo < hi)
        if not (grow.any() or shrink.any()):
            break
        lo = lo - grow + shrink
    for _ in range(n):
        grow = inside(hi)
        shrink = ~inside(hi - 1) & (hi > lo)
        if not (grow.any() or shrink.any()):
            break
        hi = hi + grow - shrink
    return hi - lo


def _mi_cd(c: np.ndarray, d: np.ndarray, n_neighbors: int) -> float:
    """scikit-learn's `_compute_mi_cd` for one feature, in one dimension."""
    n_samples = c.shape[0]
    radius = np.empty(n_samples)
    label_counts = np.empty(n_samples)
    k_all = np.empty(n_samples)
    for label in np.unique(d):
        mask = d == label
        count = int(mask.sum())
        if count > 1:
            k = min(n_neighbors, count - 1)
            where = np.flatnonzero(mask)
            if k >= count // 2:
                r = _kth_neighbour_brute(c[where], k)
            else:
                order = np.argsort(c[where], kind="stable")
                r = np.empty(count)
                r[order] = _kth_neighbour_distance(c[where][order], k)
            radius[where] = np.nextafter(r, 0)
            k_all[where] = k
        label_counts[mask] = count

    mask = label_counts > 1
    n_samples = int(mask.sum())
    label_counts = label_counts[mask]
    k_all = k_all[mask]
    c = c[mask]
    radius = radius[mask]

    m_all = _count_within(np.sort(c), c, radius)
    mi = (digamma(n_samples) + np.mean(digamma(k_all))
          - np.mean(digamma(label_counts)) - np.mean(digamma(m_all)))
    return max(0.0, mi)


def mutual_info_classif(X, y, *, n_neighbors: int = 3,
                        random_state: Optional[int] = None) -> np.ndarray:
    """Drop-in for scikit-learn's `mutual_info_classif` on dense, continuous X.

    Consumes the random generator exactly as scikit-learn does (one
    `standard_normal` draw of X's shape), so the generator is left in the
    same state and everything drawn after it is unchanged.
    """
    # scikit-learn's preprocessing line for line (`_estimate_mi`): the column
    # mask and the copies it makes decide the order `np.mean` sums in, and so
    # the last bit of the jitter -- which decides ties on discrete data.
    check_classification_targets(y)
    X, y = check_X_y(X, y, y_numeric=False)
    n_samples, n_features = X.shape
    continuous_mask = np.ones(n_features, dtype=bool)
    rng = check_random_state(random_state)
    X = X.astype(np.float64, copy=True)
    X[:, continuous_mask] = scale(X[:, continuous_mask], with_mean=False,
                                  copy=False)
    means = np.maximum(1, np.mean(np.abs(X[:, continuous_mask]), axis=0))
    X[:, continuous_mask] += (
        1e-10 * means
        * rng.standard_normal(size=(n_samples, np.sum(continuous_mask))))
    return np.array([_mi_cd(X[:, j], y, n_neighbors)
                     for j in range(n_features)])
