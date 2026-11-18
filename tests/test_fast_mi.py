"""fast_mi gives scikit-learn's mutual information, value for value.

Equality, not closeness: the mating strategy sorts and samples genes by these
values, so a difference in the last bit can change the search. The cases cover
both of scikit-learn's neighbour searches (brute force for classes with
k >= m // 2, a k-d tree otherwise), a singleton class, and discrete features,
where the jitter is below rounding error and ties are decided by how a
distance is rounded.
"""
import numpy as np
import pytest
from sklearn.feature_selection import mutual_info_classif

import fast_mi


def _cases():
    rng = np.random.RandomState(0)
    # Continuous, small classes only: every class takes the brute-force path.
    yield rng.randn(30, 40), rng.randint(0, 8, 30)
    # Continuous, large classes: the k-d tree path.
    yield rng.randn(400, 20), rng.randint(0, 3, 400)
    # Discrete values (many exact ties), mixed class sizes, one singleton.
    y = np.r_[np.zeros(40, int), np.ones(5, int), [2], np.full(12, 3)]
    yield rng.randint(0, 4, (y.size, 60)).astype(float) * 250.0, y
    # Wide, few rows, discrete: the shape of the gene-expression benchmarks.
    yield rng.randint(-3, 4, (34, 300)).astype(float), rng.randint(0, 9, 34)


@pytest.mark.parametrize("case", range(4))
@pytest.mark.parametrize("seed", [0, 7])
def test_equals_sklearn(case, seed):
    X, y = list(_cases())[case]
    expected = mutual_info_classif(X, y, random_state=seed)
    got = fast_mi.mutual_info_classif(X, y, random_state=seed)
    assert np.array_equal(got, expected)


def test_draws_from_the_global_generator_like_sklearn():
    """With no random_state both consume numpy's global generator the same
    way, so everything FASTENER draws afterwards is unchanged."""
    X, y = list(_cases())[2]
    np.random.seed(3)
    expected = mutual_info_classif(X, y)
    after_expected = np.random.rand()
    np.random.seed(3)
    got = fast_mi.mutual_info_classif(X, y)
    assert np.array_equal(got, expected)
    assert np.random.rand() == after_expected
