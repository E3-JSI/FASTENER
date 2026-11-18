"""Tests for the filter seeds: mRMR and Boruta initial populations."""
import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from conftest import N_FEATURES, selected

import seeding
from fastener import Config, EntropyOptimizer
from item import RandomEveryoneWithEveryone, RandomFlipMutationStrategy

from test_optimizer import AccuracyEvaluator, CountingModelFactory


def seeded_optimizer(small_data, seed_filter, name="seeded", **kwargs):
    X, y, X_val, y_val = small_data
    config = {"number_of_rounds": 4, "reset_to_pareto_rounds": 2,
              **kwargs.pop("config", {})}
    factory = CountingModelFactory()
    optimizer = EntropyOptimizer(
        factory, X, y, AccuracyEvaluator(X_val, y_val), N_FEATURES,
        RandomEveryoneWithEveryone(pool_size=3),
        RandomFlipMutationStrategy(1 / N_FEATURES),
        seed_filter=seed_filter,
        config=Config(name, 2020, **config),
        **kwargs)
    optimizer.factory = factory
    return optimizer


def initial_subsets(optimizer):
    return sorted((selected(item.genes) for items in
                   optimizer.population.values() for item in items),
                  key=len)


# --------------------------------------------------------------------------
# The filters
# --------------------------------------------------------------------------

def test_nested_prefixes():
    assert seeding.nested_prefixes(np.array([4, 2, 7]), 2) == [[4], [4, 2]]
    assert seeding.nested_prefixes(np.array([4, 2]), 5) == [[4], [4, 2]]


def test_discretize_bins_equal_width_and_flattens_constants():
    X = np.array([[0.0, 1.0], [5.0, 1.0], [10.0, 1.0]])
    codes = seeding.discretize(X, n_bins=2)
    assert codes[:, 0].tolist() == [0, 1, 1]
    assert codes[:, 1].tolist() == [0, 0, 0]


def test_mrmr_puts_the_informative_features_first(small_data):
    X, y, _, _ = small_data
    order = seeding.rank_mrmr(X, y, 5, random_state=0)

    assert len(order) == 5 and len(set(order)) == 5
    assert set(order[:2]) == {0, 1}


def test_mrmr_cannot_select_more_features_than_exist(small_data):
    X, y, _, _ = small_data
    order = seeding.rank_mrmr(X, y, 100, random_state=0)
    assert sorted(order) == list(range(N_FEATURES))


def test_mrmr_is_deterministic_for_a_random_state(small_data):
    X, y, _, _ = small_data
    assert (seeding.rank_mrmr(X, y, 5, random_state=3)
            == seeding.rank_mrmr(X, y, 5, random_state=3)).all()


def test_boruta_ranks_every_feature_and_counts_its_fits(small_data):
    pytest.importorskip("boruta")
    X, y, _, _ = small_data
    factory = CountingModelFactory()
    order, confirmed, fits = seeding.rank_boruta(factory, X, y,
                                                 random_state=0, max_iter=20)

    assert sorted(order) == list(range(N_FEATURES))
    assert set(order[:2]) == {0, 1}
    assert {0, 1} <= set(confirmed)
    assert 0 < fits <= 20
    assert factory.fits == fits


def test_boruta_passes_its_random_state_to_the_model(small_data):
    pytest.importorskip("boruta")
    X, y, _, _ = small_data
    model = seeding._ModelForBoruta(DecisionTreeClassifier)
    model.set_params(n_estimators=7, random_state=11).fit(X, y)
    assert model.model_.random_state == 11

    with pytest.raises(ValueError):
        model.set_params(max_depth=3)


# --------------------------------------------------------------------------
# The optimizer option
# --------------------------------------------------------------------------

def test_unknown_seed_filter_is_rejected(small_data, run_dir):
    with pytest.raises(ValueError, match="seed_filter"):
        seeded_optimizer(small_data, "relieff")


def test_no_seed_by_default(small_data, run_dir):
    X, y, X_val, y_val = small_data
    optimizer = EntropyOptimizer(
        CountingModelFactory(), X, y, AccuracyEvaluator(X_val, y_val),
        N_FEATURES, RandomEveryoneWithEveryone(pool_size=3),
        RandomFlipMutationStrategy(1 / N_FEATURES), initial_genes=[[3]],
        config=Config("plain", 2020))
    optimizer.prepare_loop()

    assert optimizer.seed_filter is None and optimizer.seed_ranking is None
    assert initial_subsets(optimizer) == [{3}]


def test_mrmr_seed_is_the_nested_top_k(small_data, run_dir):
    X, y, _, _ = small_data
    optimizer = seeded_optimizer(small_data, "mrmr", seed_size=4)
    optimizer.prepare_loop()

    order = seeding.rank_mrmr(X, y, 4, random_state=2020)
    assert optimizer.seed_ranking == order.tolist()
    assert initial_subsets(optimizer) == [set(order[:k]) for k in range(1, 5)]
    # mRMR fits no model; the four subsets cost one fit each.
    assert optimizer.seed_fits == 0
    assert optimizer.model_fits == optimizer.factory.fits == 4


def test_seed_is_added_to_initial_genes_without_duplicates(small_data,
                                                           run_dir):
    X, y, _, _ = small_data
    first = int(seeding.rank_mrmr(X, y, 1, random_state=2020)[0])
    other = next(i for i in range(N_FEATURES) if i != first)
    optimizer = seeded_optimizer(small_data, "mrmr", seed_size=3,
                                 initial_genes=[[first], [other]])
    optimizer.prepare_loop()

    subsets = initial_subsets(optimizer)
    assert len(subsets) == 4
    assert subsets.count({first}) == 1 and {other} in subsets


def test_boruta_seed_charges_its_fits_to_the_run(small_data, run_dir):
    pytest.importorskip("boruta")
    optimizer = seeded_optimizer(small_data, "boruta", seed_size=3,
                                 boruta_max_iter=15)
    optimizer.prepare_loop()

    assert optimizer.seed_fits > 0
    assert len(optimizer.seed_ranking) == 3
    assert set(optimizer.seed_ranking[:2]) == {0, 1}
    assert initial_subsets(optimizer) == [
        set(optimizer.seed_ranking[:k]) for k in range(1, 4)]
    assert optimizer.model_fits == optimizer.seed_fits + 3
    assert optimizer.factory.fits == optimizer.model_fits


def test_boruta_seed_fits_count_against_the_budget(small_data, run_dir):
    pytest.importorskip("boruta")
    budget = 40
    optimizer = seeded_optimizer(small_data, "boruta", seed_size=3,
                                 boruta_max_iter=15,
                                 config={"max_model_fits": budget,
                                         "number_of_rounds": 1000})
    optimizer.mainloop()

    assert optimizer.stop_reason == "fit_budget"
    assert optimizer.model_fits == budget
    assert optimizer.factory.fits == budget


@pytest.mark.parametrize("seed_filter", ["mrmr", "boruta"])
def test_seeded_run_completes(small_data, run_dir, seed_filter):
    if seed_filter == "boruta":
        pytest.importorskip("boruta")
    optimizer = seeded_optimizer(small_data, seed_filter, name=seed_filter,
                                 boruta_max_iter=15)
    optimizer.mainloop()

    assert optimizer.stop_reason == "rounds"
    assert {0, 1} in [selected(item.genes)
                      for item in optimizer.pareto_front.values()]
