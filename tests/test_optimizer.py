"""Tests for EntropyOptimizer: its bookkeeping, pruning, caching and main loop."""
import os
import pickle

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from conftest import N_FEATURES, eval_item, genes_from, item_from, selected

import random_utils
from fastener import Config, EntropyOptimizer, LogData
from item import (
    IntersectionMatingWithWeightedRandomInformationGain,
    Item,
    RandomEveryoneWithEveryone,
    RandomFlipMutationStrategy,
    Result,
)


# Module level, not closures: prepare_loop pickles the optimizer.

class StubModel:
    """Fits instantly; for tests where only the evaluator's score matters."""

    def fit(self, X, y):
        return self


class CountingModelFactory:
    def __init__(self):
        self.fits = 0

    def __call__(self):
        self.fits += 1
        return DecisionTreeClassifier(random_state=0)


class WeightedEvaluator:
    """Score = total weight of the selected features that were not shuffled.

    A deterministic stand-in for permutation importance: shuffling feature `g`
    costs exactly `weights[g]`.
    """

    def __init__(self, weights):
        self.weights = weights

    def __call__(self, model, genes, shuffle_indices=None):
        shuffled = set(shuffle_indices or [])
        on_genes = np.where(genes)[0]
        return Result(float(sum(self.weights[g] for pos, g in enumerate(on_genes)
                                if pos not in shuffled)))


class AccuracyEvaluator:
    """Scores a fitted model on held-out data, shuffling columns if asked."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __call__(self, model, genes, shuffle_indices=None):
        data = self.X[:, genes]
        if shuffle_indices:
            data = data.copy()
            for j in shuffle_indices:
                random_utils.shuffle(data[:, j])
        return Result(float((model.predict(data) == self.y).mean()))


def stub_optimizer(small_data, name="stub", weights=None, **config):
    X, y, _, _ = small_data
    weights = weights if weights is not None else np.ones(N_FEATURES)
    return EntropyOptimizer(
        StubModel, X, y, WeightedEvaluator(weights), N_FEATURES,
        RandomEveryoneWithEveryone(pool_size=3),
        RandomFlipMutationStrategy(1 / N_FEATURES),
        initial_genes=[[0]],
        config=Config(name, 2020, **config))


def real_optimizer(small_data, name="real", seed=2020, rounds=6, **config):
    X, y, X_val, y_val = small_data
    factory = CountingModelFactory()
    optimizer = EntropyOptimizer(
        factory, X, y, AccuracyEvaluator(X_val, y_val), N_FEATURES,
        RandomEveryoneWithEveryone(
            pool_size=3,
            mating_strategy=IntersectionMatingWithWeightedRandomInformationGain()),
        RandomFlipMutationStrategy(1 / N_FEATURES),
        initial_genes=[[2], [3], [5]],
        config=Config(name, seed, number_of_rounds=rounds,
                      reset_to_pareto_rounds=2, **config))
    optimizer.factory = factory
    return optimizer


def assert_is_a_pareto_front(front):
    sizes = sorted(front)
    assert all(front[s].size == s for s in sizes)
    scores = [front[s].result.score for s in sizes]
    # Each larger subset must be strictly better, or it would be dominated.
    assert all(a < b for a, b in zip(scores, scores[1:]))


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

def test_config_writes_under_log():
    assert Config("run", 1).output_folder == os.path.join("log", "run")


@pytest.mark.parametrize("reset", [None, 0, False])
def test_config_without_pareto_reset_resets_only_at_the_end(reset):
    config = Config("run", 1, number_of_rounds=17, reset_to_pareto_rounds=reset)

    assert config.reset_to_pareto_rounds == 17


def test_config_seeds_the_random_generator():
    Config("a", 42)
    first = random_utils.random()
    Config("b", 42)

    assert random_utils.random() == first


def test_config_defaults():
    config = Config("run", 1)

    assert config.number_of_rounds == 1000
    assert config.max_bucket_size == 3
    assert config.reset_to_pareto_rounds == 5
    assert config.cache_fitness_function is True
    assert config.max_model_fits is None
    assert config.dump_generation_logs is True


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------

def test_optimizer_creates_its_output_folder(small_data, run_dir):
    stub_optimizer(small_data, "made")

    assert (run_dir / "log" / "made").is_dir()


def test_optimizer_refuses_to_overwrite_a_previous_run(small_data, run_dir):
    stub_optimizer(small_data, "taken")

    with pytest.raises(FileExistsError):
        stub_optimizer(small_data, "taken")


def test_prepare_loop_needs_an_initial_population(small_data, run_dir):
    optimizer = stub_optimizer(small_data)
    optimizer.initial_genes = None

    with pytest.raises(AssertionError, match="initial population"):
        optimizer.prepare_loop()


def test_initial_genes_become_the_first_population(small_data, run_dir):
    optimizer = stub_optimizer(small_data)
    optimizer.initial_genes = [[0], [1], [2, 3]]

    optimizer.prepare_loop()

    assert sorted(optimizer.population) == [1, 2]
    assert len(optimizer.population[1]) == 2
    assert selected(optimizer.population[2][0].genes) == {2, 3}


def test_an_initial_population_is_evaluated(small_data, run_dir):
    optimizer = stub_optimizer(small_data)
    optimizer.initial_genes = None
    optimizer.initial_population = {2: [item_from([0, 1])]}

    optimizer.prepare_loop()

    assert optimizer.population[2][0].result == Result(2.0)


def test_prepare_loop_pickles_the_experiment(small_data, run_dir):
    optimizer = stub_optimizer(small_data, "pickled")
    optimizer.prepare_loop()

    path = run_dir / "log" / "pickled" / "experiment.pickle"
    with open(path, "rb") as f:
        restored = pickle.load(f)
    assert restored.number_of_genes == N_FEATURES


# --------------------------------------------------------------------------
# Fitness and caching
# --------------------------------------------------------------------------

def test_cached_fitness_fits_each_subset_once(small_data, run_dir):
    optimizer = stub_optimizer(small_data)
    genes = genes_from([0, 2])

    first = optimizer.cached_fitness(genes)
    second = optimizer.cached_fitness(genes)

    assert first is second
    assert optimizer.model_fits == 1
    assert optimizer.cache_counter[Item.to_number(genes)] == 2


def test_cached_fitness_distinguishes_subsets(small_data, run_dir):
    optimizer = stub_optimizer(small_data)

    optimizer.cached_fitness(genes_from([0]))
    optimizer.cached_fitness(genes_from([1]))

    assert optimizer.model_fits == 2
    assert len(optimizer.cache_data) == 2


def test_caching_is_enabled_by_prepare_loop(small_data, run_dir):
    optimizer = stub_optimizer(small_data)
    assert optimizer.fitness_function == optimizer._fitness_function

    optimizer.prepare_loop()

    assert optimizer.fitness_function == optimizer.cached_fitness


def test_caching_can_be_turned_off(small_data, run_dir):
    optimizer = stub_optimizer(small_data, cache_fitness_function=False)
    optimizer.prepare_loop()

    optimizer.fitness_function(genes_from([0, 1]))
    optimizer.fitness_function(genes_from([0, 1]))

    assert optimizer.fitness_function == optimizer._fitness_function
    assert optimizer.cache_data == {}


def test_train_model_uses_only_the_selected_columns(small_data, run_dir):
    optimizer = stub_optimizer(small_data)
    optimizer._model = CountingModelFactory()

    model = optimizer.train_model(np.array(genes_from([1, 4])))

    assert model.n_features_in_ == 2


# --------------------------------------------------------------------------
# Population and front bookkeeping
# --------------------------------------------------------------------------

def test_oversize_buckets_keep_the_best_items(small_data, run_dir):
    optimizer = stub_optimizer(small_data, max_bucket_size=2)
    optimizer.population = {
        1: [eval_item([i], s) for i, s in enumerate([0.3, 0.9, 0.1, 0.5])],
        2: [eval_item([0, 1], 0.4)],
    }

    optimizer.purge_oversize_buckets()

    assert [i.result.score for i in optimizer.population[1]] == [0.9, 0.5]
    assert len(optimizer.population[2]) == 1


def test_front_takes_the_best_of_a_new_size(small_data, run_dir):
    optimizer = stub_optimizer(small_data)
    optimizer.population = {1: [eval_item([1], 0.8), eval_item([2], 0.3)]}

    optimizer.update_front_from_population()

    assert optimizer.pareto_front[1].result.score == 0.8


def test_front_keeps_a_better_entry_and_takes_a_better_candidate(small_data, run_dir):
    optimizer = stub_optimizer(small_data)
    optimizer.pareto_front = {1: eval_item([1], 0.5), 2: eval_item([0, 1], 0.5)}
    optimizer.population = {1: [eval_item([2], 0.4)],
                            2: [eval_item([0, 2], 0.7)],
                            3: []}

    optimizer.update_front_from_population()

    assert optimizer.pareto_front[1].result.score == 0.5
    assert optimizer.pareto_front[2].result.score == 0.7
    assert 3 not in optimizer.pareto_front


def test_dominated_front_entries_are_removed(small_data, run_dir):
    optimizer = stub_optimizer(small_data)
    optimizer.pareto_front = {
        1: eval_item([0], 0.5),
        2: eval_item([0, 1], 0.4),        # worse than size 1
        3: eval_item([0, 1, 2], 0.5),     # only equal to size 1
        4: eval_item([0, 1, 2, 3], 0.8),
        5: eval_item([0, 1, 2, 3, 4], 0.7),  # worse than size 4
    }

    optimizer.remove_pareto_non_optimal()

    assert sorted(optimizer.pareto_front) == [1, 4]
    assert_is_a_pareto_front(optimizer.pareto_front)


def test_reset_population_to_front(small_data, run_dir):
    optimizer = stub_optimizer(small_data)
    front = {2: eval_item([0, 1], 0.6), 1: eval_item([0], 0.5)}
    optimizer.pareto_front = front
    optimizer.population = {3: [eval_item([0, 1, 2], 0.1)]}

    optimizer.reset_population_to_front()

    assert list(optimizer.population) == [1, 2]
    assert optimizer.population == {1: [front[1]], 2: [front[2]]}


def test_duplicates_are_cleared_keeping_the_first(small_data, run_dir):
    first = item_from([0, 1], generation=1)
    un_pop = {2: [first, item_from([0, 1], generation=2), item_from([1, 2])],
              1: [item_from([3])]}

    cleared = EntropyOptimizer.clear_duplicates_after_mating(un_pop)

    assert [selected(i.genes) for i in cleared[2]] == [{0, 1}, {1, 2}]
    assert cleared[2][0] is first
    assert len(cleared[1]) == 1


def test_empty_subsets_are_not_evaluated(small_data, run_dir):
    optimizer = stub_optimizer(small_data)

    population = optimizer.evaluate_unevaluated(
        {0: [item_from([])], 1: [item_from([2])]})

    assert population[0] == []
    assert population[1][0].result == Result(1.0)
    assert optimizer.model_fits == 1


@pytest.mark.parametrize("round_n, expected", [(1, False), (3, True), (6, True),
                                               (7, False)])
def test_default_reset_predicate(round_n, expected):
    assert EntropyOptimizer.default_reset_to_pareto(round_n, {}, {}) is expected


# --------------------------------------------------------------------------
# Pruning by permutation importance
# --------------------------------------------------------------------------

def test_pruning_drops_the_least_important_feature(small_data, run_dir):
    weights = np.array([0.5, 0.1, 0.9, 0.3, 0, 0, 0, 0])
    optimizer = stub_optimizer(small_data, weights=weights)
    item = eval_item([0, 1, 2, 3], 1.8, generation=4)

    pruned = optimizer.purge_item_with_information_gain(item)

    assert selected(pruned.genes) == {0, 2, 3}
    assert pruned.generation == 5
    assert pruned.result.score == pytest.approx(1.7)


def test_pruning_does_not_modify_the_item(small_data, run_dir):
    optimizer = stub_optimizer(small_data)
    item = eval_item([0, 1], 2.0)

    optimizer.purge_item_with_information_gain(item)

    assert selected(item.genes) == {0, 1}


def test_pruning_the_front_fills_the_size_below(small_data, run_dir):
    weights = np.array([0.5, 0.1, 0.9, 0, 0, 0, 0, 0])
    optimizer = stub_optimizer(small_data, weights=weights)
    optimizer.pareto_front = {3: eval_item([0, 1, 2], 1.5)}

    optimizer.purge_front_with_information_gain()

    assert selected(optimizer.pareto_front[2].genes) == {0, 2}
    assert_is_a_pareto_front(optimizer.pareto_front)


def test_pruning_replaces_a_worse_front_entry(small_data, run_dir):
    weights = np.array([0.5, 0.1, 0.9, 0, 0, 0, 0, 0])
    optimizer = stub_optimizer(small_data, weights=weights)
    optimizer.pareto_front = {2: eval_item([0, 1], 0.6),
                              3: eval_item([0, 1, 2], 1.5)}

    optimizer.purge_front_with_information_gain()

    assert selected(optimizer.pareto_front[2].genes) == {0, 2}


def test_single_feature_subsets_are_not_pruned(small_data, run_dir):
    optimizer = stub_optimizer(small_data)
    optimizer.pareto_front = {1: eval_item([0], 1.0)}

    optimizer.purge_front_with_information_gain()

    assert list(optimizer.pareto_front) == [1]
    assert optimizer.model_fits == 0


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def test_mainloop_runs_all_rounds(small_data, run_dir):
    optimizer = real_optimizer(small_data, rounds=5)

    optimizer.mainloop()

    assert optimizer.stop_reason == "rounds"
    assert optimizer.rounds_completed == 5
    assert optimizer.pareto_front
    assert_is_a_pareto_front(optimizer.pareto_front)


def test_mainloop_finds_the_informative_features(small_data, run_dir):
    optimizer = real_optimizer(small_data, rounds=20)

    optimizer.mainloop()

    best = max(optimizer.pareto_front.values(), key=lambda i: i.result.score)
    assert {0, 1} <= selected(best.genes)


def test_model_fit_counter_matches_real_fits(small_data, run_dir):
    optimizer = real_optimizer(small_data)

    optimizer.mainloop()

    assert optimizer.model_fits == optimizer.factory.fits
    assert optimizer.model_fits == len(optimizer.cache_data)


def test_same_seed_gives_the_same_front(small_data, run_dir):
    a = real_optimizer(small_data, "a", seed=7)
    a.mainloop()
    b = real_optimizer(small_data, "b", seed=7)
    b.mainloop()

    assert {n: i.number for n, i in a.pareto_front.items()} == \
        {n: i.number for n, i in b.pareto_front.items()}


def test_generation_logs_are_written_by_default(small_data, run_dir):
    optimizer = real_optimizer(small_data, "logged", rounds=3)
    optimizer.mainloop()

    folder = run_dir / "log" / "logged"
    for n in (1, 2, 3):
        with open(folder / f"generation_{n}.pickle", "rb") as f:
            log = pickle.load(f)
        assert isinstance(log, LogData) and log.generation == n


def test_generation_logs_can_be_turned_off(small_data, run_dir):
    optimizer = real_optimizer(small_data, "quiet", rounds=3,
                               dump_generation_logs=False)
    optimizer.mainloop()

    files = os.listdir(run_dir / "log" / "quiet")
    assert files == ["experiment.pickle"]


def test_log_data_drops_the_models_from_the_cache():
    cache = {1: (Result(0.5), "model"), 3: (Result(0.7), "model")}

    assert LogData.discard_model(cache) == {1: Result(0.5), 3: Result(0.7)}
