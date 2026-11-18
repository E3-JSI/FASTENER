"""Tests for the genetic operators: mating, mating selection, mutation, swap."""
import numpy as np
import pytest

from conftest import N_FEATURES, eval_item, genes_from, item_from, selected

import random_utils
from item import (
    InformationGainSwapStrategy,
    IntersectionMating,
    IntersectionMatingWithInformationGain,
    IntersectionMatingWithWeightedRandomInformationGain,
    Item,
    NoMating,
    RandomEveryoneWithEveryone,
    RandomFlipMutationStrategy,
    RandomSwapStrategy,
    UnionMating,
    flatten_population,
)


def information_gain(**values):
    """An MI vector that is zero except for the given `f<index>=value` entries."""
    vector = np.zeros(N_FEATURES)
    for key, value in values.items():
        vector[int(key[1:])] = value
    return vector


# --------------------------------------------------------------------------
# Mating
# --------------------------------------------------------------------------

def test_union_mating():
    child = UnionMating().mate(item_from([0, 1]), item_from([1, 5]), 4)

    assert selected(child.genes) == {0, 1, 5}


def test_intersection_mating():
    child = IntersectionMating().mate(item_from([0, 1, 2]), item_from([1, 2, 5]), 4)

    assert selected(child.genes) == {1, 2}


def test_mating_records_parents_and_generation():
    a, b = item_from([0, 1]), item_from([1, 5])

    child = UnionMating().mate(a, b, 4)

    assert child.generation == 4
    assert child.parent_a == a.genes and child.parent_b == b.genes


def test_mating_does_not_modify_the_parents():
    a, b = item_from([0, 1]), item_from([1, 5])

    IntersectionMating().mate(a, b, 1)

    assert selected(a.genes) == {0, 1} and selected(b.genes) == {1, 5}


@pytest.mark.parametrize("difference, expected", [
    (0, 0), (1, 1), (2, 2), (3, 2), (4, 3), (5, 3), (10, 6),
])
def test_default_number_of_genes_added_back(difference, expected):
    assert IntersectionMatingWithInformationGain.default_number(difference) \
        == expected


def test_information_gain_mating_adds_back_the_most_informative_genes():
    mating = IntersectionMatingWithInformationGain()
    mating.scikit_information_gain = information_gain(f2=0.3, f4=0.9, f5=0.1)
    # Intersection {0}; difference {1..5}; size gap 3 -> add back 2.
    child = mating.mate(item_from([0, 1]), item_from([0, 2, 3, 4, 5]), 1)

    assert selected(child.genes) == {0, 2, 4}


def test_information_gain_mating_with_equal_sizes_is_plain_intersection():
    mating = IntersectionMatingWithInformationGain()
    mating.scikit_information_gain = information_gain(f1=1.0, f2=1.0)

    child = mating.mate(item_from([0, 1]), item_from([0, 2]), 1)

    assert selected(child.genes) == {0}


def test_use_data_information_ranks_informative_features_first(small_data):
    X, y, _, _ = small_data
    mating = IntersectionMatingWithInformationGain()

    mating.use_data_information(X, y)

    assert len(mating.scikit_information_gain) == N_FEATURES
    top_two = set(np.argsort(mating.scikit_information_gain)[-2:])
    assert top_two == {0, 1}


def test_weighted_mating_child_lies_between_intersection_and_union():
    mating = IntersectionMatingWithWeightedRandomInformationGain()
    mating.scikit_information_gain = np.linspace(0.1, 0.8, N_FEATURES)
    a, b = item_from([0, 1, 6]), item_from([0, 2, 3, 4, 5, 7])

    random_utils.seed(0)
    for _ in range(50):
        child = selected(mating.mate(a, b, 1).genes)
        assert {0} <= child <= {0, 1, 2, 3, 4, 5, 6, 7}


def test_weighted_mating_never_picks_zero_information_genes():
    mating = IntersectionMatingWithWeightedRandomInformationGain(
        number=lambda x: 1)
    mating.scikit_information_gain = information_gain(f3=0.2, f4=0.9)
    a, b = item_from([0, 1]), item_from([0, 2, 3, 4, 5])

    random_utils.seed(0)
    for _ in range(50):
        child = selected(mating.mate(a, b, 1).genes)
        assert child - {0} <= {3, 4}


def test_weighted_mating_falls_back_to_uniform_without_information():
    mating = IntersectionMatingWithWeightedRandomInformationGain()
    mating.scikit_information_gain = np.zeros(N_FEATURES)
    a, b = item_from([0, 1]), item_from([0, 2, 3, 4, 5])

    random_utils.seed(0)
    reached = set()
    for _ in range(200):
        reached |= selected(mating.mate(a, b, 1).genes) - {0}

    assert reached == {1, 2, 3, 4, 5}


def test_weighted_mating_of_identical_parents_is_the_parent():
    mating = IntersectionMatingWithWeightedRandomInformationGain()
    mating.scikit_information_gain = np.ones(N_FEATURES)

    child = mating.mate(item_from([2, 3]), item_from([2, 3]), 1)

    assert selected(child.genes) == {2, 3}


# Parents with intersection {0}, symmetric difference {1..5}, size gap 3 -> k=2.
SMALL, LARGE = item_from([0, 1]), item_from([0, 2, 3, 4, 5])
K = 2
CLOSE_INFORMATION = information_gain(f1=0.5, f2=0.4, f3=0.4, f4=0.4, f5=0.4)


def test_without_top_genes_weighted_mating_adds_back_at_most_k_genes():
    mating = IntersectionMatingWithWeightedRandomInformationGain(
        include_top_genes=False)
    mating.scikit_information_gain = CLOSE_INFORMATION

    random_utils.seed(0)
    for _ in range(100):
        child = selected(mating.mate(SMALL, LARGE, 1).genes)
        assert 1 <= len(child - {0}) <= K


def test_weighted_mating_keeps_the_top_genes_by_default():
    assert IntersectionMatingWithWeightedRandomInformationGain().include_top_genes


def test_without_top_genes_weighted_mating_does_not_always_take_them():
    mating = IntersectionMatingWithWeightedRandomInformationGain(
        include_top_genes=False)
    mating.scikit_information_gain = CLOSE_INFORMATION

    random_utils.seed(0)
    children = [selected(mating.mate(SMALL, LARGE, 1).genes) for _ in range(100)]

    assert any(1 not in child for child in children)


def test_top_genes_flag_always_takes_the_top_genes_and_samples_more():
    mating = IntersectionMatingWithWeightedRandomInformationGain(
        include_top_genes=True)
    mating.scikit_information_gain = CLOSE_INFORMATION
    top_k = selected(IntersectionMatingWithInformationGain.mate_internal(
        mating, SMALL, LARGE))

    random_utils.seed(0)
    children = [selected(mating.mate(SMALL, LARGE, 1).genes) for _ in range(100)]

    assert all(top_k <= child for child in children)
    assert all(len(child - {0}) <= 2 * K for child in children)
    assert any(len(child - {0}) > K for child in children)


def test_top_genes_flag_reproduces_the_original_child():
    """The original: the parent class's top-k child, then k weighted draws."""
    mating = IntersectionMatingWithWeightedRandomInformationGain(
        include_top_genes=True)
    mating.scikit_information_gain = CLOSE_INFORMATION
    difference = [1, 2, 3, 4, 5]
    weights = mating.scaling([CLOSE_INFORMATION[i] for i in difference])

    for seed in range(20):
        random_utils.seed(seed)
        child = mating.mate(SMALL, LARGE, 1).genes

        random_utils.seed(seed)
        expected = IntersectionMatingWithInformationGain.mate_internal(
            mating, SMALL, LARGE)
        for i in random_utils.choices(difference, p=weights / weights.sum(),
                                      size=K):
            expected[i] = True

        assert child == expected


# --------------------------------------------------------------------------
# Mating selection
# --------------------------------------------------------------------------

def population_of(*items):
    population = {}
    for item in items:
        population.setdefault(item.size, []).append(item)
    return population


@pytest.mark.parametrize("pool", [2, 3, 5])
def test_everyone_with_everyone_mates_every_pair(pool):
    strategy = RandomEveryoneWithEveryone(UnionMating(), pool_size=pool)
    parents = [eval_item([i], 0.5) for i in range(pool)]

    children = strategy.mate_pool(parents, current_generation=3)

    assert len(children) == pool * (pool - 1) // 2
    assert all(c.generation == 4 for c in children)


def five_items():
    return population_of(*[eval_item([i], 0.1 * i) for i in range(4)],
                         eval_item([0, 1], 0.9))


def test_mating_pool_is_drawn_from_the_population():
    population = five_items()
    strategy = RandomEveryoneWithEveryone(UnionMating(), pool_size=3)

    random_utils.seed(0)
    result = strategy.mating_pool(population)

    assert len(result.mating_pool) == 3
    assert all(p in flatten_population(population) for p in result.mating_pool)
    assert result.carry_over == flatten_population(population)


def test_mating_pool_has_no_repeated_items():
    strategy = RandomEveryoneWithEveryone(UnionMating(), pool_size=4)

    random_utils.seed(0)
    for _ in range(100):
        pool = strategy.mating_pool(five_items()).mating_pool
        assert len({p.number for p in pool}) == 4


def test_mating_pool_is_capped_at_the_population_size():
    population = population_of(eval_item([0], 0.1), eval_item([1], 0.2))
    strategy = RandomEveryoneWithEveryone(UnionMating(), pool_size=10)

    pool = strategy.mating_pool(population).mating_pool

    assert {p.number for p in pool} == \
        {p.number for p in flatten_population(population)}


def test_mating_pool_of_none_takes_the_whole_population():
    population = five_items()
    strategy = RandomEveryoneWithEveryone(UnionMating(), pool_size=None)

    pool = strategy.mating_pool(population).mating_pool

    assert sorted(p.number for p in pool) == \
        sorted(p.number for p in flatten_population(population))


def test_mating_pool_of_none_works_with_self_mating():
    strategy = RandomEveryoneWithEveryone(UnionMating(), pool_size=None,
                                          allow_self_mating=True)

    pool = strategy.mating_pool(five_items()).mating_pool

    assert len(pool) == 5


def test_self_mating_flag_reproduces_the_original_draw():
    """The original pool was `choices(flatten, size=pool_size)`, repeats and all."""
    population = five_items()
    strategy = RandomEveryoneWithEveryone(UnionMating(), pool_size=4,
                                          allow_self_mating=True)

    random_utils.seed(3)
    pool = strategy.mating_pool(population).mating_pool
    random_utils.seed(3)
    expected = random_utils.choices(flatten_population(population), size=4)

    assert [p.number for p in pool] == [p.number for p in expected]


def test_self_mating_flag_allows_repeated_items():
    strategy = RandomEveryoneWithEveryone(UnionMating(), pool_size=4,
                                          allow_self_mating=True)

    random_utils.seed(0)
    repeated = sum(
        len({p.number for p in strategy.mating_pool(five_items()).mating_pool}) < 4
        for _ in range(100))

    assert repeated > 0


def test_process_population_keeps_parents_and_buckets_children_by_size():
    population = population_of(eval_item([0], 0.1), eval_item([1], 0.2),
                               eval_item([2, 3], 0.3))
    strategy = RandomEveryoneWithEveryone(UnionMating(), pool_size=3)

    random_utils.seed(0)
    new = strategy.process_population(population, 1)

    items = flatten_population(new)
    assert len(items) == 3 + 3  # carried-over parents + 3 pairs
    assert all(item.size == size for size, bucket in new.items()
               for item in bucket)
    for parent in flatten_population(population):
        assert parent in items


def test_selection_strategy_passes_data_to_its_mating_strategy(small_data):
    X, y, _, _ = small_data
    mating = IntersectionMatingWithInformationGain()
    RandomEveryoneWithEveryone(mating).use_data_information(X, y)

    assert len(mating.scikit_information_gain) == N_FEATURES


def test_no_mating_carries_each_item_to_the_next_generation():
    population = population_of(eval_item([0], 0.1, generation=2),
                               eval_item([1, 2], 0.3, generation=2))

    new = NoMating(UnionMating()).process_population(population, 2)

    items = flatten_population(new)
    assert {i.number for i in items} == \
        {i.number for i in flatten_population(population)}
    assert all(i.generation == 3 for i in items)


# --------------------------------------------------------------------------
# Mutation
# --------------------------------------------------------------------------

def test_mutation_with_zero_probability_changes_nothing():
    item = item_from([1, 4])

    assert RandomFlipMutationStrategy(0.0).mutate(item).genes == item.genes


def test_mutation_with_certain_probability_flips_everything():
    mutated = RandomFlipMutationStrategy(1.0).mutate(item_from([1, 4]))

    assert selected(mutated.genes) == set(range(N_FEATURES)) - {1, 4}


def test_mutation_keeps_generation_and_parents():
    parent_a, parent_b = genes_from([1]), genes_from([4])
    item = Item(genes_from([1, 4]), 5, parent_a, parent_b)

    mutated = RandomFlipMutationStrategy(0.5).mutate(item)

    assert mutated.generation == 5
    assert mutated.parent_a == parent_a and mutated.parent_b == parent_b


def test_mutation_rebuckets_by_the_new_size():
    population = {2: [item_from([1, 4])], 1: [item_from([0])]}

    new = RandomFlipMutationStrategy(1.0).process_population(population)

    assert sorted(new) == [N_FEATURES - 2, N_FEATURES - 1]
    assert all(item.size == size for size, bucket in new.items()
               for item in bucket)


def test_mutation_rate_is_roughly_the_probability():
    random_utils.seed(0)
    strategy = RandomFlipMutationStrategy(0.25)
    flips = sum(
        len(selected(strategy.mutate(item_from([], n_features=100)).genes))
        for _ in range(40))

    assert 0.2 < flips / 4000 < 0.3


# --------------------------------------------------------------------------
# Swap
# --------------------------------------------------------------------------

CHANGES = [(0.0, 3), (0.1, 0), (0.4, 1)]  # feature 3 matters least


@pytest.mark.parametrize("strategy_class",
                         [RandomSwapStrategy, InformationGainSwapStrategy])
def test_swap_keeps_the_size_and_changes_one_feature(strategy_class):
    strategy = strategy_class()
    strategy.scikit_information_gain = np.ones(N_FEATURES)
    item = item_from([0, 1, 3], generation=2)

    random_utils.seed(0)
    (candidate,) = strategy.propose(item, CHANGES)

    assert candidate.size == item.size
    assert len(selected(candidate.genes) ^ selected(item.genes)) == 2
    assert candidate.generation == 3


@pytest.mark.parametrize("strategy_class",
                         [RandomSwapStrategy, InformationGainSwapStrategy])
def test_swap_proposes_distinct_candidates(strategy_class):
    strategy = strategy_class(number_of_swaps=3)
    strategy.scikit_information_gain = np.ones(N_FEATURES)

    random_utils.seed(0)
    candidates = strategy.propose(item_from([0, 1, 3]), CHANGES)

    assert len(candidates) == 3
    assert len({c.number for c in candidates}) == 3


@pytest.mark.parametrize("strategy_class",
                         [RandomSwapStrategy, InformationGainSwapStrategy])
def test_swap_has_nothing_to_propose_for_a_full_or_unscored_subset(strategy_class):
    strategy = strategy_class()
    full = item_from(range(N_FEATURES))

    assert strategy.propose(full, [(0.0, 0)]) == []
    assert strategy.propose(item_from([0, 1]), []) == []


def test_guided_swap_removes_the_permutation_weakest_feature():
    strategy = InformationGainSwapStrategy()

    assert strategy.select_removal(genes_from([0, 1, 3]), CHANGES) == 3


def test_guided_swap_inserts_only_informative_features():
    strategy = InformationGainSwapStrategy()
    strategy.scikit_information_gain = information_gain(f6=0.5)

    random_utils.seed(0)
    for _ in range(20):
        (candidate,) = strategy.propose(item_from([0, 1, 3]), CHANGES)
        assert selected(candidate.genes) == {0, 1, 6}


def test_guided_swap_reuses_a_given_information_vector(small_data):
    X, y, _, _ = small_data
    vector = np.arange(N_FEATURES, dtype=float)
    strategy = InformationGainSwapStrategy()

    strategy.use_data_information(X, y, vector)

    assert strategy.scikit_information_gain is vector


def test_guided_swap_computes_its_own_vector_when_none_is_given(small_data):
    X, y, _, _ = small_data
    strategy = InformationGainSwapStrategy()

    strategy.use_data_information(X, y, None)

    assert len(strategy.scikit_information_gain) == N_FEATURES
