"""Tests for the gene encoding and the ordering of items and results."""
import itertools

import pytest

from conftest import eval_item, genes_from, item_from

from item import EvalItem, Item, Result, flatten_population


# --------------------------------------------------------------------------
# Gene encoding
# --------------------------------------------------------------------------

def test_number_is_little_endian():
    assert Item.to_number([True, False, False]) == 1
    assert Item.to_number([False, True, False]) == 2
    assert Item.to_number([False, False, True]) == 4
    assert Item.to_number([True, True, True]) == 7


def test_empty_subset_is_zero():
    assert Item.to_number([False] * 5) == 0


def test_appending_features_keeps_the_number():
    """The point of little endian: cache keys survive adding new features."""
    genes = [True, False, True]

    assert Item.to_number(genes) == Item.to_number(genes + [False, False])


@pytest.mark.parametrize("n_genes", [1, 3, 5])
def test_number_and_genes_round_trip(n_genes):
    for num in range(2 ** n_genes):
        genes = Item.num_to_genes(num, n_genes)
        assert len(genes) == n_genes
        assert Item.to_number(genes) == num


def test_every_subset_gets_a_distinct_number():
    subsets = [list(bits) for bits in itertools.product([False, True], repeat=4)]

    assert len({Item.to_number(g) for g in subsets}) == len(subsets)


def test_item_derives_size_and_number_from_genes():
    item = item_from([1, 3])

    assert item.size == 2
    assert item.number == 2 + 8


def test_from_genes_has_no_parents():
    item = Item.from_genes(genes_from([0]))

    assert item.generation == 0
    assert item.parent_a is None and item.parent_b is None


# --------------------------------------------------------------------------
# Gene set operations
# --------------------------------------------------------------------------

def test_gene_union():
    union = Item.gene_union(genes_from([0, 1]), genes_from([1, 4]))

    assert union == genes_from([0, 1, 4])


def test_gene_intersection():
    inter = Item.gene_intersection(genes_from([0, 1, 2]), genes_from([1, 2, 5]))

    assert inter == genes_from([1, 2])


def test_indexed_symmetric_difference():
    diff = Item.indexed_symmetric_difference(genes_from([0, 1, 2]),
                                             genes_from([1, 2, 5]))

    assert diff == [0, 5]


def test_identical_genes_have_no_symmetric_difference():
    genes = genes_from([2, 3])

    assert Item.indexed_symmetric_difference(genes, genes) == []


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------

def test_items_are_equal_by_subset_only():
    a = Item(genes_from([1, 2]), 0, None, None)
    b = Item(genes_from([1, 2]), 7, genes_from([1]), genes_from([2]))

    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_different_subsets_are_not_equal():
    assert item_from([1]) != item_from([2])


def test_eval_items_are_equal_by_subset_regardless_of_score():
    assert eval_item([1, 2], 0.1) == eval_item([1, 2], 0.9)
    assert len({eval_item([1, 2], 0.1), eval_item([1, 2], 0.9)}) == 1


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def test_evaluate_calls_the_fitness_function_with_the_genes():
    calls = []

    def fitness(genes):
        calls.append(list(genes))
        return Result(0.5), "model"

    parent_a, parent_b = genes_from([0]), genes_from([2])
    item = Item(genes_from([0, 2]), 3, parent_a, parent_b)
    evaluated = item.evaluate(fitness)

    assert calls == [genes_from([0, 2])]
    assert isinstance(evaluated, EvalItem)
    assert evaluated.result == Result(0.5)
    assert evaluated.genes == item.genes
    assert evaluated.generation == 3
    assert evaluated.parent_a == parent_a and evaluated.parent_b == parent_b


def test_evaluated_item_is_not_evaluated_again():
    def fitness(genes):
        raise AssertionError("an EvalItem must not be re-evaluated")

    item = eval_item([1], 0.4)

    assert item.evaluate(fitness) is item


# --------------------------------------------------------------------------
# Ordering
# --------------------------------------------------------------------------

def test_results_compare_by_score():
    assert Result(0.1) < Result(0.2)
    assert Result(0.2) <= Result(0.2)
    assert not Result(0.3) < Result(0.2)
    assert Result(0.2) == Result(0.2)


def test_same_size_items_sort_by_result():
    items = [eval_item([0], 0.5), eval_item([1], 0.9), eval_item([2], 0.1)]

    ranked = sorted(items, reverse=True)

    assert [i.result.score for i in ranked] == [0.9, 0.5, 0.1]
    assert max(items).result.score == 0.9


def test_items_of_different_sizes_are_not_ordered():
    with pytest.raises(AssertionError):
        _ = eval_item([0], 0.5) < eval_item([0, 1], 0.9)


@pytest.mark.parametrize("small, big, expected", [
    ((0.9,), (0.5,), True),    # fewer genes, better score
    ((0.5,), (0.5,), True),    # fewer genes, equal score
    ((0.4,), (0.5,), False),   # fewer genes, worse score
])
def test_pareto_better_against_a_larger_subset(small, big, expected):
    smaller = eval_item([0], small[0])
    larger = eval_item([0, 1], big[0])

    assert smaller.pareto_better(larger) is expected


def test_larger_subset_is_never_pareto_better():
    assert not eval_item([0, 1], 0.99).pareto_better(eval_item([0], 0.1))


# --------------------------------------------------------------------------
# Populations
# --------------------------------------------------------------------------

def test_flatten_population():
    a, b, c = eval_item([0], 0.1), eval_item([1], 0.2), eval_item([0, 1], 0.3)

    assert flatten_population({1: [a, b], 2: [c]}) == [a, b, c]
    assert flatten_population({}) == []
