"""Writing your own strategies.

Each operator is a small abstract class in `item.py`:

* `MutationStrategy.mutate_internal(item) -> genes`
* `MatingStrategy.mate_internal(item1, item2) -> genes`, optionally
  `use_data_information(X, y)`, called once before the search
* `MatingSelectionStrategy.mating_pool(population) -> MatingPoolResult`
  and `mate_pool(pool, generation) -> [Item]`
* `SwapStrategy.select_removal(genes, changes)` and
  `select_insertions(genes, removed, count)`

`genes` is a list of bools, one per feature. Draw random numbers through
`random_utils`, which `Config(random_seed=...)` seeds, so runs repeat.
Strategies are pickled with the optimizer: keep them module-level classes.
"""
from _common import ValidationEvaluator, classification_data, front_table, run_folder

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier

import random_utils
from fastener import Config, EntropyOptimizer
from item import (IntersectionMatingWithWeightedRandomInformationGain,
                  MatingPoolResult, MatingSelectionStrategy, MatingStrategy,
                  MutationStrategy, RandomEveryoneWithEveryone,
                  RandomFlipMutationStrategy, SwapStrategy, flatten_population)

X_train, y_train, X_val, y_val, X_test, y_test, names = classification_data()
n = X_train.shape[1]
evaluator = ValidationEvaluator(X_val, y_val, balanced_accuracy_score)


class SizeCappedMutation(MutationStrategy):
    """Random flips, then drop random features until at most `max_size`
    remain: a hard cap on subset size."""

    def __init__(self, prob, max_size):
        self.prob, self.max_size = prob, max_size

    def mutate_internal(self, item):
        genes = np.asarray(item.genes, dtype=bool) ^ \
            (random_utils.random(len(item.genes)) < self.prob)
        on = np.flatnonzero(genes)
        if len(on) > self.max_size:
            genes[random_utils.choices(on, size=len(on) - self.max_size,
                                       replace=False)] = False
        return genes.tolist()


class UniformCrossover(MatingStrategy):
    """Each gene from one parent or the other, with equal probability."""

    def mate_internal(self, item1, item2):
        take_first = random_utils.random(len(item1.genes)) < 0.5
        return np.where(take_first, item1.genes, item2.genes).tolist()


class Tournament(MatingSelectionStrategy):
    """Pick `pool_size` parents, each the best of `k` random items (compared
    by score, whatever their size), and mate consecutive pairs."""

    def __init__(self, mating_strategy, pool_size=6, k=3):
        super().__init__(mating_strategy)
        self.pool_size, self.k = pool_size, k

    def mating_pool(self, population):
        flat = flatten_population(population)
        pool = []
        for _ in range(self.pool_size):
            picks = random_utils.choices(len(flat), size=min(self.k, len(flat)),
                                         replace=False)
            pool.append(max((flat[i] for i in picks),
                            key=lambda it: it.result.score))
        # carry_over: items kept into the next generation unchanged.
        return MatingPoolResult(pool, flat)

    def mate_pool(self, mating_pool, current_generation=0):
        return [self.mating_strategy.mate(a, b, current_generation + 1)
                for a, b in zip(mating_pool[::2], mating_pool[1::2])]


class NeighbourSwap(SwapStrategy):
    """Swap the permutation-weakest feature for a column next to it: a toy
    for data whose neighbouring columns are related (e.g. spectral bands)."""

    def select_removal(self, genes, changes):
        return int(changes[0][1])

    def select_insertions(self, genes, removed, count):
        near = [j for j in (removed - 1, removed + 1)
                if 0 <= j < len(genes) and not genes[j]]
        return near[:count]


def run(name, mating_selection, mutation, **kwargs):
    optimizer = EntropyOptimizer(
        DecisionTreeClassifier, X_train, y_train, evaluator, n,
        mating_selection, mutation, initial_genes=[[i] for i in range(n)],
        config=Config(output_folder=run_folder(name), random_seed=9,
                      number_of_rounds=10_000, max_model_fits=400,
                      dump_generation_logs=False),
        **kwargs)
    optimizer.mainloop()
    print(f"\n{name}:\n" + front_table(optimizer, names))
    return optimizer


default_mating = IntersectionMatingWithWeightedRandomInformationGain
run("size_capped_mutation",
    RandomEveryoneWithEveryone(pool_size=3, mating_strategy=default_mating()),
    SizeCappedMutation(1 / n, max_size=4))
run("uniform_crossover_tournament",
    Tournament(UniformCrossover(), pool_size=6, k=3),
    RandomFlipMutationStrategy(1 / n))
# A custom mating *strategy* inside the stock selection, and vice versa:
run("tournament_with_ig_mating",
    Tournament(default_mating(), pool_size=6, k=3),
    RandomFlipMutationStrategy(1 / n))
opt = run("neighbour_swap",
          RandomEveryoneWithEveryone(pool_size=3, mating_strategy=default_mating()),
          RandomFlipMutationStrategy(1 / n), swap_strategy=NeighbourSwap())
print(f"  swaps proposed {opt.prune_stats['swap_proposed']}, "
      f"kept {opt.prune_stats['swap_kept']}")
