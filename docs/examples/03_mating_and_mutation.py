"""Mating and mutation strategies, compared at an equal budget of model fits.

A generation draws a mating pool from the population (the *selection*
strategy), mates every pair in it (the *mating* strategy) and flips genes of
every child (the *mutation* strategy). All variants below run until they
have fitted 400 models, so they are compared at equal compute.
"""
from _common import ValidationEvaluator, classification_data, run_folder

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier

from fastener import Config, EntropyOptimizer
from item import (IntersectionMating, IntersectionMatingWithInformationGain,
                  IntersectionMatingWithWeightedRandomInformationGain,
                  RandomEveryoneWithEveryone, RandomFlipMutationStrategy,
                  UnionMating)

X_train, y_train, X_val, y_val, *_ = classification_data()
n = X_train.shape[1]
evaluator = ValidationEvaluator(X_val, y_val, balanced_accuracy_score)


def half_the_difference(k):
    """How many of the parents' differing genes a child gets back: here a
    half, rounded down (the default is half + 1, capped at k)."""
    return k // 2


def gentler_scaling(mi):
    """Weights for the weighted random sample, from the mutual information
    of each candidate gene. The default is log1p applied three times, which
    flattens them heavily; a single log1p keeps more of the contrast."""
    return np.log1p(np.asarray(mi))


VARIANTS = {
    # The default since 2026-09-30: intersection of the parents, plus the
    # top-k differing genes by mutual information, plus k more sampled by it.
    "weighted_random_ig (default)": RandomEveryoneWithEveryone(
        pool_size=3,
        mating_strategy=IntersectionMatingWithWeightedRandomInformationGain()),
    # Only the weighted sample, without the top-k genes.
    "weighted_random_ig, sample only": RandomEveryoneWithEveryone(
        pool_size=3,
        mating_strategy=IntersectionMatingWithWeightedRandomInformationGain(
            include_top_genes=False)),
    # Custom number of genes and custom weight scaling.
    "weighted_random_ig, custom k and scaling": RandomEveryoneWithEveryone(
        pool_size=3,
        mating_strategy=IntersectionMatingWithWeightedRandomInformationGain(
            number=half_the_difference, scaling=gentler_scaling)),
    # Deterministic: intersection plus the top-k differing genes by MI.
    "intersection + top ig": RandomEveryoneWithEveryone(
        pool_size=3, mating_strategy=IntersectionMatingWithInformationGain()),
    # Unguided crossovers.
    "intersection": RandomEveryoneWithEveryone(
        pool_size=3, mating_strategy=IntersectionMating()),
    "union": RandomEveryoneWithEveryone(
        pool_size=3, mating_strategy=UnionMating()),
    # A bigger pool: 5 items give 10 pairs per generation instead of 3.
    "weighted_random_ig, pool of 5": RandomEveryoneWithEveryone(
        pool_size=5,
        mating_strategy=IntersectionMatingWithWeightedRandomInformationGain()),
    # The 2020 code exactly: an item may be mated with itself, and the top
    # genes are always included. Use both flags to reproduce old results.
    "as published in 2020": RandomEveryoneWithEveryone(
        pool_size=3, allow_self_mating=True,
        mating_strategy=IntersectionMatingWithWeightedRandomInformationGain(
            include_top_genes=True)),
}

MUTATIONS = {
    "flip 1/n": RandomFlipMutationStrategy(1 / n),
    "flip 3/n": RandomFlipMutationStrategy(3 / n),
}


def run(name, mating, mutation):
    optimizer = EntropyOptimizer(
        DecisionTreeClassifier, X_train, y_train, evaluator, n,
        mating, mutation, initial_genes=[[i] for i in range(n)],
        config=Config(output_folder=run_folder("mating"), random_seed=7,
                      number_of_rounds=10_000, max_model_fits=400,
                      dump_generation_logs=False))
    optimizer.mainloop()
    front = optimizer.pareto_front
    best_size, best = max(front.items(), key=lambda kv: kv[1].result.score)
    print(f"{name:42s} rounds {optimizer.rounds_completed:4d}  "
          f"front sizes {sorted(front)}  best {best.result.score:.4f} "
          f"with {best_size}")


for name, mating in VARIANTS.items():
    run(name, mating, MUTATIONS["flip 1/n"])
for name, mutation in MUTATIONS.items():
    run(f"default mating, {name}",
        RandomEveryoneWithEveryone(
            pool_size=3,
            mating_strategy=IntersectionMatingWithWeightedRandomInformationGain()),
        mutation)
