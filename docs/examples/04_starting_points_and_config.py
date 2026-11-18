"""Where the search starts, and the Config knobs that shape it.

The search needs at least one evaluated subset to start from. Give it
`initial_genes` (lists of feature indices), an `initial_population`
(unevaluated `Item`s grouped by size), a filter seed (example 07) or a
feature prior (example 08) -- or several of these at once.
"""
from _common import ValidationEvaluator, classification_data, front_table, run_folder

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier

from fastener import Config, EntropyOptimizer
from item import (IntersectionMatingWithWeightedRandomInformationGain, Item,
                  RandomEveryoneWithEveryone, RandomFlipMutationStrategy)

X_train, y_train, X_val, y_val, X_test, y_test, names = classification_data()
n = X_train.shape[1]
evaluator = ValidationEvaluator(X_val, y_val, balanced_accuracy_score)


def optimizer_with(config, **kwargs):
    return EntropyOptimizer(
        DecisionTreeClassifier, X_train, y_train, evaluator, n,
        RandomEveryoneWithEveryone(
            pool_size=3,
            mating_strategy=IntersectionMatingWithWeightedRandomInformationGain()),
        RandomFlipMutationStrategy(1 / n), config=config, **kwargs)


def config(name, **kwargs):
    kwargs.setdefault("number_of_rounds", 30)
    return Config(output_folder=run_folder(name), random_seed=3,
                  dump_generation_logs=False, **kwargs)


# 1. One single feature, as in the original example. The search has to grow
#    everything else by mating and mutation.
opt = optimizer_with(config("one_feature"), initial_genes=[[0]])
opt.mainloop()
print("\nStart from feature 0:\n" + front_table(opt, names))

# 2. Every single feature: the usual choice. Costs n fits up front.
opt = optimizer_with(config("all_singles"),
                     initial_genes=[[i] for i in range(n)])
opt.mainloop()
print("\nStart from every single feature:\n" + front_table(opt, names))

# 3. Subsets you already believe in, e.g. from domain knowledge or an
#    earlier study, mixed with singles.
by_name = {name: i for i, name in enumerate(names)}
known = [[by_name["mean radius"], by_name["mean texture"]],
         [by_name["worst area"], by_name["worst smoothness"],
          by_name["worst concavity"]]]
opt = optimizer_with(config("known_subsets"),
                     initial_genes=known + [[i] for i in range(n)])
opt.mainloop()
print("\nStart from two known subsets and the singles:\n"
      + front_table(opt, names))

# 4. An explicit initial population: unevaluated Items keyed by size. Here,
#    ten random subsets of three features.
rng = np.random.default_rng(0)
population = {}
for _ in range(10):
    genes = np.zeros(n, dtype=bool)
    genes[rng.choice(n, size=3, replace=False)] = True
    item = Item.from_genes(genes.tolist())
    population.setdefault(item.size, []).append(item)
opt = optimizer_with(config("random_population"),
                     initial_population=population)
opt.mainloop()
print("\nStart from ten random triples:\n" + front_table(opt, names))

# 5. Config knobs:
#    max_bucket_size         how many subsets of each size survive a generation
#    reset_to_pareto_rounds  every k rounds: prune the front, then restart the
#                            population from it; None or 0 means only once,
#                            at the last round
#    cache_fitness_function  reuse the score of a subset seen before
opt = optimizer_with(config("wide_buckets", max_bucket_size=6,
                            reset_to_pareto_rounds=10),
                     initial_genes=[[i] for i in range(n)])
opt.mainloop()
print(f"\nBuckets of 6, prune every 10 rounds: {opt.model_fits} fits\n"
      + front_table(opt, names))

opt = optimizer_with(config("no_cache", cache_fitness_function=False),
                     initial_genes=[[i] for i in range(n)])
opt.mainloop()
print(f"\nWithout the cache every repeat is refitted: {opt.model_fits} fits")

opt = optimizer_with(config("cache"), initial_genes=[[i] for i in range(n)])
opt.mainloop()
repeats = sum(c - 1 for c in opt.cache_counter.values())
print(f"With the cache: {opt.model_fits} fits, {repeats} lookups were repeats")
