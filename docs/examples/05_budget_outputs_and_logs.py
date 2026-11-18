"""Stopping on a fit budget, reading what a run did, and its log files.

`Config.max_model_fits` stops the search after that many model fits, which
is how two settings are compared at equal compute. After `mainloop` the
optimizer holds the front and counters; with `dump_generation_logs=True`
every generation is also pickled to log/<folder>/generation_N.pickle.
"""
from _common import ValidationEvaluator, classification_data, front_table, run_folder

import glob
import os
import pickle

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier

from fastener import Config, EntropyOptimizer
from item import (IntersectionMatingWithWeightedRandomInformationGain,
                  RandomEveryoneWithEveryone, RandomFlipMutationStrategy)

X_train, y_train, X_val, y_val, X_test, y_test, names = classification_data()
n = X_train.shape[1]

config = Config(output_folder=run_folder("budget"), random_seed=11,
                number_of_rounds=100_000,   # high, so the budget stops it
                max_model_fits=300,         # must exceed the initial population
                reset_to_pareto_rounds=5,
                dump_generation_logs=True)  # the default; False for batches
optimizer = EntropyOptimizer(
    DecisionTreeClassifier, X_train, y_train,
    ValidationEvaluator(X_val, y_val, balanced_accuracy_score), n,
    RandomEveryoneWithEveryone(
        pool_size=3,
        mating_strategy=IntersectionMatingWithWeightedRandomInformationGain()),
    RandomFlipMutationStrategy(1 / n),
    initial_genes=[[i] for i in range(n)], config=config)
optimizer.mainloop()

# What the run did.
print("\nFront:\n" + front_table(optimizer, names))
print(f"stop_reason       {optimizer.stop_reason}")       # "fit_budget" or "rounds"
print(f"model_fits        {optimizer.model_fits}")
print(f"rounds_completed  {optimizer.rounds_completed}")
print(f"prune_stats       {dict(optimizer.prune_stats)}")
print(f"cache entries     {len(optimizer.cache_data)}")

# Each front entry is an EvalItem: genes (a bool per feature), size,
# result.score, generation (when it was made) and the parents' genes.
size, item = max(optimizer.pareto_front.items(),
                 key=lambda kv: kv[1].result.score)
print(f"best: size {size}, score {item.result.score:.4f}, "
      f"made in generation {item.generation}, "
      f"features {np.flatnonzero(item.genes).tolist()}")

# The log folder: experiment.pickle (the optimizer before the search) and
# one pickle per generation.
folder = config.output_folder          # "log/<output_folder>"
files = sorted(glob.glob(os.path.join(folder, "generation_*.pickle")),
               key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
print(f"\n{folder}: experiment.pickle and {len(files)} generation logs")

with open(files[-1], "rb") as f:
    log = pickle.load(f)               # a fastener.LogData
print(f"generation {log.generation}: population sizes "
      f"{sorted(log.population)}, front sizes {sorted(log.front)}, "
      f"{len(log.cache_data)} cached results")

# The front as it grew, generation by generation.
for path in files[::10]:
    with open(path, "rb") as f:
        log = pickle.load(f)
    best = max(it.result.score for it in log.front.values())
    print(f"  gen {log.generation:3d}: best {best:.4f}, front {sorted(log.front)}")

with open(os.path.join(folder, "experiment.pickle"), "rb") as f:
    before = pickle.load(f)            # an EntropyOptimizer
print(f"experiment.pickle: initial population sizes {sorted(before.population)}")
