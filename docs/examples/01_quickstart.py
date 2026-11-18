"""Quickstart: select features for a classifier on the breast-cancer data.

The minimum FASTENER needs: a model factory, training data, an evaluator
that scores a fitted model on validation data, a mating and a mutation
strategy, and somewhere to start.
"""
from _common import run_folder, front_table  # noqa: F401  (sets sys.path)

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from fastener import Config, EntropyOptimizer
from item import (IntersectionMatingWithWeightedRandomInformationGain,
                  RandomEveryoneWithEveryone, RandomFlipMutationStrategy,
                  Result)
import random_utils

data = load_breast_cancer()
X, y = data.data, data.target
names = [str(n) for n in data.feature_names]

# Three splits: FASTENER searches on train + validation; the test split is
# kept for judging the subset you finally pick.
X_rest, X_test, y_rest, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(
    X_rest, y_rest, test_size=1 / 3, stratify=y_rest, random_state=0)


def evaluator(model, genes, shuffle_indices=None):
    """Score `model` (fitted on X_train[:, genes]) on the validation split.

    FASTENER also calls it with `shuffle_indices` to measure permutation
    importance: those are positions among the *selected* columns, whose
    values must be shuffled before predicting.
    """
    X_eval = X_val[:, genes]
    if shuffle_indices:
        X_eval = X_eval.copy()
        for j in shuffle_indices:
            random_utils.shuffle(X_eval[:, j])
    return Result(f1_score(y_val, model.predict(X_eval), average="weighted"))


n_features = X.shape[1]
optimizer = EntropyOptimizer(
    model=DecisionTreeClassifier,          # called with no arguments
    train_data=X_train, train_target=y_train,
    evaluator=evaluator,
    number_of_genes=n_features,
    mating_selection_strategy=RandomEveryoneWithEveryone(
        pool_size=3,
        mating_strategy=IntersectionMatingWithWeightedRandomInformationGain()),
    mutation_strategy=RandomFlipMutationStrategy(1 / n_features),
    initial_genes=[[i] for i in range(n_features)],  # every single feature
    config=Config(output_folder=run_folder("quickstart"), random_seed=2020,
                  number_of_rounds=50, dump_generation_logs=False),
)
optimizer.mainloop()

print("\nPareto front (validation weighted F1):")
print(front_table(optimizer, names))
print(f"\n{optimizer.model_fits} model fits, "
      f"{optimizer.rounds_completed} rounds, stopped on {optimizer.stop_reason}")

# Pick the smallest subset within 0.01 of the best validation score, then
# judge it once on the test split.
best = max(item.result.score for item in optimizer.pareto_front.values())
size, chosen = min((s, it) for s, it in optimizer.pareto_front.items()
                   if it.result.score >= best - 0.01)
cols = np.flatnonzero(chosen.genes)
model = DecisionTreeClassifier().fit(X_rest[:, cols], y_rest)
test_f1 = f1_score(y_test, model.predict(X_test[:, cols]), average="weighted")
print(f"Chosen: {size} features {[names[i] for i in cols]}; "
      f"test F1 {test_f1:.4f}")
