"""FASTENER for regression.

The search itself does not care about the task: it fits `model()` on a
subset of columns and asks the evaluator for a score, higher is better. Two
things change for a continuous target:

1. The score. Use R^2, or a negated error (-RMSE, -MAE), so that higher
   is better.
2. The mutual information. The information-gain mating strategies estimate
   MI with a *classification* estimator, which refuses a continuous target
   ("Unknown label type: continuous"). Either override
   `use_data_information` to use `mutual_info_regression` (below), or use a
   mating strategy that needs no MI (`IntersectionMating`, `UnionMating`).

The data: Friedman #1, y = 10 sin(pi x0 x1) + 20 (x2 - 0.5)^2 + 10 x3 + 5 x4
+ noise, with 20 more features that are pure noise. A good search should
find features 0..4 and nothing else.
"""
from _common import front_table, run_folder

from functools import partial

import numpy as np
from sklearn.datasets import load_diabetes, make_friedman1
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

import random_utils
from fastener import Config, EntropyOptimizer
from item import (InformationGainSwapStrategy, IntersectionMating,
                  IntersectionMatingWithWeightedRandomInformationGain,
                  RandomEveryoneWithEveryone, RandomFlipMutationStrategy, Result)


class RegressionMating(IntersectionMatingWithWeightedRandomInformationGain):
    """FASTENER's default mating, with MI estimated for a continuous target.

    `use_data_information` is called once, before the search; whatever it
    stores in `scikit_information_gain` (one value per feature) is what the
    mating, a guided swap and a feature prior all use.
    """

    def use_data_information(self, train_data, train_target):
        self.scikit_information_gain = mutual_info_regression(
            train_data, train_target, random_state=0)


class RegressionEvaluator:
    """Score a fitted regressor on the validation split, higher is better.

    `score` is "r2", "neg_rmse" or "neg_mae". A class, not a closure, so
    that it pickles (FASTENER pickles the optimizer before the search).
    """
    SCORES = {"r2": r2_score,
              "neg_rmse": lambda y, p: -root_mean_squared_error(y, p),
              "neg_mae": lambda y, p: -mean_absolute_error(y, p)}

    def __init__(self, X_val, y_val, score="r2"):
        self.X_val, self.y_val, self.score = X_val, y_val, score

    def __call__(self, model, genes, shuffle_indices=None):
        X_eval = self.X_val[:, genes]
        if shuffle_indices:
            X_eval = X_eval.copy()
            for j in shuffle_indices:
                random_utils.shuffle(X_eval[:, j])
        return Result(self.SCORES[self.score](self.y_val, model.predict(X_eval)))


X, y = make_friedman1(n_samples=2000, n_features=25, noise=1.0, random_state=0)
X_rest, X_test, y_rest, y_test = train_test_split(X, y, test_size=0.25,
                                                  random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_rest, y_rest,
                                                  test_size=1 / 3, random_state=0)
n = X.shape[1]
names = [f"x{i}" for i in range(n)]


def search(name, model, evaluator, mating_strategy=None, **kwargs):
    optimizer = EntropyOptimizer(
        model, X_train, y_train, evaluator, n,
        RandomEveryoneWithEveryone(
            pool_size=3, mating_strategy=mating_strategy or RegressionMating()),
        RandomFlipMutationStrategy(1 / n),
        config=Config(output_folder=run_folder(name), random_seed=8,
                      number_of_rounds=10_000,
                      max_model_fits=kwargs.pop("max_model_fits", 500),
                      dump_generation_logs=False),
        **kwargs)
    optimizer.mainloop()
    print(f"\n{name}:")
    print(front_table(optimizer, names))
    return optimizer


singles = [[i] for i in range(n)]

# 1. A regression tree, R^2 on validation, regression MI in the mating.
opt = search("tree_r2", partial(DecisionTreeRegressor, min_samples_leaf=5,
                                random_state=0),
             RegressionEvaluator(X_val, y_val, "r2"), initial_genes=singles)

# 2. Negative RMSE instead of R^2: the same ordering, in the target's units.
search("tree_neg_rmse", partial(DecisionTreeRegressor, min_samples_leaf=5,
                                random_state=0),
       RegressionEvaluator(X_val, y_val, "neg_rmse"), initial_genes=singles)

# 3. No mutual information at all: plain intersection mating.
search("tree_plain_intersection",
       partial(DecisionTreeRegressor, min_samples_leaf=5, random_state=0),
       RegressionEvaluator(X_val, y_val), mating_strategy=IntersectionMating(),
       initial_genes=singles)

# 4. A small forest with the guided swap (it reuses RegressionMating's MI).
search("forest_swap",
       partial(RandomForestRegressor, n_estimators=30, min_samples_leaf=3,
               n_jobs=-1, random_state=0),
       RegressionEvaluator(X_val, y_val), initial_genes=singles,
       swap_strategy=InformationGainSwapStrategy(), max_model_fits=250)

# 5. Seeds. mRMR's relevance is a classification estimator too, so it does
#    not apply to a continuous target; Boruta ranks by the regressor's
#    feature_importances_ and works (needs `pip install Boruta`).
try:
    opt_b = search("tree_boruta_seed",
                   partial(DecisionTreeRegressor, min_samples_leaf=5),
                   RegressionEvaluator(X_val, y_val), initial_genes=[[0]],
                   seed_filter="boruta", seed_size=8, boruta_max_iter=30)
    print("Boruta seed order:", opt_b.seed_ranking)
except ImportError as e:
    print(f"\nskipping the Boruta seed: {e}")

# 6. Pick a subset and judge it once on the test split: the smallest one
#    within 0.01 R^2 of the best on validation.
best = max(it.result.score for it in opt.pareto_front.values())
size, chosen = min((s, it) for s, it in opt.pareto_front.items()
                   if it.result.score >= best - 0.01)
cols = np.flatnonzero(chosen.genes)
model = DecisionTreeRegressor(min_samples_leaf=5, random_state=0)
model.fit(X_rest[:, cols], y_rest)
print(f"\nChosen {[names[i] for i in cols]}: test R^2 "
      f"{r2_score(y_test, model.predict(X_test[:, cols])):.3f} "
      f"(true relevant features: x0..x4)")

# 7. A real data set with named features: diabetes progression (442 x 10),
#    with a linear model.
d = load_diabetes()
Xd_train, Xd_val, yd_train, yd_val = train_test_split(d.data, d.target,
                                                      test_size=0.3,
                                                      random_state=0)
nd = d.data.shape[1]
opt_d = EntropyOptimizer(
    RidgeCV, Xd_train, yd_train, RegressionEvaluator(Xd_val, yd_val), nd,
    RandomEveryoneWithEveryone(pool_size=3, mating_strategy=RegressionMating()),
    RandomFlipMutationStrategy(1 / nd), initial_genes=[[i] for i in range(nd)],
    config=Config(output_folder=run_folder("diabetes"), random_seed=8,
                  number_of_rounds=40, dump_generation_logs=False))
opt_d.mainloop()
print("\ndiabetes, ridge regression, validation R^2:")
print(front_table(opt_d, list(d.feature_names)))
