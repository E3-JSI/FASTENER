# Regression

FASTENER's search does not depend on the task. It fits `model()` on a subset
of columns, and your evaluator turns the fitted model into a score. A
regression problem needs three changes, and the first two are required.

## 1. A score where higher is better

Return R², or the negative of an error:

```python
class RegressionEvaluator:
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
```

R² and −RMSE order subsets identically on a fixed validation set, so they
give the same front. Only the printed numbers differ. −MAE can order
subsets differently.

## 2. Mutual information for a continuous target

The information-gain mating strategies estimate MI with a classification
estimator, which rejects a continuous target before the first generation:

```
ValueError: Unknown label type: continuous. Maybe you are trying to fit a
classifier, which expects discrete classes on a regression target with
continuous values.
```

There are two ways round it.

**Estimate MI for regression (recommended).** Override
`use_data_information`. Whatever it stores in `scikit_information_gain`, one
value per feature, is what the mating uses. The guided swap and a feature
prior use it too.

```python
from sklearn.feature_selection import mutual_info_regression

class RegressionMating(IntersectionMatingWithWeightedRandomInformationGain):
    def use_data_information(self, train_data, train_target):
        self.scikit_information_gain = mutual_info_regression(
            train_data, train_target, random_state=0)

mating = RandomEveryoneWithEveryone(pool_size=3, mating_strategy=RegressionMating())
```

**Skip MI.** `IntersectionMating()` and `UnionMating()` do not use it. This
drops the "entropy" in FASTENER, but the rest of the search, including
pruning by permutation importance, is unchanged.

## 3. A regressor

Any estimator with `fit` and `predict`, for example
`partial(DecisionTreeRegressor, min_samples_leaf=5)`,
`partial(RandomForestRegressor, n_estimators=30)` or `RidgeCV`. For the
Boruta seed it also needs `feature_importances_`, so use a tree or a forest.

## Which options work for regression

| Option | Regression |
|---|---|
| `IntersectionMatingWith…InformationGain` | With the `use_data_information` override |
| `IntersectionMating`, `UnionMating` | Yes, as is |
| `InformationGainSwapStrategy` | Yes, when the mating strategy has the regression MI (it reuses the vector). Otherwise it falls back to the classification estimator and fails. |
| `RandomSwapStrategy` | Yes |
| `seed_filter="boruta"` | Yes, with a regressor that has `feature_importances_` |
| `seed_filter="mrmr"` | No: its relevance term is `mutual_info_classif`. Rank the features yourself and pass the nested prefixes as `initial_genes` (below). |
| `feature_prior` | Yes, when the mating strategy has the regression MI |
| `subset_proposer` | Yes |

An mRMR-like seed for regression, built by hand:

```python
import seeding
relevance = mutual_info_regression(X_train, y_train, random_state=0)
order = numpy.argsort(-relevance)          # relevance only, no redundancy term
initial_genes = seeding.nested_prefixes(order, 10)
```

## The complete example

[examples/09_regression.py](examples/09_regression.py) runs the variants
above on Friedman #1:

y = 10 sin(π x0 x1) + 20 (x2 − 0.5)² + 10 x3 + 5 x4 + noise

The data set has 20 more features that are pure noise. With a regression tree
and R² on the validation split, the front is:

```
  1 features  score 0.1330  ['x3']
  2 features  score 0.3861  ['x0', 'x3']
  3 features  score 0.6418  ['x0', 'x1', 'x3']
  4 features  score 0.7062  ['x0', 'x1', 'x2', 'x3']
  5 features  score 0.7497  ['x0', 'x1', 'x2', 'x3', 'x4']
  6 features  score 0.7511  ['x0', 'x1', 'x2', 'x3', 'x4', 'x5']
```

The smallest subset within 0.01 R² of the best is exactly the five relevant
features (test R² 0.775). The step from five features to six gains 0.001 and
picks up a noise feature. That is the overfitting to the validation split
that a held-out test catches. The script ends with ridge regression on the
diabetes data set, which has named features.
