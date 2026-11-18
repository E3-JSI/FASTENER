"""Ready-made evaluators: score a fitted model on validation data.

FASTENER asks an evaluator for every score it uses:

    evaluator(model, genes, shuffle_indices=None) -> Result

`Evaluator` is one such evaluator for the common metrics. It holds the
validation data, predicts with the model on the selected columns, and returns
the metric as a `Result`, higher being better. When FASTENER prunes the front,
it passes `shuffle_indices` to measure permutation importance; `Evaluator`
shuffles those columns of a copy with `random_utils.shuffle`, so runs repeat
with the same seed.

    from evaluators import Evaluator

    Evaluator(X_val, y_val)                          # weighted F1
    Evaluator(X_val, y_val, "accuracy")
    Evaluator(X_val, y_val, "f1", average="macro")   # passed to f1_score
    Evaluator(X_val, y_val, "roc_auc")               # uses predict_proba
    Evaluator(X_val, y_val, "r2")                    # regression
    Evaluator(X_val, y_val, "rmse")                  # scored as -RMSE

Your own metric: pass a module-level function `metric(y_true, y_pred)`, with
`greater_is_better=False` for an error. Your own scoring of the model, for
example from probabilities: subclass `Evaluator` and override `score`.

`Evaluator` is a class rather than a closure because FASTENER pickles the
optimizer, evaluator included, before the search.
"""
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from sklearn import metrics as skm

import random_utils
from item import Result


def _rmse(y_true, y_pred, **kwargs):
    # root_mean_squared_error only exists from scikit-learn 1.4.
    return float(np.sqrt(skm.mean_squared_error(y_true, y_pred, **kwargs)))


def _roc_auc(y_true, y_score, **kwargs):
    # Binary: the positive class's probability. Multiclass: one-vs-rest.
    y_score = np.asarray(y_score)
    if y_score.ndim == 2 and y_score.shape[1] == 2:
        y_score = y_score[:, 1]
    elif y_score.ndim == 2:
        kwargs.setdefault("multi_class", "ovr")
    return skm.roc_auc_score(y_true, y_score, **kwargs)


# name -> (function, greater_is_better, uses predict_proba, default kwargs)
METRICS: Dict[str, tuple] = {
    # classification
    "f1": (skm.f1_score, True, False, {"average": "weighted"}),
    "precision": (skm.precision_score, True, False,
                  {"average": "weighted", "zero_division": 0}),
    "recall": (skm.recall_score, True, False,
               {"average": "weighted", "zero_division": 0}),
    "accuracy": (skm.accuracy_score, True, False, {}),
    "balanced_accuracy": (skm.balanced_accuracy_score, True, False, {}),
    "matthews": (skm.matthews_corrcoef, True, False, {}),
    "roc_auc": (_roc_auc, True, True, {}),
    "log_loss": (skm.log_loss, False, True, {}),
    # regression
    "r2": (skm.r2_score, True, False, {}),
    "mae": (skm.mean_absolute_error, False, False, {}),
    "mse": (skm.mean_squared_error, False, False, {}),
    "rmse": (_rmse, False, False, {}),
}

Metric = Union[str, Callable[..., float]]


class Evaluator:
    """Score a fitted model on validation data with a metric.

    Args:
        X_val, y_val: the validation data. FASTENER fits on the training
            data it is given and never passes data to the evaluator.
        metric: a name from `METRICS` ("f1", "precision", "recall",
            "accuracy", "balanced_accuracy", "matthews", "roc_auc",
            "log_loss", "r2", "mae", "mse", "rmse"), or a function
            `metric(y_true, y_pred) -> float`. A function must be defined
            at module level so that it pickles.
        greater_is_better: only for a function metric. With False the score
            is negated, so that an error such as MAE can be minimised.
            Named metrics know their own direction: "mae", "mse", "rmse"
            and "log_loss" are returned negated.
        use_proba: only for a function metric: pass it
            `model.predict_proba(X)` instead of `model.predict(X)`.
        **metric_kwargs: passed to the metric, overriding its defaults, e.g.
            `average="macro"` or `pos_label=0` for "f1".

    Calling it gives `Result(score)`, higher is better.
    """

    def __init__(self, X_val, y_val, metric: Metric = "f1", *,
                 greater_is_better: Optional[bool] = None,
                 use_proba: Optional[bool] = None, **metric_kwargs) -> None:
        if isinstance(metric, str):
            if metric not in METRICS:
                raise ValueError(f"unknown metric {metric!r}; choose one of "
                                 f"{sorted(METRICS)} or pass a function")
            if greater_is_better is not None or use_proba is not None:
                raise ValueError(
                    f"{metric!r} knows its direction and input; "
                    "greater_is_better and use_proba are for function metrics")
            _, greater_is_better, use_proba, defaults = METRICS[metric]
            metric_kwargs = {**defaults, **metric_kwargs}
        elif not callable(metric):
            raise TypeError("metric must be a name or a function")
        self.X_val = X_val
        self.y_val = y_val
        self.metric = metric
        self.greater_is_better = True if greater_is_better is None \
            else greater_is_better
        self.use_proba = bool(use_proba)
        self.metric_kwargs = metric_kwargs

    def __call__(self, model, genes: List[bool],
                 shuffle_indices: Optional[List[int]] = None) -> Result:
        X_eval = self.X_val[:, genes]
        if shuffle_indices:
            X_eval = X_eval.copy()
            for j in shuffle_indices:
                random_utils.shuffle(X_eval[:, j])
        return Result(float(self.score(model, X_eval)))

    def score(self, model, X_eval) -> float:
        """The score of `model` on `X_eval` (the selected, possibly shuffled
        validation columns), higher is better. Override this to score in
        your own way; `self.y_val` holds the targets."""
        function = METRICS[self.metric][0] if isinstance(self.metric, str) \
            else self.metric
        prediction = model.predict_proba(X_eval) if self.use_proba \
            else model.predict(X_eval)
        value = function(self.y_val, prediction, **self.metric_kwargs)
        return value if self.greater_is_better else -value

    def __repr__(self) -> str:
        if type(self).score is not Evaluator.score:   # own scoring: no metric
            return f"{type(self).__name__}()"
        name = self.metric if isinstance(self.metric, str) \
            else getattr(self.metric, "__name__", repr(self.metric))
        extra = "".join(f", {k}={v!r}" for k, v in self.metric_kwargs.items())
        return f"Evaluator({name!r}{extra})"
