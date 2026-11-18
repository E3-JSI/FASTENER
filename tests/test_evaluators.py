"""evaluators.Evaluator: the ready-made metrics, permutation calls, pickling."""
import pickle

import numpy as np
import pytest
from sklearn import metrics as skm
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

import random_utils
from evaluators import METRICS, Evaluator
from item import Result

from conftest import N_FEATURES, genes_from


@pytest.fixture(scope="module")
def fitted(small_data):
    X_train, y_train, X_val, y_val = small_data
    genes = genes_from([0, 1, 2])
    model = DecisionTreeClassifier(max_depth=3, random_state=0)
    model.fit(X_train[:, genes], y_train)
    return model, genes, X_val, y_val


def _mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def _penalised_accuracy(y_true, y_pred):
    return skm.accuracy_score(y_true, y_pred) - 0.1


@pytest.mark.parametrize("name, expected", [
    ("f1", lambda y, p, pr: skm.f1_score(y, p, average="weighted")),
    ("precision", lambda y, p, pr: skm.precision_score(y, p, average="weighted")),
    ("recall", lambda y, p, pr: skm.recall_score(y, p, average="weighted")),
    ("accuracy", lambda y, p, pr: skm.accuracy_score(y, p)),
    ("balanced_accuracy", lambda y, p, pr: skm.balanced_accuracy_score(y, p)),
    ("matthews", lambda y, p, pr: skm.matthews_corrcoef(y, p)),
    ("roc_auc", lambda y, p, pr: skm.roc_auc_score(y, pr[:, 1])),
    ("log_loss", lambda y, p, pr: -skm.log_loss(y, pr)),
])
def test_classification_metrics_match_scikit_learn(fitted, name, expected):
    model, genes, X_val, y_val = fitted
    X = X_val[:, genes]
    result = Evaluator(X_val, y_val, name)(model, genes)
    assert isinstance(result, Result)
    assert result.score == pytest.approx(
        expected(y_val, model.predict(X), model.predict_proba(X)))


def test_regression_metrics_are_higher_is_better(small_data):
    X_train, _, X_val, _ = small_data
    rng = np.random.RandomState(1)
    y_train = X_train[:, 0] * 2 + rng.normal(size=len(X_train))
    y_val = X_val[:, 0] * 2 + rng.normal(size=len(X_val))
    genes = genes_from([0, 3])
    model = LinearRegression().fit(X_train[:, genes], y_train)
    pred = model.predict(X_val[:, genes])
    score = {m: Evaluator(X_val, y_val, m)(model, genes).score
             for m in ("r2", "mae", "mse", "rmse")}
    assert score["r2"] == pytest.approx(skm.r2_score(y_val, pred))
    assert score["mae"] == pytest.approx(-skm.mean_absolute_error(y_val, pred))
    assert score["mse"] == pytest.approx(-skm.mean_squared_error(y_val, pred))
    assert score["rmse"] == pytest.approx(-np.sqrt(-score["mse"]))


def test_keyword_arguments_override_the_defaults(fitted):
    model, genes, X_val, y_val = fitted
    pred = model.predict(X_val[:, genes])
    for kwargs in ({"average": "macro"}, {"average": "binary", "pos_label": 0}):
        score = Evaluator(X_val, y_val, "f1", **kwargs)(model, genes).score
        assert score == pytest.approx(skm.f1_score(y_val, pred, **kwargs))


def test_default_metric_is_weighted_f1(fitted):
    model, genes, X_val, y_val = fitted
    assert Evaluator(X_val, y_val)(model, genes).score == pytest.approx(
        skm.f1_score(y_val, model.predict(X_val[:, genes]), average="weighted"))


def test_a_function_metric_and_its_direction(fitted):
    model, genes, X_val, y_val = fitted
    pred = model.predict(X_val[:, genes])
    up = Evaluator(X_val, y_val, _penalised_accuracy)(model, genes).score
    assert up == pytest.approx(skm.accuracy_score(y_val, pred) - 0.1)
    down = Evaluator(X_val, y_val, _mae, greater_is_better=False)(model, genes)
    assert down.score == pytest.approx(-_mae(y_val, pred))


def test_a_function_metric_can_take_probabilities(fitted):
    model, genes, X_val, y_val = fitted
    ev = Evaluator(X_val, y_val, skm.log_loss, greater_is_better=False,
                   use_proba=True)
    assert ev(model, genes).score == pytest.approx(
        -skm.log_loss(y_val, model.predict_proba(X_val[:, genes])))


def test_overriding_score(fitted):
    model, genes, X_val, y_val = fitted

    class Margin(Evaluator):
        def score(self, model, X_eval):
            return float(np.mean(np.max(model.predict_proba(X_eval), axis=1)))

    expected = np.mean(np.max(model.predict_proba(X_val[:, genes]), axis=1))
    assert Margin(X_val, y_val)(model, genes).score == pytest.approx(expected)


def test_shuffling_matches_the_hand_written_evaluator(fitted):
    """The same columns of a copy, shuffled with random_utils in order: the
    random stream, and so the search, is the one a hand-written evaluator
    would give."""
    model, genes, X_val, y_val = fitted
    random_utils.seed(7)
    got = Evaluator(X_val, y_val)(model, genes, [0, 2]).score
    random_utils.seed(7)
    X = X_val[:, genes].copy()
    for j in (0, 2):
        random_utils.shuffle(X[:, j])
    want = skm.f1_score(y_val, model.predict(X), average="weighted")
    assert got == pytest.approx(want)


def test_shuffling_leaves_the_validation_data_alone(fitted):
    model, genes, X_val, y_val = fitted
    before = X_val.copy()
    Evaluator(X_val, y_val)(model, genes, [0, 1, 2])
    np.testing.assert_array_equal(X_val, before)


def test_it_pickles(fitted):
    model, genes, X_val, y_val = fitted
    for ev in (Evaluator(X_val, y_val, "roc_auc"),
               Evaluator(X_val, y_val, _mae, greater_is_better=False)):
        back = pickle.loads(pickle.dumps(ev))
        assert back(model, genes).score == ev(model, genes).score


def test_bad_arguments(small_data):
    _, _, X_val, y_val = small_data
    with pytest.raises(ValueError, match="unknown metric"):
        Evaluator(X_val, y_val, "f2")
    with pytest.raises(ValueError, match="function metrics"):
        Evaluator(X_val, y_val, "f1", greater_is_better=False)
    with pytest.raises(TypeError):
        Evaluator(X_val, y_val, 3)


def test_every_named_metric_is_listed():
    assert set(METRICS) == {"f1", "precision", "recall", "accuracy",
                            "balanced_accuracy", "matthews", "roc_auc",
                            "log_loss", "r2", "mae", "mse", "rmse"}
    assert N_FEATURES >= 3


class _HandWrittenAccuracy:
    """The evaluator users wrote before evaluators.py existed."""

    def __init__(self, X, y):
        self.X, self.y = X, y

    def __call__(self, model, genes, shuffle_indices=None):
        data = self.X[:, genes]
        if shuffle_indices:
            data = data.copy()
            for j in shuffle_indices:
                random_utils.shuffle(data[:, j])
        return Result(float((model.predict(data) == self.y).mean()))


def _front(small_data, evaluator, name):
    from fastener import Config, EntropyOptimizer
    from item import (IntersectionMatingWithWeightedRandomInformationGain,
                      RandomEveryoneWithEveryone, RandomFlipMutationStrategy)
    X, y, _, _ = small_data
    optimizer = EntropyOptimizer(
        DecisionTreeClassifier, X, y, evaluator, N_FEATURES,
        RandomEveryoneWithEveryone(
            pool_size=3,
            mating_strategy=IntersectionMatingWithWeightedRandomInformationGain()),
        RandomFlipMutationStrategy(1 / N_FEATURES),
        initial_genes=[[i] for i in range(N_FEATURES)],
        config=Config(output_folder=name, random_seed=3, number_of_rounds=8,
                      dump_generation_logs=False))
    optimizer.mainloop()
    return {s: (list(it.genes), it.result.score)
            for s, it in optimizer.pareto_front.items()}, optimizer.model_fits


def test_a_search_is_the_same_as_with_a_hand_written_evaluator(small_data,
                                                                run_dir):
    _, _, X_val, y_val = small_data
    assert _front(small_data, Evaluator(X_val, y_val, "accuracy"), "lib") == \
        _front(small_data, _HandWrittenAccuracy(X_val, y_val), "hand")
