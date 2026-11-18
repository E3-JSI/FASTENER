"""Any model, any score.

`model` is a factory that FASTENER calls with no arguments, once per fit,
so set hyperparameters with `functools.partial` or a module-level function.
The evaluator decides what "good" means: it returns a `Result` whose `score`
is higher for better subsets.

`evaluators.Evaluator` covers the usual metrics: "f1", "precision",
"recall", "accuracy", "balanced_accuracy", "matthews", "roc_auc",
"log_loss", and for regression "r2", "mae", "mse", "rmse" (errors are
negated, so higher is still better). For anything else there are three ways
to write your own evaluator, from least to most code: a metric function, a
subclass that overrides `score`, or a callable from scratch (5-7 below).

Model and evaluator must pickle: FASTENER writes the whole optimizer to
log/<folder>/experiment.pickle before the search starts, so a lambda or a
closure as `model` or `evaluator` fails with "Can't pickle local object".
"""
from functools import partial

from _common import classification_data, front_table, run_folder

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import random_utils
from evaluators import Evaluator
from fastener import Config, EntropyOptimizer
from item import (IntersectionMatingWithWeightedRandomInformationGain,
                  RandomEveryoneWithEveryone, RandomFlipMutationStrategy,
                  Result)

X_train, y_train, X_val, y_val, X_test, y_test, names = classification_data()
n = X_train.shape[1]


def search(name, model, evaluator, rounds=20):
    optimizer = EntropyOptimizer(
        model, X_train, y_train, evaluator, n,
        RandomEveryoneWithEveryone(
            pool_size=3,
            mating_strategy=IntersectionMatingWithWeightedRandomInformationGain()),
        RandomFlipMutationStrategy(1 / n),
        initial_genes=[[i] for i in range(n)],
        config=Config(output_folder=run_folder(name), random_seed=1,
                      number_of_rounds=rounds, dump_generation_logs=False))
    optimizer.mainloop()
    print(f"\n{name} ({evaluator!r}):")
    print(front_table(optimizer, names))
    return optimizer


# 1. A tree with hyperparameters, scored by balanced accuracy.
search("tree_balanced_accuracy",
       partial(DecisionTreeClassifier, max_depth=5, class_weight="balanced",
               random_state=0),
       Evaluator(X_val, y_val, "balanced_accuracy"))

# 2. A small random forest (slower: keep it small while searching), scored
#    by the recall of class 0 (malignant): keyword arguments go to the
#    metric, here sklearn's recall_score.
search("forest_recall",
       partial(RandomForestClassifier, n_estimators=30, n_jobs=-1,
               random_state=0),
       Evaluator(X_val, y_val, "recall", average="binary", pos_label=0),
       rounds=10)


# 3. A pipeline: scale, then logistic regression. A module-level function
#    builds a new pipeline for every fit (a lambda would not pickle).
def scaled_logistic():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))


search("logistic_f1", scaled_logistic, Evaluator(X_val, y_val, "f1"))

# 4. A score from probabilities: "roc_auc" and "log_loss" call
#    predict_proba instead of predict.
search("tree_auc", partial(DecisionTreeClassifier, max_depth=4, random_state=0),
       Evaluator(X_val, y_val, "roc_auc"))


# 5. Your own metric: any module-level function metric(y_true, y_pred).
#    Here F2, which weighs recall twice as much as precision. For an error,
#    add greater_is_better=False; for probabilities, use_proba=True.
def f2(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2, pos_label=0)


search("tree_f2", partial(DecisionTreeClassifier, random_state=0),
       Evaluator(X_val, y_val, f2))


# 6. Your own scoring of the model: subclass Evaluator and override
#    score(model, X_eval). X_eval is the selected validation columns,
#    already shuffled when FASTENER measures permutation importance, and
#    self.y_val holds the targets. Here: accuracy, but only counting
#    predictions the model is at least 80% sure of.
class ConfidentAccuracy(Evaluator):
    def score(self, model, X_eval):
        proba = model.predict_proba(X_eval)
        sure = proba.max(axis=1) >= 0.8
        right = model.classes_[proba.argmax(axis=1)] == self.y_val
        return float(np.mean(right & sure))


search("tree_confident", partial(DecisionTreeClassifier, max_depth=4,
                                 random_state=0),
       ConfidentAccuracy(X_val, y_val))


# 7. An evaluator from scratch: any picklable callable with this signature.
#    It must shuffle the columns at `shuffle_indices` (positions among the
#    selected columns) of a copy, with random_utils so runs repeat, and
#    return Result(score), higher is better. Here a penalised score that
#    trades accuracy against subset size inside the score itself. The front
#    already gives one subset per size, so this is rarely needed.
class PenalisedAccuracy:
    def __init__(self, X_val, y_val, penalty=0.002):
        self.X_val, self.y_val, self.penalty = X_val, y_val, penalty

    def __call__(self, model, genes, shuffle_indices=None):
        X_eval = self.X_val[:, genes]
        if shuffle_indices:
            X_eval = X_eval.copy()
            for j in shuffle_indices:
                random_utils.shuffle(X_eval[:, j])
        accuracy = np.mean(model.predict(X_eval) == self.y_val)
        return Result(float(accuracy - self.penalty * sum(genes)))

    def __repr__(self):
        return f"PenalisedAccuracy(penalty={self.penalty})"


search("tree_penalised", partial(DecisionTreeClassifier, random_state=0),
       PenalisedAccuracy(X_val, y_val))
