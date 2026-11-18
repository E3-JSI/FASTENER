"""Any model, any score.

`model` is a factory that FASTENER calls with no arguments, once per fit,
so set hyperparameters with `functools.partial` or a module-level function.
The evaluator decides what "good" means: it returns a `Result` whose `score`
is higher for better subsets.

Both must pickle: FASTENER writes the whole optimizer to
log/<folder>/experiment.pickle before the search starts, so a lambda or a
closure as `model` or `evaluator` fails with "Can't pickle local object".
"""
from functools import partial

from _common import (ValidationEvaluator, classification_data, front_table,
                     run_folder)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from fastener import Config, EntropyOptimizer
from item import (IntersectionMatingWithWeightedRandomInformationGain,
                  RandomEveryoneWithEveryone, RandomFlipMutationStrategy,
                  Result)
import random_utils

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
    print(f"\n{name}:")
    print(front_table(optimizer, names))
    return optimizer


# 1. A tree with hyperparameters, scored by balanced accuracy.
search("tree_balanced_accuracy",
       partial(DecisionTreeClassifier, max_depth=5, class_weight="balanced",
               random_state=0),
       ValidationEvaluator(X_val, y_val, balanced_accuracy_score))

# 2. A small random forest (slower: keep it small while searching).
search("forest",
       partial(RandomForestClassifier, n_estimators=30, n_jobs=-1,
               random_state=0),
       ValidationEvaluator(X_val, y_val, balanced_accuracy_score), rounds=10)

# 3. A pipeline: scale, then logistic regression. A module-level function
#    builds a new pipeline for every fit (a lambda would not pickle).
def scaled_logistic():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))


search("logistic", scaled_logistic,
       ValidationEvaluator(X_val, y_val, balanced_accuracy_score))


# 4. A score that needs probabilities: write the evaluator directly.
def auc_evaluator(model, genes, shuffle_indices=None):
    X_eval = X_val[:, genes]
    if shuffle_indices:
        X_eval = X_eval.copy()
        for j in shuffle_indices:
            random_utils.shuffle(X_eval[:, j])
    return Result(roc_auc_score(y_val, model.predict_proba(X_eval)[:, 1]))


search("tree_auc", partial(DecisionTreeClassifier, max_depth=4, random_state=0),
       auc_evaluator)


# 5. A penalised score: trade accuracy against subset size inside the score
#    itself. The front already gives one subset per size, so this is rarely
#    needed, but the score is yours to define.
balanced = ValidationEvaluator(X_val, y_val, balanced_accuracy_score)


def penalised(model, genes, shuffle_indices=None):
    score = balanced(model, genes, shuffle_indices).score
    return Result(score - 0.002 * sum(genes))


search("tree_penalised", partial(DecisionTreeClassifier, random_state=0),
       penalised)
