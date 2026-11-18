"""Start the search from a filter's ranking: mRMR or Boruta (off by default).

`seed_filter="mrmr"` or `"boruta"` adds the filter's nested top-k subsets --
the best feature, the best two, ..., the best `seed_size` -- to the initial
population. The filters can also be used on their own, from `seeding`.
"""
from _common import ValidationEvaluator, classification_data, front_table, run_folder

from functools import partial

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier

import seeding
from fastener import Config, EntropyOptimizer
from item import (IntersectionMatingWithWeightedRandomInformationGain,
                  RandomEveryoneWithEveryone, RandomFlipMutationStrategy)

X_train, y_train, X_val, y_val, X_test, y_test, names = classification_data()
n = X_train.shape[1]
evaluator = ValidationEvaluator(X_val, y_val, balanced_accuracy_score)


def run(name, model=DecisionTreeClassifier, **kwargs):
    optimizer = EntropyOptimizer(
        model, X_train, y_train, evaluator, n,
        RandomEveryoneWithEveryone(
            pool_size=3,
            mating_strategy=IntersectionMatingWithWeightedRandomInformationGain()),
        RandomFlipMutationStrategy(1 / n),
        config=Config(output_folder=run_folder(name), random_seed=2,
                      number_of_rounds=10_000, max_model_fits=300,
                      dump_generation_logs=False),
        **kwargs)
    optimizer.mainloop()
    print(f"\n{name}: seed {optimizer.seed_ranking}, "
          f"seed cost {optimizer.seed_fits} fits")
    print(front_table(optimizer, names))
    return optimizer


# mRMR fits no model. The seed comes on top of any initial_genes.
run("mrmr_seed", seed_filter="mrmr", seed_size=8, initial_genes=[[0]])

# Boruta needs `pip install Boruta` and a model with feature_importances_.
# Its fits come out of the same max_model_fits budget.
try:
    run("boruta_seed", seed_filter="boruta", seed_size=8, boruta_max_iter=30,
        initial_genes=[[0]])
    run("boruta_seed_forest",
        model=partial(RandomForestClassifier, n_estimators=50, n_jobs=-1),
        seed_filter="boruta", seed_size=8, boruta_max_iter=20,
        initial_genes=[[0]])
except ImportError as e:
    print(f"\nskipping the Boruta seed: {e}")

# The filters on their own.
order = seeding.rank_mrmr(X_train, y_train, 5, random_state=0)
print("\nmRMR top 5:", [names[i] for i in order])
print("nested prefixes:", seeding.nested_prefixes(order, 3))
try:
    order, confirmed, fits = seeding.rank_boruta(
        DecisionTreeClassifier, X_train, y_train, random_state=0, max_iter=30)
    print(f"Boruta: {len(confirmed)} confirmed in {fits} fits; top 5:",
          [names[i] for i in order[:5]])
except ImportError:
    pass
