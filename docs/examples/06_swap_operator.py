"""The swap operator in the pruning step (off by default).

Every `reset_to_pareto_rounds` rounds FASTENER prunes the front: for each
subset it finds the feature whose permutation hurts the score least and
tries the subset without it. A swap strategy also tries *replacing* that
feature with one that is not selected, so the candidate competes with the
front entry of the same size. Each swap candidate costs one model fit.
"""
from _common import ValidationEvaluator, classification_data, front_table, run_folder

from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier

from fastener import Config, EntropyOptimizer
from item import (InformationGainSwapStrategy,
                  IntersectionMatingWithWeightedRandomInformationGain,
                  RandomEveryoneWithEveryone, RandomFlipMutationStrategy,
                  RandomSwapStrategy)

X_train, y_train, X_val, y_val, X_test, y_test, names = classification_data()
n = X_train.shape[1]
evaluator = ValidationEvaluator(X_val, y_val, balanced_accuracy_score)

SWAPS = {
    "no swap (published algorithm)": None,
    # Drop the permutation-weakest feature; insert one sampled by mutual
    # information (reusing the mating strategy's MI vector).
    "guided swap": InformationGainSwapStrategy(),
    # Three replacement candidates per pruned subset: three fits each time.
    "guided swap x3": InformationGainSwapStrategy(number_of_swaps=3),
    # The ablation: a random feature out, a uniformly random one in.
    "random swap": RandomSwapStrategy(),
}

for name, swap in SWAPS.items():
    optimizer = EntropyOptimizer(
        DecisionTreeClassifier, X_train, y_train, evaluator, n,
        RandomEveryoneWithEveryone(
            pool_size=3,
            mating_strategy=IntersectionMatingWithWeightedRandomInformationGain()),
        RandomFlipMutationStrategy(1 / n),
        initial_genes=[[i] for i in range(n)],
        swap_strategy=swap,
        config=Config(output_folder=run_folder("swap"), random_seed=5,
                      number_of_rounds=10_000, max_model_fits=500,
                      dump_generation_logs=False))
    optimizer.mainloop()
    stats = optimizer.prune_stats
    print(f"\n{name}: {optimizer.rounds_completed} rounds within the budget")
    print(front_table(optimizer, names))
    print(f"  pruned {stats['pruned_items']} subsets; deletions kept "
          f"{stats['deletion_kept']}; swaps proposed {stats['swap_proposed']}, "
          f"fitted {stats['swap_fits']}, kept {stats['swap_kept']}; "
          f"front entries made by a swap now: {optimizer.front_from_swap()}")
