# FASTENER documentation

FASTENER (FeAture SelecTion ENabled by EntRopy) is a genetic algorithm for
feature selection. It searches for the best subset of features *for every
subset size* at once, and returns a Pareto front: for each size on the front,
the best-scoring subset found, where every larger subset scores strictly
better than every smaller one. Mutual information between features and the
target guides the crossover, and permutation importance guides the pruning
of the front. Optionally, a large language model guides the search too
([llm-guidance.md](llm-guidance.md)).

FASTENER is a wrapper. It fits your model on candidate subsets and scores
them with your evaluator, so it works with any model that has `fit` and
`predict`, any score, and both classification and regression.

## Contents

| Page | What it covers |
|---|---|
| [concepts.md](concepts.md) | How the search works: genes, population, the Pareto front, a generation step by step, pruning, caching, randomness |
| [reference.md](reference.md) | Every class, parameter and attribute: `Config`, `EntropyOptimizer`, the evaluator and model contracts, strategies, seeds, guidance, logs |
| [examples.md](examples.md) | A catalogue of recipes, each backed by a runnable script in [examples/](examples/) |
| [llm-guidance.md](llm-guidance.md) | Guiding the search with a large language model: the feature prior and the subset proposer |
| [regression.md](regression.md) | Using FASTENER with a continuous target |
| [pitfalls.md](pitfalls.md) | Errors you will meet and how to avoid them, and questions that come up |

For what changed since the 2020 paper, see [../CHANGELOG.md](../CHANGELOG.md).

## Install

FASTENER is not a package with a `setup.py`; it is a directory of modules
with flat imports (`from item import ...`). Put the directory on `sys.path`:

```bash
git clone https://github.com/klemenkenda/FASTENER fastener
pip install -r fastener/requirements.txt   # numpy, scikit-learn
pip install Boruta                         # optional: only for seed_filter="boruta"
```

```python
import sys
sys.path.insert(0, "path/to/fastener")

from fastener import Config, EntropyOptimizer
from item import Result, RandomEveryoneWithEveryone, RandomFlipMutationStrategy
```

## A minimal run

```python
import random_utils
from fastener import Config, EntropyOptimizer
from item import (IntersectionMatingWithWeightedRandomInformationGain,
                  RandomEveryoneWithEveryone, RandomFlipMutationStrategy, Result)
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

def evaluator(model, genes, shuffle_indices=None):
    X = X_val[:, genes]
    if shuffle_indices:                      # permutation importance calls
        X = X.copy()
        for j in shuffle_indices:
            random_utils.shuffle(X[:, j])
    return Result(f1_score(y_val, model.predict(X), average="weighted"))

n = X_train.shape[1]
optimizer = EntropyOptimizer(
    DecisionTreeClassifier, X_train, y_train, evaluator, n,
    RandomEveryoneWithEveryone(
        pool_size=3,
        mating_strategy=IntersectionMatingWithWeightedRandomInformationGain()),
    RandomFlipMutationStrategy(1 / n),
    initial_genes=[[i] for i in range(n)],
    config=Config(output_folder="my_run", random_seed=2020, number_of_rounds=100))
optimizer.mainloop()

for size, item in sorted(optimizer.pareto_front.items()):
    print(size, item.result.score, [i for i, g in enumerate(item.genes) if g])
```

The full version is [examples/01_quickstart.py](examples/01_quickstart.py).

## Running the examples

Each script in [examples/](examples/) runs on its own, on data that ships with
scikit-learn, in seconds to under a minute:

```bash
cd fastener/docs/examples
python 01_quickstart.py
```

FASTENER writes its log folders to `log/` under the current directory, so run
the examples from a directory where that is fine. They were checked with
Python 3.11, numpy 2.4, scikit-learn 1.9 and Boruta 0.4.

## Citing

If you use FASTENER, cite the paper. The reference and BibTeX are in the
[README](../README.md#citation).
