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
| [variants.md](variants.md) | The variants of FASTENER: the historical mating versions, the implemented variant families and the experiment map |
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
from evaluators import Evaluator
from fastener import Config, EntropyOptimizer
from item import (IntersectionMatingWithWeightedRandomInformationGain,
                  RandomEveryoneWithEveryone, RandomFlipMutationStrategy)
from sklearn.tree import DecisionTreeClassifier

evaluator = Evaluator(X_val, y_val, "f1")    # weighted F1 on the validation split

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

## Tests

Unit tests live in [`tests/`](../tests/) and need only FASTENER's own
requirements plus pytest. From the repo root:

```sh
pip install pytest
python -m pytest tests
```

| File | What it covers |
|---|---|
| `test_item.py` | Genes, items and results |
| `test_strategies.py` | Mating, mutation and swap strategies |
| `test_optimizer.py` | `EntropyOptimizer`: the main loop, pruning, budgets and the Pareto front |
| `test_fast_mi.py` | The fast mutual-information estimator against scikit-learn's |
| `test_seeding.py` | mRMR and Boruta seeds |
| `test_guidance.py` | The feature prior and the subset proposer |

## Citing

If you use FASTENER, cite the paper. The reference and BibTeX are in the
[README](../README.md#-citation).
