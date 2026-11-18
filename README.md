<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo_dark.svg">
    <img src="assets/logo.svg" alt="FASTENER" width="460">
  </picture>
</p>

<p align="center">
  <a href="https://doi.org/10.3390/e22111198"><img src="https://img.shields.io/badge/paper-Entropy%202020-1f3a5f" alt="paper"></a>
  <a href="docs/README.md"><img src="https://img.shields.io/badge/docs-read-f5a623" alt="documentation"></a>
  <img src="https://img.shields.io/badge/python-3.11%20tested-3776ab?logo=python&logoColor=white" alt="python">
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/changelog-2026-5b6b80" alt="changelog"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="license"></a>
</p>

<p align="center">
  FASTENER is a genetic algorithm for <b>feature selection</b> that finds the best subset of <b>every size</b> at once.
</p>

<br>

FASTENER wraps any model with `fit` and `predict`, and any score. Instead of a
single subset, it returns a **Pareto front**: one subset per size, each scoring
strictly better than every smaller one. That lets you trade accuracy for
fewer features after a single run. Mutual information between features and the
target guides the crossover. Permutation importance guides the pruning of the
front. You can also let a large language model, a domain expert or another
selector guide the search.

FASTENER was first developed for inference from Earth observation data
([Entropy, 2020](https://doi.org/10.3390/e22111198)). Since then it has gained
**LLM-driven guidance**: a large language model reads the feature names and the
task and scores how relevant each feature is likely to be, which biases the
search, and it proposes new subsets while the search runs. Every score still
comes from fitting your model ([LLM guidance](docs/llm-guidance.md)). It has
also gained a swap operator, fit-capped budgets, filter seeds, faster mutual
information, tests and full documentation. See the [changelog](CHANGELOG.md).

## ⚡️ Quickstart

Select features for a decision tree on the breast-cancer data:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from evaluators import Evaluator
from fastener import Config, EntropyOptimizer
from item import (IntersectionMatingWithWeightedRandomInformationGain,
                  RandomEveryoneWithEveryone, RandomFlipMutationStrategy)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, random_state=0)

n = X.shape[1]
optimizer = EntropyOptimizer(
    DecisionTreeClassifier, X_train, y_train,
    Evaluator(X_val, y_val, "f1"),          # score subsets by F1 on validation
    n,
    RandomEveryoneWithEveryone(
        pool_size=3,
        mating_strategy=IntersectionMatingWithWeightedRandomInformationGain()),
    RandomFlipMutationStrategy(1 / n),
    initial_genes=[[i] for i in range(n)],
    config=Config(output_folder="my_run", random_seed=2020, number_of_rounds=50))
optimizer.mainloop()

for size, item in sorted(optimizer.pareto_front.items()):
    print(size, round(item.result.score, 4))
```

`optimizer.pareto_front` maps each size to the best subset found
(`item.genes`, `item.result.score`). The
[full quickstart](docs/examples/01_quickstart.py) also holds out a test split
and picks the smallest subset close to the best.

### Choosing the score

`Evaluator` scores a fitted model on your validation data, higher is better.
Pick a metric by name. Keyword arguments go to the scikit-learn metric:

```python
Evaluator(X_val, y_val, "f1")                         # weighted F1 (the default)
Evaluator(X_val, y_val, "f1", average="macro")
Evaluator(X_val, y_val, "recall", average="binary", pos_label=0)
Evaluator(X_val, y_val, "roc_auc")                    # uses predict_proba
Evaluator(X_val, y_val, "r2")                         # regression
Evaluator(X_val, y_val, "rmse")                       # errors are negated: -RMSE
```

The names are `f1`, `precision`, `recall`, `accuracy`, `balanced_accuracy`,
`matthews`, `roc_auc`, `log_loss`, `r2`, `mae`, `mse` and `rmse`.

For your own evaluator, start with the least code that does the job:

```python
# 1. Your own metric: a module-level function metric(y_true, y_pred)
def f2(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2)

Evaluator(X_val, y_val, f2)            # greater_is_better=False for an error

# 2. Your own scoring of the model: override score(); shuffling is handled
class ConfidentAccuracy(Evaluator):
    def score(self, model, X_eval):
        proba = model.predict_proba(X_eval)
        right = model.classes_[proba.argmax(axis=1)] == self.y_val
        return float(np.mean(right & (proba.max(axis=1) >= 0.8)))

# 3. From scratch: any picklable callable
#    evaluator(model, genes, shuffle_indices=None) -> Result(score)
```

[Example 02](docs/examples/02_models_and_metrics.py) runs all of these, and the
[reference](docs/reference.md#the-evaluator) gives the contract for writing one
from scratch.

## 🛠 Installation

FASTENER is a directory of modules with flat imports, not a pip package.
Clone it and put it on `sys.path`:

```sh
git clone https://github.com/klemenkenda/FASTENER fastener
pip install -r fastener/requirements.txt   # numpy, scikit-learn
pip install Boruta                         # optional, only for seed_filter="boruta"
```

```python
import sys
sys.path.insert(0, "path/to/fastener")
```

It is tested on Python 3.11 with numpy 2.4 and scikit-learn 1.9.

## 🔮 Features

- **Any model, any score.** FASTENER fits your model and scores it with your
  evaluator: a built-in one for F1, precision, recall, accuracy, ROC AUC, R²,
  RMSE and more, or your own. It works for classification and, with a regression estimate of
  mutual information, for [regression](docs/regression.md).
- **A whole Pareto front per run.** The best subset of every size, not only one
  subset.
- **Guided genetic operators.** Crossover weighted by mutual information, bit-flip
  mutation, and pruning by permutation importance. Union, intersection and your
  own strategies can be plugged in.
- **Swap in pruning** (`swap_strategy=`). Pruning also tries replacing the
  weakest feature with another one, chosen by mutual information or at random.
- **Filter seeds** (`seed_filter="mrmr"` / `"boruta"`). The search starts from a
  filter's nested top-k subsets.
- **Fit budgets** (`Config.max_model_fits`). The search stops after an exact
  number of model fits, so variants can be compared at equal compute.
- **LLM and expert guidance** (`feature_prior=`, `subset_proposer=`). A prior
  biases mating, mutation and the starting subsets. A proposer suggests subsets
  during the search. Every score still comes from your model, and FASTENER
  itself makes no network calls. See [LLM guidance](docs/llm-guidance.md).
- **Reproducible.** A fixed seed gives the same front. With two flags, a run is
  identical to the 2020 code (see the [changelog](CHANGELOG.md)).

Every option is off by default, and they combine freely. How the variants
relate, from the 2020 behaviour to the current one, and which experiment arm
uses which, is in [docs/variants.md](docs/variants.md).

```python
from item import InformationGainSwapStrategy

optimizer = EntropyOptimizer(
    ...,
    swap_strategy=InformationGainSwapStrategy(),   # swap in pruning
    seed_filter="mrmr", seed_size=10,              # start from mRMR's top 10
    config=Config(..., max_model_fits=6000))       # stop after 6,000 fits
```

## 🤔 Should I be using FASTENER?

FASTENER is a **wrapper** method: it judges a subset by fitting your model on
it and scoring it with your evaluator. That makes it accurate for your model,
and also more costly than a filter. It is a good fit if:

- **The model decides what matters.** Interactions and redundancy that your
  model actually uses count, not a proxy such as correlation with the target.
- **You want to choose the size yourself.** The Pareto front shows how the
  score falls as features are removed, so you can pick the smallest subset
  that is good enough, for example for cheaper sensors, faster inference or a
  model people can read.
- **You have many features and fits are cheap.** Hundreds to thousands of
  features with a decision tree, a forest or a similar fast model.
- **You know something the data does not show.** Feature names, a data
  dictionary or an expert's ranking can guide the search through a prior or a
  subset proposer.
- **You compare selectors.** Fit budgets, fixed seeds and logged fronts make
  runs reproducible and comparable at equal compute.

It is probably not the right tool if:

- **One model fit is expensive**, for example a large neural network. A search
  needs thousands of fits.
- **You only need a ranking.** A filter such as mutual information, mRMR or
  Boruta is cheaper. FASTENER can still start from such a ranking with
  `seed_filter` and improve on it.
- **You need a pip package or a scikit-learn transformer.** FASTENER is a
  directory of modules. You call the optimizer and read its front yourself.

## 🔗 Useful links

- [Documentation](docs/README.md)
  - [How the search works](docs/concepts.md)
  - [Reference](docs/reference.md): every class, parameter and attribute
  - [Examples](docs/examples.md), with [ten runnable scripts](docs/examples/)
  - [Variants](docs/variants.md): configuring the search operators and LLM guidance, and how the variants relate
  - [LLM guidance](docs/llm-guidance.md)
  - [Regression](docs/regression.md)
  - [Pitfalls and questions](docs/pitfalls.md)
- [Changelog](CHANGELOG.md): what changed since the 2020 paper
- [Issue tracker](https://github.com/klemenkenda/FASTENER/issues)

## 👐 Contributing

Issues and pull requests are welcome. Run the unit tests before you open a pull
request; [Tests](docs/README.md#tests) shows how and what each file covers.

Changes to the search should stay off by default, so that existing runs keep
their results. Record each one in the [changelog](CHANGELOG.md).

## 🤝 Affiliations

FASTENER is developed at the [Jožef Stefan Institute](https://www.ijs.si),
Ljubljana, Slovenia, within these projects:

| Project | Funding | Grant |
|---|---|---|
| **PerceptiveSentinel**: big data knowledge extraction and re-creation platform for Copernicus Sentinel data | EU Horizon 2020 | [776115](https://cordis.europa.eu/project/id/776115) |
| **NAIADES**: a holistic water ecosystem for digitisation of the urban water sector | EU Horizon 2020 | [820985](https://cordis.europa.eu/project/id/820985) |
| **Plooto**: product passport through twinning of circular value chains | EU Horizon Europe | [101092008](https://cordis.europa.eu/project/id/101092008) |
| **EnviroViroSeq**: advanced HTS and AI-powered analysis of viromes in hops and the environment | Slovenian Research and Innovation Agency (ARIS) | [J4-70162](https://cris.cobiss.net/ecris/si/en/project/24463) |

## 💬 Citation

If you use FASTENER, please cite the paper:

> Koprivec, F.; Kenda, K.; Šircelj, B. FASTENER Feature Selection for Inference
> from Earth Observation Data. *Entropy* **2020**, 22(11), 1198.
> [doi:10.3390/e22111198](https://doi.org/10.3390/e22111198)

```bibtex
@article{koprivec2020fastener,
  author  = {Koprivec, Filip and Kenda, Klemen and {\v{S}}ircelj, Beno},
  title   = {{FASTENER} Feature Selection for Inference from Earth Observation Data},
  journal = {Entropy},
  volume  = {22},
  number  = {11},
  pages   = {1198},
  year    = {2020},
  doi     = {10.3390/e22111198}
}
```

The Earth observation dataset used in the paper (Sentinel-2 land patch samples
over Slovenia, 2017) is published separately:

> Šircelj, B.; Kenda, K.; Koprivec, F. Land Patch Samples. *PANGAEA* **2020**.
> [doi:10.1594/PANGAEA.914271](https://doi.org/10.1594/PANGAEA.914271) (CC-BY-4.0)

```bibtex
@misc{sircelj2020landpatch,
  author    = {{\v{S}}ircelj, Beno and Kenda, Klemen and Koprivec, Filip},
  title     = {Land Patch Samples},
  publisher = {PANGAEA},
  year      = {2020},
  doi       = {10.1594/PANGAEA.914271}
}
```

### Applications

Published work that applies FASTENER:

- Uku, S.; Kenda, K. Comparative Analysis of Feature Selection and Imbalance
  Handling for Semiconductor Defect Detection on the SECOM Dataset. In
  *Proceedings of Information Society 2026 (SiKDD)*, Ljubljana, Slovenia, 2026.
  [doi:10.70314/is.2026.sikdd.125](https://doi.org/10.70314/is.2026.sikdd.125).
  FASTENER is one of three selectors for semiconductor defect detection on SECOM,
  compared with a Pearson filter and Boruta.

## 📝 License

FASTENER is free and open source software under the [MIT license](LICENSE).
