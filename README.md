# FASTENER
___FeAture SelecTion ENabled by EntRopy___ for Python

FASTENER is a state-of-the-art feature selecton algorithm for remote sensing, but performs well also on several other data sets. It is most suitable for large datasets with several hundreds of features. It has been develped for the use case of crop/land-cover classification based on Sentinel-II data.

## Prerequisites
For further details see `requirements.txt`.

* Python 3.6+
* numpy (1.18-1+)
* scipy (1.4.1+)
* typed-ast (1.4.1.+)
* scikit-learn (0.22.2+)
* typing-extensions (3.7.4.1+)
* For example:
    * Pandas

## Users' Manual
Basic documentation is available within the code.

A simple workflow is described below.

0) Includes
```python
# import dataset
from sklearn.datasets import load_breast_cancer

# import preprocessing tools
from sklearn import preprocessing
import numpy as np
import pandas as pd

# import learning/evaluation
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

# typing
from typing import Dict, List, Callable, Any, Tuple, Optional, \
    Counter as CounterType, Set

# FASTENER specific imports
from src.random_utils import shuffle
from src import random_utils
from src.item import Item, EvalItem, Result, Population, flatten_population, FitnessFunction, \
    Genes, EvalItem, RandomFlipMutationStrategy, RandomEveryoneWithEveryone, \
    IntersectionMating, UnionMating, IntersectionMatingWithInformationGain, \
    IntersectionMatingWithWeightedRandomInformationGain, UnevaluatedPopulation, \
    MatingStrategy, MutationStrategy, MatingSelectionStrategy
from src import fastener
```

1) Prepare data
```python
# loading breast cancer dataset
# scikit-learn 0.22+ is needed
cancer = load_breast_cancer(as_frame=True)
X_df = cancer.data
y_df = cancer.target

# basic dataset split
n_sample = X_df.shape[0]
n_test = int(n_sample * 0.8)

labels_train = y_df.to_numpy()[:n_test]
labels_test = y_df.to_numpy()[n_test:]

XX_train = X_df.to_numpy()[:n_test, :]
XX_test = X_df.to_numpy()[n_test:, :]
```

2) Define feature evaluation function
```python
def eval_fun(model: Any, genes: "Genes", shuffle_indices: Optional[List[int]] = None) -> "Result":
    test_data = XX_test[:, genes]
    if shuffle_indices:
        test_data = test_data.copy()
        for j in shuffle_indices:
            shuffle(test_data[:, j])
    pred = model.predict(test_data)
    res = Result(f1_score(labels_test, pred))
    return res
```

3) Configure the FASTENER

By default fastener runs for 1000 iterations. The number of iterations can be adjusted with `number_of_rounds` parameter in the `fastener.Config()`.

```python
number_of_genes = XX_train.shape[1]
general_model = DecisionTreeClassifier
# output folder name must be changed every time the algorithm is run
output_folder_name="output"

# to start the algorithm initial_genes or initial_population must be provided
initial_genes = [
    [0]
]

# Select mating selection strategies (RandomEveryoneWithEveryone, NoMating) and mating strategy
# (UnionMating, IntersectionMating, IntersectionMatingWithInformationGain, IntersectionMatingWithWeightedRandomInformationGain)
mating = RandomEveryoneWithEveryone(pool_size=3, mating_strategy=IntersectionMatingWithWeightedRandomInformationGain())

# Random mutation (probability of gene mutating: 1 / number_of_genes)
mutation = RandomFlipMutationStrategy(1 / number_of_genes)

entropy_optimizer = fastener.EntropyOptimizer(
    general_model, XX_train, labels_train, eval_fun,
    number_of_genes, mating, mutation, initial_genes=initial_genes,
    config=fastener.Config(output_folder=output_folder_name, random_seed=2020, reset_to_pareto_rounds=5)
)
```

4) Run FASTENER loop
```python
entropy_optimizer.mainloop()
```

5) Check evaluation of the 1000th iteration
```python
# read log from last generation
object = pd.read_pickle(f'log/{output_folder_name}/generation_1000.pickle')

# list of best-scoring EvalItem objects for each number of features
best = list(object.front.values())

for item in best:
    # names of best features
    selected_features = X_df.iloc[:, item.genes].columns.tolist()

    X = X_df[selected_features].values.astype(float)
    y = y_df.values.astype(float)

    # evaluates each set of features with cross validation
    model = DecisionTreeClassifier()
    cvs = cross_val_score(model, X, y, cv=10)
    print("Features:", selected_features)
    print("Accuracy: ", cvs.mean(), " stdev: ", cvs.std(), "\n")
```



For detailed workflow check `Example.ipynb`.

## Future Work

* Create PyPI module
* Update documentation
* Prepare example notebooks
* Create unit tests

## Publications

If you use the algorithm, please cite the following paper:

* Koprivec, F.; Kenda, K.; Šircelj, B., FASTENER Feature Selection for Inference from Earth Observation Data. Entropy 2020, ?, ?. (_minor reviews pending_)

__Abstract__:

> In this paper, FASTENER feature selection algorithm is presented.
    The algorithm exploits entropy-based measures such as mutual information in the crossover phase of the genetic algorithm approach.
    FASTENER converges to an (near) optimal subset of features faster than previous state-of-the-art algorithms and achieves better classification accuracy than similarity-based methods such as KBest or ReliefF or wrapper methods such as POSS.
    The approach was evaluated using the Earth Observation dataset for land-cover classification from ESA's Sentinel-2 mission, the digital elevation model and the ground truth data of the Land Parcel Identification System from Slovenia.
    The algorithm can be used in any statistical learning scenario.


### Acknowledgements
This research was funded by European Union's Horizon 2020 programme project PerceptiveSentinel (Research and Innovation) grant number [776115](https://cordis.europa.eu/project/id/776115), project NAIADES (Innovation Action) grant number [820985](https://cordis.europa.eu/project/id/820985).