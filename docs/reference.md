# Reference

Modules, all importable once the FASTENER directory is on `sys.path`:

| Module | Contents |
|---|---|
| `fastener` | `Config`, `EntropyOptimizer`, `FitBudgetExhausted`, `LogData` |
| `item` | `Item`, `EvalItem`, `Result`; mating, selection, mutation and swap strategies |
| `seeding` | the mRMR and Boruta filters behind `seed_filter` |
| `guidance` | feature priors, subset proposers, `PriorGuidedMutationStrategy` |
| `fast_mi` | `mutual_info_classif`, a faster drop-in for scikit-learn's |
| `random_utils` | the random generator FASTENER draws from |

Importing `fastener` also imports `dummy_data`, which builds a small
synthetic data set for the module's own `main()`, and seeds numpy's global
generator while doing so.

---

## The two things you supply

### The model factory

`model` is any callable that returns a new, unfitted estimator when called
with **no arguments**: `DecisionTreeClassifier`,
`functools.partial(RandomForestClassifier, n_estimators=50)`, or a
module-level function returning a pipeline. FASTENER calls
`model().fit(train_data[:, genes], train_target)` once per fit. The factory
is pickled with the optimizer, so it cannot be a lambda or a closure.

Requirements beyond `fit`: whatever your evaluator calls (`predict`,
`predict_proba`, …). `seed_filter="boruta"` also needs `feature_importances_`
after `fit`.

### The evaluator

```python
def evaluator(model, genes, shuffle_indices=None) -> Result
```

* `model`: the estimator, fitted on `train_data[:, genes]`.
* `genes`: the subset as a list of bools, one per feature. Index your
  validation data with it: `X_val[:, genes]`.
* `shuffle_indices`: `None` for a plain evaluation. During pruning it is a
  list of **positions among the selected columns** (0 … size − 1, not
  original feature indices) whose values you must shuffle before
  predicting. Shuffle a copy, and use `random_utils.shuffle` so runs repeat.
* Return `Result(score)`, where higher is better. For an error, return its
  negative.

The evaluator is pickled with the optimizer: use a module-level function or
a class with `__call__` (examples use `ValidationEvaluator` in
[examples/_common.py](examples/_common.py)).

---

## `fastener.Config`

A dataclass. Creating one **seeds the global random generator** and fixes the
log folder.

| Field | Default | Meaning |
|---|---|---|
| `output_folder` | required | Run name. Logs go to `log/<output_folder>` under the current directory, which must not exist yet. After construction the field holds the full path. |
| `random_seed` | required | Seed for `random_utils` (numpy's global generator). Applied when the `Config` is created. |
| `number_of_rounds` | `1000` | Maximum number of generations. |
| `max_bucket_size` | `3` | Subsets kept per size after each generation. |
| `reset_to_pareto_rounds` | `5` | Every this many generations, prune the front and reset the population to it. `None` or `0` means once, at the last round. |
| `cache_fitness_function` | `True` | Fit each distinct subset once; later lookups reuse the score. |
| `max_model_fits` | `None` | Stop after this many model fits (cache hits excluded). Must exceed the size of the initial population. |
| `dump_generation_logs` | `True` | Pickle each generation to `generation_N.pickle`. These run to megabytes per generation on wide data, so turn them off for batches. |

---

## `fastener.EntropyOptimizer`

```python
EntropyOptimizer(model, train_data, train_target, evaluator, number_of_genes,
                 mating_selection_strategy, mutation_strategy, *,
                 initial_population=None, initial_genes=None, config=None,
                 swap_strategy=None,
                 seed_filter=None, seed_size=10, boruta_max_iter=100,
                 feature_prior=None, prior_weight=0.5,
                 prior_uses=("mating", "mutation", "initial"),
                 prior_initial_singles=5, guidance_context=None,
                 subset_proposer=None, proposer_every=5,
                 reset_to_front_predicate=None)
```

The constructor creates the log folder (`FileExistsError` if it exists).

### Required

| Parameter | Meaning |
|---|---|
| `model` | Model factory (above). |
| `train_data` | 2-D numpy array, samples × features. Convert a DataFrame with `.to_numpy()`. |
| `train_target` | 1-D array. Class labels for the default mutual-information estimate. A continuous target needs a regression MI hook, see [regression.md](regression.md). |
| `evaluator` | Evaluator (above). It holds the validation data. FASTENER never passes data to it. |
| `number_of_genes` | Number of features, `train_data.shape[1]`. |
| `mating_selection_strategy` | How parents are chosen and mated, e.g. `RandomEveryoneWithEveryone(...)`. |
| `mutation_strategy` | E.g. `RandomFlipMutationStrategy(1 / number_of_genes)`. |

### Starting points (give at least one)

| Parameter | Meaning |
|---|---|
| `initial_genes` | List of subsets as lists of feature indices, e.g. `[[i] for i in range(n)]`. |
| `initial_population` | `{size: [Item, ...]}` of unevaluated items. |
| `seed_filter`, `seed_size`, `boruta_max_iter` | `"mrmr"` or `"boruta"`: add the filter's top-1 … top-`seed_size` nested subsets. Boruta runs at most `boruta_max_iter` iterations of the model, which count as fits. See [Seeds](#seeding). |
| `feature_prior` with `"initial"` in `prior_uses` | Add `prior_initial_singles` single features drawn from the prior-weighted MI. |

### Optional operators

| Parameter | Meaning |
|---|---|
| `config` | A `Config`. The default is `Config("data", 2020)`, i.e. `log/data`, so a second run with the default fails. Always pass one. |
| `swap_strategy` | A `SwapStrategy` for the pruning step. `None` keeps deletion-only pruning. |
| `feature_prior`, `prior_weight`, `prior_uses`, `prior_initial_singles`, `guidance_context` | LLM guidance: relevance scores per feature. See [LLM guidance](#guidance). |
| `subset_proposer`, `proposer_every` | LLM guidance: a callable proposing subsets every `proposer_every` generations. See [LLM guidance](#guidance). |
| `reset_to_front_predicate` | Accepted and stored, but **not used**: the pruning cadence is `Config.reset_to_pareto_rounds`. |

### Methods

| Method | Meaning |
|---|---|
| `mainloop()` | Run the whole search. Prints `Round: N` each generation. Raises `FitBudgetExhausted` only if the budget runs out while the initial population is being evaluated. |
| `prepare_loop()` | The set-up step of `mainloop`, for calling on its own: evaluate the initial population, compute the data information, apply a prior, a seed and a swap strategy. |
| `run_round(n)` | One generation. |
| `front_from_swap()` | How many current front entries a swap made. |

### Attributes after a run

| Attribute | Meaning |
|---|---|
| `pareto_front` | `{size: EvalItem}`, the result. |
| `population` | The last population, `{size: [EvalItem, ...]}`. |
| `model_fits` | Models fitted, including Boruta seed fits. |
| `stop_reason` | `"rounds"` or `"fit_budget"`. |
| `rounds_completed` | Generations finished. One interrupted by the budget is not counted. |
| `prune_stats` | Counter: `pruned_items`, `deletion_fits`, `deletion_kept`, `swap_proposed`, `swap_fits`, `swap_kept`. |
| `cache_data` | `{Item.number: (Result, fitted model)}` for every subset fitted. |
| `cache_counter` | `{Item.number: lookups}`. |
| `seed_ranking`, `seed_fits` | The filter's first `seed_size` features, and the fits it cost. |
| `prior_scores`, `feature_weights`, `prior_singles` | The resolved prior, the combined weights, and the singles drawn. |
| `proposal_log` | One dict per proposer call: `generation`, `proposed`, `evaluated` (features and score), `fits`. |

---

## Items and results (`item`)

| Class | Fields and methods |
|---|---|
| `Result(score)` | `score: float`, compared by score. Subclass it to carry more, e.g. extra metrics. Comparisons use `score` only. |
| `Item(genes, generation, parent_a, parent_b)` | `genes`, `size`, `number`, `generation`, parents' genes. `Item.from_genes(genes)`, `Item.to_number(genes)`, `Item.num_to_genes(number, n)`. |
| `EvalItem(..., result)` | An evaluated `Item`. `pareto_better(other)` is true when it is no larger and scores at least as well. |

To read a subset's features: `numpy.flatnonzero(item.genes)`.

---

## Mating selection strategies (`item`)

| Class | Behaviour |
|---|---|
| `RandomEveryoneWithEveryone(mating_strategy=UnionMating(), pool_size=10, allow_self_mating=False)` | Draw `pool_size` items from the population and mate every pair. `pool_size=None` uses the whole population. The original algorithm uses `pool_size=3`. `allow_self_mating=True` draws with replacement, as the 2020 code did. |
| `NoMating(mating_strategy)` | Pass the population through unmated, so only mutation searches. It requires exactly one item per size, e.g. right after a reset to the front. |

Write your own by subclassing `MatingSelectionStrategy` and implementing
`mating_pool(population) -> MatingPoolResult(mating_pool, carry_over)` and
`mate_pool(pool, generation) -> [Item]`. See
[examples/10_custom_strategies.py](examples/10_custom_strategies.py).

## Mating strategies (`item`)

| Class | Child's genes |
|---|---|
| `UnionMating()` | Union of the parents. |
| `IntersectionMating()` | Intersection. |
| `IntersectionMatingWithInformationGain(number=None)` | Intersection plus the `number(k)` differing genes with the highest MI, where `k` is the difference in the parents' sizes. |
| `IntersectionMatingWithWeightedRandomInformationGain(number=None, scaling=None, include_top_genes=True)` | As above, plus `number(k)` differing genes sampled with replacement in proportion to `scaling(MI)`. `include_top_genes=False` keeps only the sample. The default, and what the published algorithm uses. |

`number(k)` defaults to `min(k // 2 + 1, k)`. `scaling(values)` defaults to
`log1p(log1p(log1p(values)))` and must return a numpy array. Both are pickled
with the optimizer, so make them module-level functions.

The MI vector lives in `scikit_information_gain` and is filled by
`use_data_information(X, y)`. Override that method to change the estimator,
for example for regression.

## Mutation strategies

| Class | Behaviour |
|---|---|
| `item.RandomFlipMutationStrategy(prob=0.01)` | Flip each gene independently with probability `prob`. |
| `guidance.PriorGuidedMutationStrategy(prob=0.01)` | The same expected number of flips, but unselected features are switched on in proportion to their weight and selected ones switched off in proportion to how weak they are. Without weights it is `RandomFlipMutationStrategy`, draw for draw. Required for `"mutation"` in `prior_uses`. |

Subclass `MutationStrategy` and implement `mutate_internal(item) -> genes`.

## Swap strategies (`item`)

| Class | Behaviour |
|---|---|
| `InformationGainSwapStrategy(number_of_swaps=1, scaling=None)` | Drop the permutation-weakest feature, and insert `number_of_swaps` distinct unselected features sampled by `scaling(MI)`. |
| `RandomSwapStrategy(number_of_swaps=1)` | Drop a random selected feature and insert random unselected ones. The ablation for the guided swap. |

Subclass `SwapStrategy` and implement `select_removal(genes, changes)`, where
`changes` is a list of `(score_drop, gene_index)` sorted ascending, and
`select_insertions(genes, removed, count)`.

---

## <a id="seeding"></a>Seeds (`seeding`)

| Function | Returns |
|---|---|
| `rank_mrmr(X, y, n_select, n_bins=10, random_state=None)` | The first `n_select` features in mRMR (MID) order. Relevance is scikit-learn's `mutual_info_classif` (classification only). Redundancy is MI between features binned into `n_bins` equal-width bins. Fits no model. |
| `rank_boruta(model_factory, X, y, random_state=None, max_iter=100)` | `(order, confirmed, fits)`: every feature ordered by Boruta rank, then by mean shadow-relative importance; the confirmed features; the iterations run. Needs `pip install Boruta` and a model with `feature_importances_`. Works for classifiers and regressors. |
| `nested_prefixes(order, k)` | `[[f1], [f1, f2], …, [f1 … fk]]`. |

`EntropyOptimizer(seed_filter=...)` calls these with `Config.random_seed` as the
random state.

## <a id="guidance"></a>LLM guidance (`guidance`)

The feature prior and the subset proposer are how a large language model
guides the search. [llm-guidance.md](llm-guidance.md) explains them, and
[examples/08_llm_guidance.py](examples/08_llm_guidance.py) implements both
with an LLM. FASTENER calls them as plain functions and makes no network
calls itself, so they also accept guidance from other sources.

**Feature prior.** `feature_prior` is either one non-negative score per
feature, or a callable `prior(GuidanceContext) -> scores`. It is resolved once,
in `prepare_loop`, and combined with mutual information:

```
w = unit(MI) ** (1 - prior_weight) * unit(prior) ** prior_weight
```

`unit` scales a vector to a maximum of 1 and floors it at 0.001, so a prior
biases the search but forbids nothing. `prior_weight=0` uses MI only, `1` the
prior only. `prior_uses` picks what `w` drives:

| Use | Effect | Requires |
|---|---|---|
| `"mating"` | `w` replaces the mating strategy's MI vector. | An information-gain mating strategy. |
| `"mutation"` | `PriorGuidedMutationStrategy` flips by `w`. | That mutation strategy. |
| `"initial"` | `prior_initial_singles` single features drawn from `w` join the initial population. | — |

The default `prior_uses` has all three, so with a plain
`RandomFlipMutationStrategy` you must leave out `"mutation"`.

`GuidanceContext(n_features, feature_names, task=None, class_names=None,
metadata={})` is what a callable prior and a proposer can know about the
problem. It defaults to names `f0, f1, …`, and `n_features` must equal
`number_of_genes`.

**Subset proposer.** `subset_proposer(ProposalContext) -> [[feature indices],
...]` is called after the front update every `proposer_every` generations.
`ProposalContext` has `guidance`, `generation`, `front` (a list of
`{"size", "features", "score"}`), `model_fits`, `max_model_fits` and
`feature_weights` (`None` without a prior). Proposed subsets are cleaned:
out-of-range indices, empty subsets and duplicates are dropped. They are then
evaluated from the same fit budget and join the population. A subset already
in the cache costs nothing and adds nothing, so a useful proposer remembers
what it has proposed.

The optimizer's pickle leaves out a callable prior and the proposer, so those
two, unlike the model and evaluator, may be closures or hold network
clients.

Helpers: `resolve_prior`, `combine`, `sample_singles`, `front_summary`,
`proposals_to_genes`.

---

## Mutual information (`fast_mi`)

`fast_mi.mutual_info_classif(X, y, n_neighbors=3, random_state=None)` returns
the same values as `sklearn.feature_selection.mutual_info_classif` on dense,
continuous `X`, several times faster on wide data. It consumes the random
generator exactly as scikit-learn does. `y` must be class labels.

## Logs (`fastener.LogData`)

`log/<output_folder>/experiment.pickle` is the optimizer after `prepare_loop`
(training data included). `generation_N.pickle` is a `LogData` with
`generation`, `population`, `front`, `mated`, `mutated`, `cache_counter`,
`cache_data` (results only, no models), `random_state` and `timestamp`.
Unpickling needs the FASTENER directory on `sys.path`, and the same classes
your evaluator and model come from.
