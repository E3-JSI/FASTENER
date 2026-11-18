# Pitfalls and questions

## Errors

### `FileExistsError: ... 'log\\my_run'`

`EntropyOptimizer` creates `log/<output_folder>` and refuses to reuse it.
Give every run its own name, for example with a timestamp, or delete the
old folder. Without a `config` the folder is `log/data`, so the second run
without one fails.

### `Can't pickle local object ...` or `Can't pickle <function <lambda>>`

Before the search, `prepare_loop` pickles the whole optimizer, model factory
and evaluator included, to `experiment.pickle`. Lambdas, closures and classes
defined inside functions do not pickle. Use module-level functions,
`functools.partial` of a class, or a class with `__call__`. A callable
`feature_prior` and the `subset_proposer` are exempt: they are left out of
the pickle.

### `ValueError: Unknown label type: continuous`

The target is continuous and a component estimated classification mutual
information. The component is the mating strategy, the mRMR seed, or a
feature prior or guided swap without an MI vector. See
[regression.md](regression.md).

### `ValueError: prior_uses has 'mutation', but the mutation strategy takes no feature weights`

The default `prior_uses` includes `"mutation"`. Use
`guidance.PriorGuidedMutationStrategy`, or pass
`prior_uses=("mating", "initial")`.

### `ValueError: prior_uses has 'mating', but the mating strategy uses no information gain`

`UnionMating` and `IntersectionMating` have no MI vector for the prior to
reweight. Drop `"mating"` from `prior_uses`, or use an information-gain
mating strategy.

### `FitBudgetExhausted` out of `mainloop`

`max_model_fits` was spent before the first generation, while the initial
population (or a Boruta seed) was being evaluated. Raise the budget above the
number of initial subsets plus `boruta_max_iter`.

### `AssertionError: Some sort of initial population must be provided`

Give `initial_genes`, `initial_population`, a `seed_filter`, or a feature
prior with `"initial"` in `prior_uses`.

### `IndexError` or wrong scores in the evaluator during pruning

`shuffle_indices` are positions among the *selected* columns of
`X_val[:, genes]`, not original feature indices.

## Behaviour that surprises

### The front is optimistic

Every front score is the best of many subsets compared on the same
validation split, so it overstates how the subset will do on new data. The
gap grows with the number of subsets tried. Keep a test split the search
never sees, choose a subset by a rule (for example the smallest within ε of
the best), and score it there once. On small data, repeat the whole search
over several splits.

### The front can skip sizes

A size is missing when no subset of that size beat a smaller one. That is the
point of a Pareto front.

### A larger subset with an equal score is dropped

Ties go to the smaller subset.

### Every generation prints `Round: N`

`mainloop` prints this unconditionally. To silence it:

```python
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    optimizer.mainloop()
```

### Runs with the same seed differ

`Config(random_seed=...)` seeds numpy's global generator when the `Config` is
**created**. Anything that draws from `np.random` between then and the end of
`mainloop` changes the run. That includes another `Config`, an evaluator
shuffling with `np.random` in some other order, or a model that uses the
global generator. Create the `Config` right before the optimizer and draw
only through `random_utils`. Importing `fastener` also seeds the generator
(through `dummy_data`), so import it before your own seeding.

### Runs with old and new defaults are not comparable

The mating defaults changed in 2026 (see [CHANGELOG.md](../CHANGELOG.md)). To
reproduce the 2020 results, set `allow_self_mating=True` and
`include_top_genes=True`.

### `reset_to_front_predicate` does nothing

It is accepted for compatibility, but the pruning cadence is
`Config.reset_to_pareto_rounds`.

### With `reset_to_pareto_rounds=None`, pruning hardly happens

`None` and `0` mean "every `number_of_rounds` rounds", i.e. once, at the end.
With a large `number_of_rounds` and a fit budget that ends the run first, the
front is never pruned.

### The log folder grows large

Each `generation_N.pickle` holds the population, the front and the whole
fitness cache. On thousands of features that is megabytes per generation. Set
`dump_generation_logs=False` unless you read them. `experiment.pickle` also
contains the training data.

## Questions

**Classification or regression?** Both. See [regression.md](regression.md).

**Which model?** Any estimator with `fit` and `predict`. A decision tree is
fast and is what the paper used. Each fit is the unit of cost, so a slow
model calls for a `max_model_fits` budget rather than many rounds.

**Pandas?** Pass `df.to_numpy()` and keep `df.columns` to name the selected
features.

**Wide data (thousands of features)?** Starting from every single feature
costs one fit per feature. Start from a filter seed (`seed_filter="mrmr"`) or
a feature prior instead, set `dump_generation_logs=False`, and stop on
`max_model_fits`.

**Cross-validation inside the search?** The optimizer fits on `train_data`
and hands the fitted model to your evaluator, so the evaluator scores one
split. To use another protocol, subclass `EntropyOptimizer` and override
`train_model(genes)`, keeping in mind that the evaluator must still accept
`shuffle_indices` for the permutation step.

**How many rounds?** Watch the best front score per generation (see
[examples/05_budget_outputs_and_logs.py](examples/05_budget_outputs_and_logs.py)).
When comparing settings, fix `max_model_fits` and set `number_of_rounds` high,
so that the budget, not the round count, ends every run.
