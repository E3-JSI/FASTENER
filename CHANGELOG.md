# Changelog

Changes to FASTENER since the version published in *Entropy* (2020). For the
paper, see the Citation section of [README.md](README.md).

## 2026-10-29 — Outside guidance: a feature prior and a subset proposer (off by default)

Two hooks let the caller guide the search with knowledge FASTENER does not
have -- a language model, a domain expert's ranking, another selector. Both
live in `guidance.py`; with neither, the search is unchanged.

**Feature prior.** One non-negative relevance score per feature, as an array
or as a callable that receives a `GuidanceContext` (feature names, task,
class names, free metadata such as feature descriptions) and returns the
scores. It is resolved once, before the search, and combined with mutual
information as `w = MI^(1 - lambda) * prior^lambda` (each scaled to a maximum
of 1 and floored at 0.001, so the prior biases rather than forbids).
`prior_uses` chooses what it guides:

- `"mating"`: `w` replaces the information-gain vector the mating strategy
  sorts and samples by;
- `"mutation"`: `guidance.PriorGuidedMutationStrategy` adds features in
  proportion to `w` and drops them in proportion to `max(w) - w`, at the
  uniform mutation's average rate; without weights it is the uniform mutation,
  draw for draw;
- `"initial"`: `prior_initial_singles` single features are drawn from `w`
  and added to the initial population.

```python
optimizer = EntropyOptimizer(
    ..., PriorGuidedMutationStrategy(1 / n_features),
    feature_prior=scores_or_function, prior_weight=0.5,
    prior_uses=("mating", "mutation", "initial"), prior_initial_singles=5,
    guidance_context=GuidanceContext(n_features, names, task="..."))
```

After `prepare_loop`, `prior_scores`, `feature_weights` and `prior_singles`
hold what was used.

**Subset proposer.** A callable that receives a `ProposalContext` -- the
generation, the current front (features and validation score per size), the
fits spent and the budget, and the weights if there is a prior -- and returns
subsets to try. It is called every `proposer_every` generations (default 5),
after the front update; its subsets are evaluated like any candidate, from
the same fit budget, and join the population. Invalid indices, empty and
duplicate subsets are dropped. Every call, what it proposed, what was
evaluated and the fits it cost are kept in `proposal_log`.

**Pickling.** The optimizer's pickled state leaves out a callable prior and
the proposer, which may hold network clients; their results stay.

## 2026-10-29 — Filter seeds (off by default)

FASTENER can start from a filter's ranking instead of only the `initial_genes`
it is given: `seed_filter="mrmr"` or `seed_filter="boruta"` adds the filter's
top-k nested subsets -- the best feature, the best two, ..., the best
`seed_size` (default 10) -- to the initial population. Subsets already in
`initial_genes` are not added twice. `seed_filter=None`, the default, runs the
search unchanged.

```python
# mRMR (MID): relevance minus mean redundancy to the features already chosen.
optimizer = EntropyOptimizer(..., seed_filter="mrmr")

# Boruta on the optimizer's own model; needs `pip install Boruta`.
optimizer = EntropyOptimizer(..., seed_filter="boruta", seed_size=10,
                             boruta_max_iter=100)
```

After `prepare_loop` (or `mainloop`), `seed_ranking` holds the filter's first
`seed_size` features and `seed_fits` the model fits the filter cost.

**Cost.** Each seed subset is one model fit, as for any initial gene. mRMR
fits no model of its own. Boruta fits the model once per iteration, and those
fits are added to `model_fits`, so a run capped by `Config.max_model_fits`
pays for its seed out of the same budget.

**Model.** Boruta ranks by the optimizer's model, which must expose
`feature_importances_` (a decision tree or forest does). Each iteration's
`random_state` from Boruta is passed on to the model if it takes one. Both
filters use `Config.random_seed` as their random state.

**Details.** mRMR's relevance is scikit-learn's `mutual_info_classif`; its
redundancy is the mutual information between features binned into 10
equal-width bins. Boruta ranks every confirmed feature 1 and every tentative
one 2, so ties are broken on mean shadow-relative importance. Without that, a
top-k prefix inside the confirmed block would be an arbitrary choice from it.
The filters live in `seeding.py` and can be used on their own:
`rank_mrmr(X, y, k)`, `rank_boruta(model_factory, X, y)` and
`nested_prefixes(order, k)`.

## 2026-09-25 — Swap operator and fit-capped stopping

Adds guided and random swap strategies to the pruning step, and an optional
cap on model fits. Both are off by default, so the published algorithm runs
unchanged.

### Optional: swap operator in the pruning step (off by default)

Pruning (`purge_front_with_information_gain`) identifies a weak feature by
permutation importance and tests *deleting* it. It never tests *replacing* it,
and no other operator preserves subset size -- mating pulls a child towards the
middle size, pruning strictly decreases it, and mutation holds the size constant
only when two bits flip at once. `SwapStrategy` is that missing operator: it
drops one selected feature and inserts one unselected feature in its place, so
the candidate competes directly against the front entry at its own size.

It is opt-in. `EntropyOptimizer(..., swap_strategy=None)` -- the default -- runs
the published algorithm unchanged.

```python
from item import InformationGainSwapStrategy, RandomSwapStrategy

# Guided: drop the permutation-weakest feature, insert one sampled with
# probability weighted by mutual information.
optimizer = EntropyOptimizer(..., swap_strategy=InformationGainSwapStrategy())

# Ablation: drop a random selected feature, insert a uniformly random one.
optimizer = EntropyOptimizer(..., swap_strategy=RandomSwapStrategy())

# More than one replacement candidate per pruned subset (one fit each).
optimizer = EntropyOptimizer(
    ..., swap_strategy=InformationGainSwapStrategy(number_of_swaps=3))
```

`InformationGainSwapStrategy` reuses the mutual-information vector the mating
strategy already computed -- `EntropyOptimizer.prepare_loop` hands it over -- so
the guidance adds no scoring pass. It falls back to computing its own vector if
the mating strategy has none (e.g. `UnionMating`).

**Cost.** One extra model *fit* per proposed candidate per pruned subset. The
permutation scores it uses are already computed by the pruning step, so nothing
else is added.

**Compatibility.** `purge_item_with_information_gain(item)` keeps its name,
signature and single-`EvalItem` return value; `purge_item_candidates(item)` is
the new entry point returning the deletion candidate first and any swap
candidates after it.

### Optional: fit-capped stopping (off by default)

`Config.max_model_fits` stops the search once that many models have been fitted,
instead of running all `number_of_rounds` generations. `None` (the default)
keeps the original round-limited behaviour.

```python
config = Config(output_folder="run", random_seed=2020,
                number_of_rounds=100000,   # high: let the budget be the limit
                max_model_fits=6000)
```

It exists because comparing two variants of the algorithm is only meaningful at
equal compute, and a variant that evaluates more candidates per generation is
otherwise silently given more search. The cap is checked inside
`_fitness_function` — the single place a model is fitted — so it is exact rather
than per-generation, and nothing can spend budget unmetered. Cache hits are not
counted (they cost nothing), so the budget is one of *unique* subsets; neither
is permutation scoring, which predicts but never fits.

After a run, `optimizer.model_fits` is what was spent, `optimizer.stop_reason`
is `"fit_budget"` or `"rounds"`, and `optimizer.rounds_completed` is how many
generations finished. A generation interrupted by the cap is abandoned, and the
front is re-pruned so it holds no dominated entry.

Note that the cap must be larger than the initial population, which is evaluated
before the first generation; a smaller one raises `FitBudgetExhausted` out of
`mainloop` rather than running a search with nothing in it.

## 2026-09-30 — Top genes kept by default

`IntersectionMatingWithWeightedRandomInformationGain` again always adds the
top-k genes by mutual information before its weighted sample:
`include_top_genes` now defaults to `True`, as in the 2020 code. That leaves
one change to the original mating behaviour, no self-mating in
`RandomEveryoneWithEveryone`. `include_top_genes=False` gives the sample-only
child described below. Runs with the 2026-09-25 defaults are not comparable
to runs with these.

## 2026-09-25 — Bug fixes in the mating step

### Changes to the original mating behaviour

Two defaults in the mating step now differ from the 2020 code that produced
the published results (see issue #1). **Runs with the new defaults are not
comparable to earlier results**: both the search and the random-number stream
change. The original behaviour is kept behind a flag for each:

- **Weighted information-gain mating no longer always adds the top genes.**
  `IntersectionMatingWithWeightedRandomInformationGain` used to start from its
  parent class's child: the intersection *plus* the top-k genes by mutual
  information. It then sampled k more on top of those. So the top-k were always
  included and a child got up to 2k genes back (1.3k on average), and the
  "weighted random" part rarely added anything new. It now starts from the
  plain intersection and adds only the weighted sample.
  `include_top_genes=True` restores the original.
- **An item is no longer mated with itself.** `RandomEveryoneWithEveryone` drew
  its mating pool with replacement, so the same item could be picked twice and
  paired with itself. The child was a copy of the parent, which took up a
  crossover slot (about 13% of pairs with `pool_size=3`). The pool is now drawn
  without replacement and is capped at the population size.
  `allow_self_mating=True` restores the original.

To reproduce results produced before this change, set both flags:

```python
mating = RandomEveryoneWithEveryone(
    pool_size=3,
    allow_self_mating=True,
    mating_strategy=IntersectionMatingWithWeightedRandomInformationGain(
        include_top_genes=True))
```

With both flags set, a run is identical to the original code: the same fits
and the same Pareto front for the same seed.

One bug fix needs no flag: `RandomEveryoneWithEveryone(pool_size=None)` is
meant to use the whole population as the mating pool. It used to crash, and
now works.
